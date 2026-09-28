#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoder_comparison_d3.py
Hybrid error management vs. pure error correction for the d=3 surface code
(Figure 6): p_local = 0.01, target acceptance 50%.
Produces two figures:
  (a) Decoder = MWPM (PyMatching): unfiltered vs SW + MWPM vs DS + MWPM
  (b) Decoder = NN:                unfiltered vs SW + NN   vs DS + NN
Caches results to CSV (default: cache_decoder_comparison) and only recomputes missing points;
if the cache is complete, the script only plots.

Usage:
  python decoder_comparison_d3.py            # compute (or read from the cache) and plot
  python decoder_comparison_d3.py --quick    # smoke test with 2000 shots per point (writes to smoke_test/fig6)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Settings
# ---------------------------
P_LOCAL = 0.01
P_TRANS_LIST = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
D = 3
NUM_LAYER = 1
TARGET_ACCEPT = 0.50
SHOTS_CAL = 150000
SHOTS_EVAL = 150000
SEED = 12345
QUICK_SHOTS = 2000

CACHEDIR = os.path.join(HERE, "cache_decoder_comparison")
OUTDIR = os.path.join(HERE, "figs_decoder_comparison")
SMOKE_DIR = os.path.join(HERE, "smoke_test", "fig6")

# NN options
NN_MODEL_PATH = os.path.join(HERE, "Model", f"model_d_{D}.pt")
NN_BATCH_SIZE = 512
NN_DEVICE = None
NN_CAL_SHOTS_CAP = 30_000

# Method names in the cache files
TAG = f"@{round(100 * TARGET_ACCEPT)}%"
M_SW_PM, M_DS_PM = f"SW{TAG}", f"ZCoset{TAG}"
M_SW_NN, M_DS_NN = "SW+NN", f"NNFilter{TAG}"

# Common y-axis of both panels
YLIM = (-0.01, 0.35)

# ---------------------------
# Imports from the project
# ---------------------------
from surface import State_Encoding
import pymatching as pm
from estimator_weight import SyndromeWeightEstimator
from estimator_z import ZCosetGapEstimator
from estimator_NN import NNFilterDecoder
from estimator_swNN import SWFilterNNDecoder


# ---------------------------
# Core helpers
# ---------------------------
def build_circuit_and_sampler(d, num_layer, p_local, p_trans):
    circuit = State_Encoding(d=d, num_layer=num_layer, p_local=p_local, p_trans=p_trans)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()
    return circuit, dem, sampler

def unconditional_ler_pymatching(sampler, matcher_std, shots):
    synd_eval, obs_eval = sampler.sample(shots, separate_observables=True)
    preds = np.asarray(matcher_std.decode_batch(synd_eval))
    y_true = obs_eval[:, 0].astype(np.uint8)
    # decode_batch returns shape (shots, num_observables): compare shot by shot on observable 0.
    y_pred = (preds.reshape(len(y_true), -1)[:, 0] % 2).astype(np.uint8)
    assert y_pred.shape == y_true.shape
    return float(np.mean(y_true != y_pred))

def unconditional_ler_nn(nndecoder, sampler, shots):
    synd_eval, obs_eval = sampler.sample(shots, separate_observables=True)
    event_tensor = nndecoder._fill_event_tensor(synd_eval)
    p = nndecoder._predict_soft(event_tensor)
    y_hat = (p > 0.5).astype(np.uint8).reshape(-1)
    y_true = obs_eval[:, 0].astype(np.uint8)
    return float(np.mean(y_true != y_hat))

def make_nn(cls, circuit):
    return cls(circuit=circuit, model_path=NN_MODEL_PATH, d=D, num_layer=NUM_LAYER,
               batch_size=NN_BATCH_SIZE, device=NN_DEVICE, cal_shots_cap=NN_CAL_SHOTS_CAP)

# ---------------------------
# Cache I/O
# ---------------------------
CACHE_COLS = ["decoder", "method", "p_local", "p_trans", "ler", "accept", "meta"]
PURE_COLS = ["p_trans", "ler", "stderr"]

def cache_paths(cachedir):
    return (os.path.join(cachedir, f"near_term_pymatching_d{D}_plocal_{P_LOCAL:.3f}.csv"),
            os.path.join(cachedir, f"pure_mwpm_d{D}_plocal{P_LOCAL}.csv"),
            os.path.join(cachedir, f"near_term_nn_d{D}_plocal_{P_LOCAL:.3f}.csv"))

def load_csv(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    return df

def row_mask(df, decoder, method, p_trans):
    return ((df["decoder"] == decoder) &
            (df["method"] == method) &
            np.isclose(df["p_local"].astype(float), P_LOCAL) &
            np.isclose(df["p_trans"].astype(float), float(p_trans)))

def have_row(df, decoder, method, p_trans):
    return bool(row_mask(df, decoder, method, p_trans).any())

def upsert_row(df, row):
    df = df.loc[~row_mask(df, row["decoder"], row["method"], row["p_trans"])]
    parts = [df] if len(df) else []
    return pd.concat(parts + [pd.DataFrame([row])], ignore_index=True)

def save_cache(df, path):
    df = df.sort_values(["decoder", "method", "p_trans"]).reset_index(drop=True)
    df.to_csv(path, index=False)

def get_ler(df, decoder, method):
    return [float(df.loc[row_mask(df, decoder, method, pT), "ler"].iloc[0]) for pT in P_TRANS_LIST]

# ---------------------------
# Filtered rows: calibrate on SHOTS_CAL, evaluate on SHOTS_EVAL
# ---------------------------
def filtered_row(est, sampler, decoder, method, pT, **meta):
    synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
    tau = est.quantile_threshold(est.score_batch(synd_cal), TARGET_ACCEPT)
    synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
    acc_rate, condL = est.evaluate_batch(synd_eval, obs_eval, tau)
    return {"decoder": decoder, "method": method, "p_local": P_LOCAL, "p_trans": pT,
            "ler": condL, "accept": acc_rate,
            "meta": json.dumps({"shots_cal": SHOTS_CAL, "shots_eval": SHOTS_EVAL, "tau": float(tau), **meta})}

# ---------------------------
# (a) MWPM: compute + cache
# ---------------------------
def compute_pure_pm(csv_pure):
    df = load_csv(csv_pure, PURE_COLS)
    new_rows = []
    for pT in P_TRANS_LIST:
        if np.isclose(df["p_trans"].astype(float), pT).any():
            continue
        circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
        matcher_std = pm.Matching.from_detector_error_model(dem)
        ler = unconditional_ler_pymatching(sampler, matcher_std, SHOTS_EVAL)
        new_rows.append({"p_trans": pT, "ler": ler, "stderr": float(np.sqrt(ler * (1 - ler) / SHOTS_EVAL))})
    if new_rows:
        parts = [df] if len(df) else []
        df = pd.concat(parts + [pd.DataFrame(new_rows)], ignore_index=True)
        df.sort_values("p_trans").to_csv(csv_pure, index=False)
    return [float(df.loc[np.isclose(df["p_trans"].astype(float), pT), "ler"].iloc[0]) for pT in P_TRANS_LIST]

def compute_pm_curve(csv_pm):
    df = load_csv(csv_pm, CACHE_COLS)
    updated = False
    for pT in P_TRANS_LIST:
        if not have_row(df, "PM", M_SW_PM, pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            est = SyndromeWeightEstimator(circuit, dem=dem)
            df = upsert_row(df, filtered_row(est, sampler, "PM", M_SW_PM, pT))
            updated = True

        if not have_row(df, "PM", M_DS_PM, pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            est = ZCosetGapEstimator(circuit, dem=dem)
            df = upsert_row(df, filtered_row(est, sampler, "PM", M_DS_PM, pT))
            updated = True

    if updated:
        save_cache(df, csv_pm)
    return get_ler(df, "PM", M_SW_PM), get_ler(df, "PM", M_DS_PM)

# ---------------------------
# (b) NN: compute + cache
# ---------------------------
def compute_nn_curve(csv_nn):
    df = load_csv(csv_nn, CACHE_COLS)
    updated = False
    for pT in P_TRANS_LIST:
        if not have_row(df, "NN", "unfiltered", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = make_nn(NNFilterDecoder, circuit)
            ler = unconditional_ler_nn(nnd, sampler, SHOTS_EVAL)
            df = upsert_row(df, {"decoder": "NN", "method": "unfiltered", "p_local": P_LOCAL, "p_trans": pT,
                                 "ler": ler, "accept": 1.0,
                                 "meta": json.dumps({"shots_eval": SHOTS_EVAL, "finetuned": False})})
            updated = True

        # SW filter, NN decoder on the accepted shots
        if not have_row(df, "NN", M_SW_NN, pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = make_nn(SWFilterNNDecoder, circuit)
            df = upsert_row(df, filtered_row(nnd, sampler, "NN", M_SW_NN, pT, finetuned=False))
            updated = True

        # DS filter (NN confidence), NN decoder
        if not have_row(df, "NN", M_DS_NN, pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = make_nn(NNFilterDecoder, circuit)
            df = upsert_row(df, filtered_row(nnd, sampler, "NN", M_DS_NN, pT, finetuned=False))
            updated = True

    if updated:
        save_cache(df, csv_nn)
    return get_ler(df, "NN", "unfiltered"), get_ler(df, "NN", M_SW_NN), get_ler(df, "NN", M_DS_NN)

# ---------------------------
# Plotters
# ---------------------------
def plot_panel(x, y_unf, y_sw, y_ds, labels, colors, path):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y_unf, color=colors[0], marker='o', linewidth=2.5, linestyle='--', label=labels[0])
    plt.plot(x, y_sw, color=colors[1], marker='s', linewidth=2.5, label=labels[1])
    plt.plot(x, y_ds, color=colors[2], marker='^', linewidth=2.5, label=labels[2])

    # --- break-even line: an unencoded |0> read out in Z fails with probability 2 p_trans / 3 ---
    p_line = np.linspace(min(x), max(x), 100)
    plt.plot(p_line, 2 * p_line / 3, color='gray', linestyle=':', linewidth=2,
             label=r"Break-even line ($2p_{\mathrm{trans}}/3$)")

    plt.xticks(x)
    plt.ylim(*YLIM)
    plt.yticks(np.arange(0.0, YLIM[1] + 1e-9, 0.05))
    plt.xlabel("Transmission Error Rate", fontweight="bold")
    plt.ylabel("Logical Error Rate", fontweight="bold")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.03)
    plt.close()

def plot_pm(x, y_unf, y_sw, y_ds, path):
    plot_panel(x, y_unf, y_sw, y_ds, ["Pure MWPM Decoder", "SW + MWPM", "DS + MWPM"],
               ["#8B0000", "#0073E6", "#FF3333"], path)

def plot_nn(x, y_unf, y_sw, y_ds, path):
    plot_panel(x, y_unf, y_sw, y_ds, ["Pure NN Decoder", "SW + NN", "DS + NN"],
               ["#006400", "#e88800", "#37A546"], path)

# ---------------------------
# Main
# ---------------------------
def main():
    global SHOTS_CAL, SHOTS_EVAL, NN_CAL_SHOTS_CAP
    ap = argparse.ArgumentParser(description="Figure 6: d=3 decoder comparison (p_local=0.01, target acceptance 50%).")
    ap.add_argument("--cache_dir", default=None, help="CSV cache directory (default: cache_decoder_comparison)")
    ap.add_argument("--out_dir", default=None, help="output directory for the PDFs (default: figs_decoder_comparison)")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test: {QUICK_SHOTS} shots per point; cache and figures go to smoke_test/fig6")
    args = ap.parse_args()

    cachedir, outdir = CACHEDIR, OUTDIR
    if args.quick:
        SHOTS_CAL = SHOTS_EVAL = NN_CAL_SHOTS_CAP = QUICK_SHOTS
        cachedir = outdir = SMOKE_DIR
    cachedir = args.cache_dir or cachedir
    outdir = args.out_dir or outdir
    os.makedirs(cachedir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(SEED)

    csv_pm, csv_pure, csv_nn = cache_paths(cachedir)
    y_unf_pm = compute_pure_pm(csv_pure)
    y_sw_pm, y_ds_pm = compute_pm_curve(csv_pm)
    y_unf_nn, y_sw_nn, y_ds_nn = compute_nn_curve(csv_nn)

    f1 = os.path.join(outdir, "decoder_compare_PM.pdf")
    plot_pm(P_TRANS_LIST, y_unf_pm, y_sw_pm, y_ds_pm, f1)
    f2 = os.path.join(outdir, "decoder_compare_NN.pdf")
    plot_nn(P_TRANS_LIST, y_unf_nn, y_sw_nn, y_ds_nn, f2)

    print("Saved figures:")
    print(" ", f1)
    print(" ", f2)
    print("Caches:")
    for p in (csv_pure, csv_pm, csv_nn):
        print(" ", p)

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    main()
