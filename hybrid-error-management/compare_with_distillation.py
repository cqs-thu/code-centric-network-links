#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_with_distillation.py
Hybrid filters vs. three-layer distillation at matched acceptance (Figure 7).

For each p_local in {0.005, 0.01} and each p_trans in KEEP_PTS:
  - compute the success rate and logical error rate of three-layer distillation with
    cal_DEJMPS_detection_per_basis (three-protocols/cals/cal_logical_error.py): the output parities are
    read out without noise, and the logical error rate is the average of the ZZ- and XX-parity errors.
    The surface code is symmetric, so its Z_L readout error of |0>_L also gives its X-basis error; the
    distillation protocol is not, so it enters through the average over the two bases
  - use the distillation success rate as the target acceptance of the hybrid filters
  - measure the conditional logical error rate of
      * SW + MWPM  (SyndromeWeightEstimator)
      * DS + MWPM  (ZCosetGapEstimator)
      * DS + NN    (NNFilterDecoder)
  - plot the three filters and the distillation logical error rate against p_trans.
Caches results to CSV (default: cache_distill_compare) and only recomputes missing points;
if the cache is complete, the script only plots.

Usage:
  python compare_with_distillation.py           # compute (or read from the cache) and plot
  python compare_with_distillation.py --quick   # smoke test with 2000 shots per point (writes to smoke_test/fig7)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Settings
# ---------------------------
D = 3
NUM_LAYER = 1
P_LOCAL_LIST = [0.005, 0.01]
KEEP_PTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]  # p_trans values shown in the figure
SHOTS_CAL = 250_000
SHOTS_EVAL = 250_000
DISTILL_SHOTS = 10_000_000  # shots per point of the distillation baseline
SEED = 12345
QUICK_SHOTS = 2000

THREE_PROTOCOLS = os.path.join(HERE, "..", "three-protocols")
CACHEDIR = os.path.join(HERE, "cache_distill_compare")
OUTDIR = os.path.join(HERE, "figs_distill_compare")
SMOKE_DIR = os.path.join(HERE, "smoke_test", "fig7")

# NN options
NN_MODEL_PATH = os.path.join(HERE, "Model", f"model_d_{D}.pt")
NN_BATCH_SIZE = 512
NN_DEVICE = None
NN_CAL_SHOTS_CAP = 250000

# Colors
CLR_DISTILL = "#4B0082"  # dark purple
CLR_SW      = "#0073E6"  # blue
CLR_ZG      = "#FF3333"  # red
CLR_NN      = "#37A546"  # green

# ---------------------------
# Imports from the project
# ---------------------------
from surface import State_Encoding
from estimator_weight import SyndromeWeightEstimator
from estimator_z import ZCosetGapEstimator
from estimator_NN import NNFilterDecoder

# ---------------------------
# Helpers
# ---------------------------
def cache_path(cachedir, p_local):
    return os.path.join(cachedir, f"hybrid_vs_distill_d{D}_plocal_{p_local:.3f}.csv")

DISTILL_COLS = ["p_local", "p_transmission", "distill_success_rate", "distill_zz", "distill_xx", "distill_logical_error"]

def distillation_baseline(cachedir):
    """
    Success rate and logical error rate of three-layer distillation at every (p_local, p_trans)
    of the figure. Missing points are computed with cal_DEJMPS_detection_per_basis and cached in
    <cachedir>/distillation_basis_avg_cache.csv.
    """
    path = os.path.join(cachedir, "distillation_basis_avg_cache.csv")
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=DISTILL_COLS)
    missing = [(pl, pT) for pl in P_LOCAL_LIST for pT in KEEP_PTS
               if not (np.isclose(df["p_local"].astype(float), pl) &
                       np.isclose(df["p_transmission"].astype(float), pT)).any()]
    if missing:
        sys.path.insert(0, os.path.abspath(THREE_PROTOCOLS))
        from cals.cal_logical_error import cal_DEJMPS_detection_per_basis
        rows = []
        for pl, pT in missing:
            acc, zz, xx = cal_DEJMPS_detection_per_basis(depth=3, p_local=pl, p_trans=pT, shots=DISTILL_SHOTS)
            rows.append({"p_local": pl, "p_transmission": pT, "distill_success_rate": acc,
                         "distill_zz": zz, "distill_xx": xx, "distill_logical_error": (zz + xx) / 2})
        parts = [df] if len(df) else []
        df = pd.concat(parts + [pd.DataFrame(rows)], ignore_index=True)
        df = df.sort_values(["p_local", "p_transmission"]).reset_index(drop=True)
        df.to_csv(path, index=False)
    return df

def build_circuit_and_sampler(d, num_layer, p_local, p_trans):
    circuit = State_Encoding(d=d, num_layer=num_layer, p_local=p_local, p_trans=p_trans)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()
    return circuit, dem, sampler

CACHE_COLS = ["method", "p_local", "p_trans", "accept", "ler", "meta"]

def load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=CACHE_COLS)
    df = pd.read_csv(path)
    for col in CACHE_COLS:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    return df

def row_mask(df, method, p_local, p_trans, accept):
    return ((df["method"] == method) &
            np.isclose(df["p_local"].astype(float), float(p_local)) &
            np.isclose(df["p_trans"].astype(float), float(p_trans)) &
            np.isclose(df["accept"].astype(float), float(accept)))

def have_row(df, method, p_local, p_trans, accept):
    return bool(row_mask(df, method, p_local, p_trans, accept).any())

def upsert_row(df, row):
    df = df.loc[~row_mask(df, row["method"], row["p_local"], row["p_trans"], row["accept"])]
    parts = [df] if len(df) else []
    return pd.concat(parts + [pd.DataFrame([row])], ignore_index=True)

def save_cache(df, path):
    df = df.sort_values(["method", "p_trans"]).reset_index(drop=True)
    df.to_csv(path, index=False)

def filtered_row(est, sampler, method, p_local, pT, acc, **meta):
    """Calibrate the threshold for acceptance `acc` on SHOTS_CAL shots, evaluate on SHOTS_EVAL shots."""
    synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
    tau = est.quantile_threshold(est.score_batch(synd_cal), acc)
    synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
    acc_emp, condL = est.evaluate_batch(synd_eval, obs_eval, tau)
    return {"method": method, "p_local": p_local, "p_trans": pT, "accept": acc, "ler": condL,
            "meta": json.dumps({"shots_cal": SHOTS_CAL, "shots_eval": SHOTS_EVAL,
                                "tau": float(tau), "accept_emp": float(acc_emp), **meta})}

# ---------------------------
# Core: hybrid curves at the distillation acceptance
# ---------------------------
def compute_for_plocal(df_distill, p_local, cachedir):
    """
    For a fixed p_local, read p_trans, distill_success_rate (used as target acceptance) and
    distill_logical_error from the distillation CSV, compute the SW / ZCoset / NN conditional
    logical error rates at the same acceptance and cache them.
    Returns: p_trans list, dict of series {"Distill", "SW", "ZCoset", "NN"}, acceptance list.
    """
    path = cache_path(cachedir, p_local)
    cache = load_cache(path)
    sub = df_distill[np.isclose(df_distill["p_local"].astype(float), float(p_local))]
    sub = sub.sort_values("p_transmission")
    pT_list = sub["p_transmission"].astype(float).tolist()
    accept_list = sub["distill_success_rate"].astype(float).tolist()
    y_distill = sub["distill_logical_error"].astype(float).tolist()

    # Clamp accept into the open interval (0, 1)
    acc_list = [float(np.clip(a, 1e-6, 1.0 - 1e-6)) for a in accept_list]

    updated = False
    for pT, acc in zip(pT_list, acc_list):
        # SW + MWPM
        if not have_row(cache, "SW", p_local, pT, acc):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, p_local, pT)
            est = SyndromeWeightEstimator(circuit, dem=dem)
            cache = upsert_row(cache, filtered_row(est, sampler, "SW", p_local, pT, acc))
            updated = True

        # DS + MWPM
        if not have_row(cache, "ZCoset", p_local, pT, acc):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, p_local, pT)
            est = ZCosetGapEstimator(circuit, dem=dem)
            cache = upsert_row(cache, filtered_row(est, sampler, "ZCoset", p_local, pT, acc))
            updated = True

        # DS + NN
        if not have_row(cache, "NN", p_local, pT, acc):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, p_local, pT)
            nnd = NNFilterDecoder(
                circuit=circuit, model_path=NN_MODEL_PATH, d=D, num_layer=NUM_LAYER,
                batch_size=NN_BATCH_SIZE, device=NN_DEVICE, cal_shots_cap=NN_CAL_SHOTS_CAP
            )
            cache = upsert_row(cache, filtered_row(nnd, sampler, "NN", p_local, pT, acc, finetuned=False))
            updated = True

    if updated:
        save_cache(cache, path)

    def series(method):
        vals = []
        for pT, acc in zip(pT_list, acc_list):
            v = cache[row_mask(cache, method, p_local, pT, acc)]
            vals.append(float(v.iloc[-1]["ler"]) if len(v) else np.nan)
        return vals

    curves = {"Distill": y_distill, "SW": series("SW"), "ZCoset": series("ZCoset"), "NN": series("NN")}
    return pT_list, curves, accept_list

# ---------------------------
# Plotting
# ---------------------------
def plot_compare(x_vals, curves, accepts, output_filename):
    # keep only the selected transmission points
    mask = [np.isclose(x, KEEP_PTS).any() for x in x_vals]
    x_vals = [x for x, m in zip(x_vals, mask) if m]
    curves = {key: [y for y, m in zip(ys, mask) if m] for key, ys in curves.items()}
    accepts = [a for a, m in zip(accepts, mask) if m]

    plt.figure(figsize=(6.5, 4.5))
    # Baseline (distillation)
    plt.plot(x_vals, curves["Distill"], color=CLR_DISTILL, linestyle='--', marker='o', linewidth=2.5,
             label="3-layer Distillation")
    # Hybrid filters
    plt.plot(x_vals, curves["SW"],     color=CLR_SW, marker='s', linewidth=2.5, label="SW + MWPM")
    plt.plot(x_vals, curves["ZCoset"], color=CLR_ZG, marker='^', linewidth=2.5, label="DS + MWPM")
    plt.plot(x_vals, curves["NN"],     color=CLR_NN, marker='D', linewidth=2.5, label="DS + NN")

    # Acceptance rate (distillation success rate) above each x position
    ax = plt.gca()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, a in zip(x_vals, accepts):
        ax.text(x, 1.01, f"{a:.2f}", transform=trans, ha='center', va='bottom', fontsize=14, color='black')
    ax.text(0.5, 1.09, "Acceptance Rate", transform=ax.transAxes, ha='center', va='bottom', fontsize=14,
            color='black', fontweight="bold")

    plt.yscale('log')
    plt.xticks(x_vals)
    plt.xlabel("Transmission Error Rate", fontweight="bold")
    plt.ylabel("Logical Error Rate", fontweight="bold")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.03)
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    global SHOTS_CAL, SHOTS_EVAL, NN_CAL_SHOTS_CAP, DISTILL_SHOTS
    ap = argparse.ArgumentParser(description="Figure 7: hybrid filters vs. three-layer distillation at matched acceptance.")
    ap.add_argument("--cache_dir", default=None, help="CSV cache directory (default: cache_distill_compare)")
    ap.add_argument("--out_dir", default=None, help="output directory for the PDFs (default: figs_distill_compare)")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test: {QUICK_SHOTS} shots per point; cache and figures go to smoke_test/fig7")
    args = ap.parse_args()

    cachedir, outdir = CACHEDIR, OUTDIR
    if args.quick:
        SHOTS_CAL = SHOTS_EVAL = NN_CAL_SHOTS_CAP = DISTILL_SHOTS = QUICK_SHOTS
        cachedir = outdir = SMOKE_DIR
    cachedir = args.cache_dir or cachedir
    outdir = args.out_dir or outdir
    os.makedirs(cachedir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(SEED)

    df = distillation_baseline(cachedir)

    for p_local in P_LOCAL_LIST:
        x_vals, curves, accepts = compute_for_plocal(df, p_local, cachedir)
        fig_path = os.path.join(outdir, f"distill_vs_hybrid_d{D}_plocal_{p_local:.3f}.pdf")
        plot_compare(x_vals, curves, accepts, fig_path)
        print("Saved:", fig_path)
        print("Cache:", cache_path(cachedir, p_local))

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
