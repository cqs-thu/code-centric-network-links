#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
near_term_smallcode.py
Fixed small-code experiment for d=3 (near-term).
Produces two figures:
  (1) Decoder = PyMatching: unfiltered vs SW@50% vs ZCoset@50%
  (2) Decoder = NN: unfiltered (hard) vs NNFilter@50%
Caches results to CSV and only recomputes missing points.

Assumes project modules on PYTHONPATH:
  - surface.State_Encoding
  - estimator_weight.SyndromeWeightEstimator
  - estimator_z.ZCosetGapEstimator
  - estimator_NN.NNFilterDecoder
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# User-tunable knobs
# ---------------------------
CAL_NUM = 100000
NUM_TRIALS = 2          # 2 x SHOTS_EVAL = 2e6 evaluation shots per point (figure 8)
P_LOCAL_LIST = [0.01, 0.005, 0.001]
P_TRANS_LIST = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.25]
D = 3
NUM_LAYER = 1
TARGET_ACCEPT_LIST = [0.20, 0.50, 0.80]
SHOTS_CAL = CAL_NUM
SHOTS_EVAL = 10*CAL_NUM
SEED = 12345

OUTDIR = "figs_nearterm"
CACHEDIR = "cache_nearterm"

# NN options
NN_MODEL_DIR = "./Model"
NN_MODEL_PATH = os.path.join(NN_MODEL_DIR, f"model_d_{D}.pt")
NN_BATCH_SIZE = 512
NN_DEVICE = None
NN_CAL_SHOTS_CAP = 10*CAL_NUM
DO_FINETUNE = False
FT_SHOTS = 80000
FT_EPOCHS = 1

# ---------------------------
# Imports from your project
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
    # decode_batch returns shape (shots, num_observables); compare shot by shot.
    y_pred = (preds.reshape(len(y_true), -1)[:, 0] % 2).astype(np.uint8)
    assert y_pred.shape == y_true.shape
    return float(np.mean(y_true != y_pred))

def unconditional_ler_nn(nndecoder, sampler, shots):
    synd_eval, obs_eval = sampler.sample(shots, separate_observables=True)
    # Convert numpy syndromes to torch tensor (on CPU, float32)
    event_tensor = nndecoder._fill_event_tensor(synd_eval)  # use the NN’s own converter
    p = nndecoder._predict_soft(event_tensor)                # expects torch.Tensor
    y_hat = (p > 0.5).astype(np.uint8).reshape(-1)
    y_true = obs_eval[:, 0].astype(np.uint8)
    return float(np.mean(y_true != y_hat))

def unconditional_ler_pymatching_avg(sampler, matcher_std, shots, num_trials):
    vals = []
    for _ in range(num_trials):
        vals.append(unconditional_ler_pymatching(sampler, matcher_std, shots))
    return float(np.mean(vals))

def unconditional_ler_nn_avg(nndecoder, sampler, shots, num_trials):
    vals = []
    for _ in range(num_trials):
        vals.append(unconditional_ler_nn(nndecoder, sampler, shots))
    return float(np.mean(vals))

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CACHEDIR, exist_ok=True)

def _round(x, nd=12):
    return float(np.round(float(x), nd))

# ---------------------------
# Cache I/O
# ---------------------------
def load_cache(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # basic schema sanity
        for col in ["decoder","method","p_trans","ler","accept"]:
            if col not in df.columns:
                raise ValueError(f"{path} missing column: {col}")
        return df
    else:
        return pd.DataFrame(columns=["decoder","method","p_trans","ler","accept"])

def save_cache(df, path):
    df = df.sort_values(["decoder","method","p_trans"]).reset_index(drop=True)
    df.to_csv(path, index=False)

def have_row(df, decoder, method, p_trans):
    mask = (
        (df["decoder"] == decoder) &
        (df["method"] == method) &
        (np.isclose(df["p_trans"].astype(float), float(p_trans)))
    )
    return bool(mask.any())

def upsert_row(df, row):
    # drop exact key then append
    mask = (
        (df["decoder"] == row["decoder"]) &
        (df["method"] == row["method"]) &
        (np.isclose(df["p_trans"].astype(float), float(row["p_trans"])))
    )
    df = df.loc[~mask].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

# ---------------------------
# PM figure: compute + cache
# ---------------------------
def compute_pure_pm(P_LOCAL = 0.01, CSV_PM = None):
    df = load_cache(CSV_PM)
    y_unf = []
    for pT in P_TRANS_LIST:
        if not have_row(df,"PM","unfiltered",pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            matcher_std = pm.Matching.from_detector_error_model(dem)
            ler_avg = unconditional_ler_pymatching_avg(sampler, matcher_std, SHOTS_EVAL, NUM_TRIALS)
            row = {"decoder":"PM","method":"unfiltered","p_trans":pT,"ler":ler_avg,"accept":1.0}
            df = upsert_row(df, row)
    save_cache(df, CSV_PM)
    dfp = df
    y_unf = [float(dfp[(dfp.decoder=="PM")&(dfp.method=="unfiltered")&(np.isclose(dfp.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    return y_unf

def compute_pm_curve(P_LOCAL = 0.01, TARGET_ACCEPT = 0.50, CSV_PM = None):
    df = load_cache(CSV_PM)
    y_swpm, y_dspm = [], []
    for pT in P_TRANS_LIST:
        if not have_row(df,"PM",f"SW+{TARGET_ACCEPT}", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            est = SyndromeWeightEstimator(circuit, dem=dem)
            synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
            tau = est.quantile_threshold(est.score_batch(synd_cal), TARGET_ACCEPT)
            acc_vals, ler_vals = [], []
            for _ in range(NUM_TRIALS):
                synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
                acc_rate, condL = est.evaluate_batch(synd_eval, obs_eval, tau)
                acc_vals.append(acc_rate); ler_vals.append(condL)
            row = {"decoder":"PM","method":f"SW+{TARGET_ACCEPT}","p_trans":pT,
                   "ler":float(np.mean(ler_vals)), "accept":float(np.mean(acc_vals))}
            df = upsert_row(df, row)

        if not have_row(df,"PM",f"DS+{TARGET_ACCEPT}", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            est = ZCosetGapEstimator(circuit, dem=dem)
            synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
            tau = est.quantile_threshold(est.score_batch(synd_cal), TARGET_ACCEPT)
            acc_vals, ler_vals = [], []
            for _ in range(NUM_TRIALS):
                synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
                acc_rate, condL = est.evaluate_batch(synd_eval, obs_eval, tau)
                acc_vals.append(acc_rate); ler_vals.append(condL)
            row = {"decoder":"PM","method":f"DS+{TARGET_ACCEPT}","p_trans":pT,
                   "ler":float(np.mean(ler_vals)), "accept":float(np.mean(acc_vals))}
            df = upsert_row(df, row)

    save_cache(df, CSV_PM)
    dfp = df
    y_swpm = [float(dfp[(dfp.decoder=="PM")&(dfp.method==f"SW+{TARGET_ACCEPT}")&(np.isclose(dfp.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    y_dspm = [float(dfp[(dfp.decoder=="PM")&(dfp.method==f"DS+{TARGET_ACCEPT}")&(np.isclose(dfp.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    return y_swpm, y_dspm

# ---------------------------
# NN figure: compute + cache
# ---------------------------

def compute_pure_nn(P_LOCAL = 0.01, CSV_NN = None):
    df = load_cache(CSV_NN)
    y_unf = []
    for pT in P_TRANS_LIST:
        if not have_row(df,"NN","unfiltered", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = NNFilterDecoder(
                circuit=circuit, model_path=NN_MODEL_PATH, d=D, num_layer=NUM_LAYER,
                batch_size=NN_BATCH_SIZE, device=NN_DEVICE, cal_shots_cap=NN_CAL_SHOTS_CAP
            )
            if DO_FINETUNE:
                nnd.finetune(p_local=P_LOCAL, p_trans=pT, shots=FT_SHOTS, epochs=FT_EPOCHS)
            ler_avg = unconditional_ler_nn_avg(nnd, sampler, SHOTS_EVAL, NUM_TRIALS)
            row = {"decoder":"NN","method":"unfiltered","p_trans":pT,"ler":ler_avg,"accept":1.0}
            df = upsert_row(df, row)
    save_cache(df, CSV_NN)
    dfp = df
    y_unf = [float(dfp[(dfp.decoder=="NN")&(dfp.method=="unfiltered")&(np.isclose(dfp.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    return y_unf


def compute_nn_curve(P_LOCAL = 0.01, TARGET_ACCEPT = 0.50, CSV_NN = None):
    df = load_cache(CSV_NN)
    y_swnn, y_dsnn = [], []
    for pT in P_TRANS_LIST:
        if not have_row(df,"NN",f"SW+{TARGET_ACCEPT}", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = SWFilterNNDecoder(
                circuit=circuit, model_path=NN_MODEL_PATH, d=D, num_layer=NUM_LAYER,
                batch_size=NN_BATCH_SIZE, device=NN_DEVICE, cal_shots_cap=NN_CAL_SHOTS_CAP
            )
            if DO_FINETUNE:
                nnd.finetune(p_local=P_LOCAL, p_trans=pT, shots=FT_SHOTS, epochs=FT_EPOCHS)

            synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
            tau = nnd.quantile_threshold(nnd.score_batch(synd_cal), TARGET_ACCEPT)

            acc_vals, ler_vals = [], []
            for _ in range(NUM_TRIALS):
                synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
                acc, condL = nnd.evaluate_batch(synd_eval, obs_eval, tau)
                acc_vals.append(acc); ler_vals.append(condL)
            row = {"decoder":"NN","method":f"SW+{TARGET_ACCEPT}","p_trans":pT,
                   "ler":float(np.mean(ler_vals)), "accept":float(np.mean(acc_vals))}
            df = upsert_row(df, row)

        if not have_row(df,"NN",f"DS+{TARGET_ACCEPT}", pT):
            circuit, dem, sampler = build_circuit_and_sampler(D, NUM_LAYER, P_LOCAL, pT)
            nnd = NNFilterDecoder(
                circuit=circuit, model_path=NN_MODEL_PATH, d=D, num_layer=NUM_LAYER,
                batch_size=NN_BATCH_SIZE, device=NN_DEVICE, cal_shots_cap=NN_CAL_SHOTS_CAP
            )
            if DO_FINETUNE:
                nnd.finetune(p_local=P_LOCAL, p_trans=pT, shots=FT_SHOTS, epochs=FT_EPOCHS)

            synd_cal, _ = sampler.sample(SHOTS_CAL, separate_observables=True)
            tau = nnd.quantile_threshold(nnd.score_batch(synd_cal), TARGET_ACCEPT)

            acc_vals, ler_vals = [], []
            for _ in range(NUM_TRIALS):
                synd_eval, obs_eval = sampler.sample(SHOTS_EVAL, separate_observables=True)
                acc, condL = nnd.evaluate_batch(synd_eval, obs_eval, tau)
                acc_vals.append(acc); ler_vals.append(condL)
            row = {"decoder":"NN","method":f"DS+{TARGET_ACCEPT}","p_trans":pT,
                   "ler":float(np.mean(ler_vals)), "accept":float(np.mean(acc_vals))}
            df = upsert_row(df, row)

    save_cache(df, CSV_NN)
    dfn = df
    y_swnn = [float(dfn[(dfn.decoder=="NN")&(dfn.method==f"SW+{TARGET_ACCEPT}")&(np.isclose(dfn.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    y_dsnn = [float(dfn[(dfn.decoder=="NN")&(dfn.method==f"DS+{TARGET_ACCEPT}")&(np.isclose(dfn.p_trans,pT))]["ler"]) for pT in P_TRANS_LIST]
    return y_swnn, y_dsnn

# ---------------------------
# Plotters
# ---------------------------
def plot_pm(x, y_unf, y_sw, y_zg, path):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y_unf, color="#8B0000", marker='o', linewidth=2.5, linestyle='--',
             label="Pure MWPM Decoder")
    plt.plot(x, y_sw, color="#0073E6", marker='s', linewidth=2.5,
             label=f"SW + MWPM")
    plt.plot(x, y_zg, color="#FF3333", marker='^', linewidth=2.5,
             label=f"DS + MWPM")
    
     # --- break-even line: an unencoded |0> read out in Z fails with probability 2 p_trans / 3 ---
    xmin, xmax = min(x), max(x)
    p_line = np.linspace(xmin, xmax, 100)
    plt.plot(p_line, 2 * p_line / 3, color = 'gray', linestyle = ':', linewidth=2, label="Break-even line")

    plt.ylim(top=0.49)
    # plt.yscale('log')
    plt.xticks(x)
    plt.xlabel("Transmission Error Rate", fontweight="bold")
    plt.ylabel("Logical Error Rate", fontweight="bold")
    # plt.title(fr"Decoder: PyMatching, $d=3$, $p_{{local}}={P_LOCAL}$")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(0, 0.6))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.03)
    plt.close()

def plot_nn(x, y_unf, y_swnn, y_nnfilter, path):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y_unf, color="#006400", marker='o', linewidth=2.5, linestyle='--',
             label="Pure NN Decoder")
    plt.plot(x, y_swnn, color="#e88800", marker='s', linewidth=2.5,
             label=f"SW + NN")
    plt.plot(x, y_nnfilter, color="#37A546", marker='^', linewidth=2.5,
             label=f"DS + NN")

     # --- break-even line: an unencoded |0> read out in Z fails with probability 2 p_trans / 3 ---
    xmin, xmax = min(x), max(x)
    p_line = np.linspace(xmin, xmax, 100)
    plt.plot(p_line, 2 * p_line / 3, color = 'gray', linestyle = ':', linewidth=2, label="Break-even line")

    # plt.yscale('log')
    plt.ylim(top=0.49)
    plt.xticks(x)
    plt.xlabel("Transmission Error Rate", fontweight="bold")
    plt.ylabel("Logical Error Rate", fontweight="bold")
    # plt.title(fr"Decoder: NN, $d=3$, $p_{{local}}={P_LOCAL}$")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.03)
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    np.random.seed(SEED)
    ensure_dirs()
    for p_local in P_LOCAL_LIST:
        print(f"=== P_LOCAL = {p_local} ===")
        CSV_PM = os.path.join(CACHEDIR, f"PM_d{D}_plocal{p_local}_pure.csv")
        CSV_NN = os.path.join(CACHEDIR, f"NN_d{D}_plocal{p_local}_pure.csv")
        y_pure_pm = compute_pure_pm(p_local, CSV_PM)
        y_pure_nn = compute_pure_nn(p_local, CSV_NN)
        print("Saved caches:")
        print(" ", CSV_PM)
        print(" ", CSV_NN)
        for target_accept in TARGET_ACCEPT_LIST:
            print(f"=== TARGET_ACCEPT = {target_accept} ===")
            CSV_PM = os.path.join(CACHEDIR, f"PM_d{D}_plocal{p_local}_accept{target_accept}.csv")
            CSV_NN = os.path.join(CACHEDIR, f"NN_d{D}_plocal{p_local}_accept{target_accept}.csv")
            # Compute/repair caches
            y_swpm, y_dspm = compute_pm_curve(p_local, target_accept, CSV_PM)
            y_swnn, y_dsnn = compute_nn_curve(p_local, target_accept, CSV_NN)

            # Plot
            f1 = os.path.join(OUTDIR, f"PM_plocal{p_local}_accept{target_accept}.pdf")
            plot_pm(P_TRANS_LIST, y_pure_pm, y_swpm, y_dspm, f1)

            f2 = os.path.join(OUTDIR, f"NN_plocal{p_local}_accept{target_accept}.pdf")
            plot_nn(P_TRANS_LIST, y_pure_nn, y_swnn, y_dsnn, f2)

            print("Saved figures:")
            print(" ", f1)
            print(" ", f2)
            print("Saved caches:")
            print(" ", CSV_PM)
            print(" ", CSV_NN)

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
    })
    main()
