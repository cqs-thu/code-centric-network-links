#!/usr/bin/env python3
"""
Collect the benchmark output (results/*.jsonl) into CSV tables.  No value is edited: every number is
copied or computed arithmetically (mean, sample std, sums) from the recorded per-repetition timings/counts.

  runtime_results_regen.csv  one row per (tag, method, device, mode, d, p_trans): the plotted numbers
                             (mean and sample std over repetitions) + raw counts + measurement conditions
  runtime_reps_regen.csv     one row per repetition (batch mode: one 500-round batch; latency mode: one
                             block of 200 single rounds) with raw T_filter / T_decode / counts
  thresholds_regen.csv       thresholds and calibration acceptance per (method, d, p_trans, NN batch size)
  environment_regen.csv      software/hardware record of every benchmark process
"""
import os
import sys
import glob
import json

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def conditions(r, items):
    """Measurement conditions of the chosen section, or of the chosen units (unit-level validity)."""
    if r.get("validity") == "unit" or ("mon" not in r and "mon" in items[0]):
        mons = [x["mon"] for x in items]
        valid = all(x.get("ok", False) for x in items)
        n_extra = r.get("n_discarded", 0)
        cores = sorted({m["core"] for m in mons})
    else:
        mons = [r["mon"]]
        valid = bool(r.get("valid"))
        n_extra = r.get("n_attempts", 1) - 1
        cores = [r["mon"]["core"]]
    mhz = [m["mhz_avg"] for m in mons if m.get("mhz_avg") is not None]
    return dict(valid=valid, repeated_sections_or_units=n_extra, cores=";".join(map(str, cores)),
                sibling_busy_max=max(m.get("sibling_busy", np.nan) for m in mons),
                runqueue_wait_frac_max=max(m["wait_frac"] for m in mons),
                probe_ms_max=(max(max(m["probe_before_ms"], m["probe_after_ms"]) for m in mons)
                              if all("probe_before_ms" in m for m in mons) else None),
                clock_mhz_min=(min(mhz) if mhz else None), clock_mhz_mean=(float(np.mean(mhz)) if mhz else None))


def load():
    rows, reps, envs = [], [], []
    files = sorted(glob.glob(os.path.join(RES, "cpu_mwpm_*.jsonl")) + glob.glob(os.path.join(RES, "nn_gpu_*.jsonl"))
                   + glob.glob(os.path.join(RES, "nn_cpu1_*.jsonl")))
    for fn in files:
        for line in open(fn):
            r = json.loads(line)
            if r["kind"] == "env":
                envs.append(dict(file=os.path.basename(fn), **{k: (json.dumps(r.get(k)) if isinstance(r.get(k), (list, dict)) else r.get(k))
                                 for k in ["tag", "cpu_model", "gpu_name", "gpu_pci", "torch", "cuda", "cudnn",
                                           "torch_threads", "torch_interop_threads", "stim", "pymatching", "numpy",
                                           "python", "device", "cores", "freq_min_mhz", "probe_max_ms", "sib_max", "wait_max",
                                           "max_attempts", "warm_min_s", "env"]}))
                continue
            if r["kind"] not in ("batch", "latency"):
                continue
            mode = "batch500" if r["kind"] == "batch" else "batch1"
            items = r["reps"] if r["kind"] == "batch" else r["blocks"]
            t = np.array([x["t_round_us"] for x in items], float)
            f_us = np.array([x["T_filter_ns"] / x["n"] / 1e3 for x in items], float)
            d_us = np.array([x["T_decode_ns"] / max(1, x["n_acc"]) / 1e3 for x in items], float)
            row = dict(tag=r["tag"], method=r["method"], device=r["device"], mode=mode, d=r["d"],
                       p_trans=r["p_trans"], tau=r["tau"], rounds_per_rep=items[0]["n"], n_reps=len(items),
                       t_round_us_mean=float(t.mean()), t_round_us_std=float(t.std(ddof=1)),
                       t_round_us_median=float(np.median(t)), t_round_us_min=float(t.min()),
                       t_round_us_max=float(t.max()),
                       filter_us_per_round_mean=float(f_us.mean()),
                       decode_us_per_accepted_mean=float(d_us.mean()),
                       total_rounds=r["total_rounds"], total_accepted=r["total_accepted"],
                       total_failures=r["total_failures"],
                       accept_frac=r["total_accepted"] / r["total_rounds"],
                       median_end_to_end_accepted_us=r.get("median_end_to_end_accepted_us"),
                       p99_end_to_end_accepted_us=r.get("p99_end_to_end_accepted_us"),
                       **conditions(r, items),
                       loadavg_1min=(r.get("snap_before") or {}).get("loadavg", [None])[0],
                       time=(r.get("snap_before") or {}).get("t"),
                       single_pass_check_maxabsdiff=(r.get("check") or {}).get("single_pass_vs_score_batch_maxabsdiff"),
                       forward_calls_per_batch=str((r.get("check") or {}).get("forward_calls_per_batch", "")),
                       forward_calls_per_round=str((r.get("check") or {}).get("forward_calls_per_round", "")))
            rows.append(row)
            for i, x in enumerate(items):
                m = x.get("mon") or {}
                reps.append(dict(tag=r["tag"], method=r["method"], device=r["device"], mode=mode, d=r["d"],
                                 p_trans=r["p_trans"], rep=i, T_filter_ns=x["T_filter_ns"],
                                 T_decode_ns=x["T_decode_ns"], rounds=x["n"], accepted=x["n_acc"],
                                 failures=x["fails"], t_round_us=x["t_round_us"],
                                 unit_core=m.get("core"), unit_sibling_busy=m.get("sibling_busy"),
                                 unit_clock_mhz=m.get("mhz_avg")))
    return pd.DataFrame(rows), pd.DataFrame(reps), pd.DataFrame(envs)


def thresholds():
    rows = []
    for fn in sorted(glob.glob(os.path.join(RES, "cpu_mwpm_*.jsonl"))):
        for line in open(fn):
            r = json.loads(line)
            if r["kind"] == "calib":
                rows.append(dict(tag=r["tag"], method=r["method"], d=r["d"], p_trans=r["p_trans"],
                                 nn_inference_batch="", tau=r["tau"], n_cal_scored=r["n_cal"],
                                 cal_accepted=r["cal_accepted"], cal_accept_frac=r["cal_accept_frac"],
                                 cal_failures_in_sample=r["cal_failures_in_sample"],
                                 cal_unfiltered_failures=r["cal_unfiltered_failures"]))
    fn = os.path.join(RES, "nn_tau.jsonl")
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line)
            if r.get("kind") != "nn_tau":
                continue
            rows.append(dict(tag="nn_tau", method="DS + NN", d=r["d"], p_trans=r["p_trans"], nn_inference_batch=512,
                             tau=r["tau_b512"], n_cal_scored=r["n_scored_b512"],
                             cal_accepted=int(round(r["cal_accept_frac_b512"] * r["n_scored_b512"])),
                             cal_accept_frac=r["cal_accept_frac_b512"], cal_failures_in_sample=None,
                             cal_unfiltered_failures=None))
            rows.append(dict(tag="nn_tau", method="DS + NN", d=r["d"], p_trans=r["p_trans"], nn_inference_batch=1,
                             tau=r["tau_b1"], n_cal_scored=r["n_cal_b1"], cal_accepted=r["cal_accepted_b1"],
                             cal_accept_frac=r["cal_accept_frac_b1"],
                             cal_failures_in_sample=r["cal_failures_in_sample_b1"],
                             cal_unfiltered_failures=r["cal_unfiltered_failures_b1"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, reps, envs = load()
    key = ["tag", "method", "device", "mode", "d", "p_trans"]
    df = df.sort_values(key).reset_index(drop=True)
    reps = reps.sort_values(key + ["rep"]).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, "runtime_results_regen.csv"), index=False)
    reps.to_csv(os.path.join(HERE, "runtime_reps_regen.csv"), index=False)
    thresholds().to_csv(os.path.join(HERE, "thresholds_regen.csv"), index=False)
    envs.to_csv(os.path.join(HERE, "environment_regen.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["tag", "method", "device", "mode", "d", "p_trans", "n_reps", "t_round_us_mean", "t_round_us_std",
            "accept_frac", "total_failures", "valid", "repeated_sections_or_units", "sibling_busy_max",
            "clock_mhz_min"]
    print(df[cols].to_string(index=False))
