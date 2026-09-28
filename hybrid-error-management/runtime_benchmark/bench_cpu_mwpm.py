#!/usr/bin/env python3
"""
SW+MWPM and DS+MWPM runtime on ONE CPU core.

For every (d, p_trans) in D_LIST x P_TRANS_GRID (p_local = 0.003, num_layer = 1):
  1. circuit/DEM as in plot_running_time.py (State_Encoding, decompose_errors=True); estimators
     SyndromeWeightEstimator(circuit, dem) and ZCosetGapEstimator(circuit, dem), unmodified.
  2. threshold: tau = est.quantile_threshold(est.score_batch(synd_cal), 0.20) on N_CAL = 50,000
     calibration shots (the calibration rule of hybrid_qecc_figs.py), or taken from --taus.
  3. for each method separately (a node runs one method), two timed sections:
       batch mode   : warm-up on batches 0-4 (>= 0.5 s), then --reps timed batches of 500 rounds
                      (batch indices 5, 6, ...)
       latency mode : warm-up on rounds 0-49 (>= 0.5 s), then --lat-blocks blocks of 200 rounds, every
                      round processed on its own (batch size 1)
     Section protocol (rt_common.run_section): wait for a physical core whose two hardware threads are
     both idle and pin the process to one of them; warm-up >= 0.5 s; speed probe; timed units; speed
     probe.  Valid if the other hardware thread of the core was busy <= 5% of the section, this process
     waited on the run queue <= 2% of the wall time, both speed probes (pure-Python loop) took
     <= 3.9 ms (3.4-3.6 ms at turbo clock; slower at the 2.6 GHz base clock or with a busy sibling),
     and the core's average clock read from scaling_cur_freq over the section was >= 3.0 GHz.
     Otherwise the section is repeated on a newly picked core (up to 20 attempts, all saved).
Output: results/cpu_mwpm_<tag>.jsonl (restart-safe) and results/raw/lat_<M>_cpu1_<tag>_d<d>_p<pT>.npz.

Usage: taskset -c 12 python3 bench_cpu_mwpm.py --tag A --cores 4,5,6,12 [--reps 20] [--lat-blocks 10]
       [--taus FILE]
"""
import os
import sys
import time
import json
import argparse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rt_common as C
from estimator_weight import SyndromeWeightEstimator
from estimator_z import ZCosetGapEstimator


def strip_raw(units):
    return [{k: v for k, v in u.items() if k != "_raw"} for u in units]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lat-blocks", type=int, default=10)
    ap.add_argument("--taus", default=None, help="thresholds_regen.csv-like json {method|d|pT: tau}")
    ap.add_argument("--points", default=None, help="subset 'd:pT,d:pT'")
    ap.add_argument("--cores", required=True, help="candidate logical cpus (disjoint from other workers)")
    args = ap.parse_args()
    cores = [int(x) for x in args.cores.split(",")]
    assert args.reps + C.W_THR <= C.N_THR_BATCHES and args.lat_blocks <= C.N_LAT_BLOCKS_MAX

    out = os.path.join(C.RES, f"cpu_mwpm_{args.tag}.jsonl")
    prev = C.read_jsonl(out)
    done = {(r["method"], r["d"], r["p_trans"], r["kind"]) for r in prev if r["kind"] in ("batch", "latency")}
    prev_tau = {(r["method"], r["d"], r["p_trans"]): r["tau"] for r in prev if r["kind"] == "calib"}
    ext_tau = {}
    if args.taus:
        ext_tau = {(k.split("|")[0], int(k.split("|")[1]), float(k.split("|")[2])): v
                   for k, v in json.load(open(args.taus)).items()}

    info = C.env_info()
    C.append_jsonl(out, dict(kind="env", tag=args.tag, reps=args.reps, lat_blocks=args.lat_blocks,
                             B=C.B_THR, W=C.W_THR, lat_block=C.N_LAT_BLOCK, lat_warm=C.N_LAT_WARM,
                             n_cal=C.N_CAL, target_acc=C.TARGET_ACC, taus_from=args.taus,
                             idle_max=C.IDLE_MAX, sib_max=C.SIB_MAX, wait_max=C.WAIT_MAX, max_attempts=C.MAX_ATTEMPTS,
                             warm_min_s=C.WARM_MIN_S, freq_min_mhz=C.FREQ_MIN_MHZ, probe_max_ms=C.PROBE_MAX_MS,
                             probe_n=C.PROBE_N, cores=cores, **info))
    print(json.dumps(info), flush=True)

    points = [(d, pT) for d in C.D_LIST for pT in C.P_TRANS_GRID]
    if args.points:
        points = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in args.points.split(",")]
    names = ["SW + MWPM", "DS + MWPM"]

    for d, pT in points:
        if all((m, d, pT, k) in done for m in names for k in ("batch", "latency")):
            print(f"[skip] d={d} pT={pT}", flush=True)
            continue
        t_pt = time.time()
        circuit, dem = C.build(d, pT)
        ests = {"SW + MWPM": SyndromeWeightEstimator(circuit, dem),
                "DS + MWPM": ZCosetGapEstimator(circuit, dem)}

        # ---- thresholds (20% target acceptance, estimator calibration rule)
        taus = {}
        synd_cal = obs_cal = None
        for m in names:
            if (m, d, pT) in ext_tau:
                taus[m] = float(ext_tau[(m, d, pT)])
                continue
            if (m, d, pT) in prev_tau:
                taus[m] = float(prev_tau[(m, d, pT)])
                continue
            if synd_cal is None:
                synd_cal, obs_cal = C.sample(circuit, d, pT, C.STREAM_CAL, C.N_CAL)
            est = ests[m]
            t0 = time.time()
            sc = est.score_batch(synd_cal)
            tau = est.quantile_threshold(sc, C.TARGET_ACC)
            t_cal = time.time() - t0
            mask = est.accept_mask(sc, tau)
            pred = np.asarray(est.matcher_std.decode_batch(synd_cal)).reshape(-1)
            ob = np.asarray(obs_cal).reshape(-1)
            rec = dict(kind="calib", tag=args.tag, method=m, d=d, p_trans=pT, tau=float(tau),
                       n_cal=int(len(synd_cal)), cal_accepted=int(mask.sum()),
                       cal_accept_frac=float(mask.mean()),
                       cal_failures_in_sample=int(np.sum(pred[mask] != ob[mask])),
                       cal_unfiltered_failures=int(np.sum(pred != ob)),
                       n_zero_score=int(np.sum(np.asarray(sc) == 0)), cal_time_s=t_cal)
            C.append_jsonl(out, rec)
            print(json.dumps(rec), flush=True)
            taus[m] = float(tau)

        synd_thr, obs_thr = C.sample(circuit, d, pT, C.STREAM_THR, C.B_THR * C.N_THR_BATCHES)
        batches = [(synd_thr[k * C.B_THR:(k + 1) * C.B_THR], obs_thr[k * C.B_THR:(k + 1) * C.B_THR])
                   for k in range(C.N_THR_BATCHES)]
        synd_lat, obs_lat = C.sample(circuit, d, pT, C.STREAM_LAT, C.N_LAT_TOTAL)

        for m in names:
            est, tau = ests[m], taus[m]

            # ---- batch mode
            if (m, d, pT, "batch") not in done:
                def bunit(k, est=est, tau=tau):
                    r = C.time_batch_mwpm(est, batches[k][0], batches[k][1], tau)
                    r["t_round_us"] = C.t_round_us(r["T_filter_ns"], r["T_decode_ns"], r["n"], r["n_acc"])
                    r["batch_index"] = k
                    return r
                warm_fns = [lambda k=k: bunit(k) for k in range(C.W_THR)]
                unit_fns = [lambda k=k: bunit(k) for k in range(C.W_THR, C.W_THR + args.reps)]
                chosen, attempts = C.run_section(unit_fns, warm_fns, cores)
                reps = chosen["units"]
                v = [x["t_round_us"] for x in reps]
                mu, sd = C.mean_std(v)
                rec = dict(kind="batch", tag=args.tag, method=m, device="cpu1", d=d, p_trans=pT, tau=tau,
                           B=C.B_THR, n_reps=len(v), t_round_us_mean=mu, t_round_us_std=sd,
                           t_round_us_median=float(np.median(v)),
                           total_rounds=int(sum(x["n"] for x in reps)),
                           total_accepted=int(sum(x["n_acc"] for x in reps)),
                           total_failures=int(sum(x["fails"] for x in reps)),
                           reps=reps, valid=chosen["ok"], chosen_attempt=chosen["attempt"],
                           n_attempts=len(attempts), mon=chosen["mon"], core_pick=chosen["core_pick"],
                           n_warm_units=chosen["n_warm_units"], snap_before=chosen["snap"],
                           attempts=[{k: v for k, v in a.items()} for a in attempts if a is not chosen])
                C.append_jsonl(out, rec)
                print(f"[batch] {m:10s} d={d:2d} pT={pT:.2f} t_round={mu:9.3f} +- {sd:7.3f} us  "
                      f"acc={rec['total_accepted']}/{rec['total_rounds']} fails={rec['total_failures']} "
                      f"attempts={len(attempts)} valid={chosen['ok']} sib={chosen['mon']['sibling_busy']} "
                      f"wait={chosen['mon']['wait_frac']} mhz={chosen['mon']['mhz_avg']}", flush=True)

            # ---- latency mode (batch size 1)
            if (m, d, pT, "latency") not in done:
                def lunit(lo, hi, est=est, tau=tau):
                    tf, td, acc, fail = C.time_rounds("mwpm", est, synd_lat[lo:hi], obs_lat[lo:hi], tau)
                    r = C.block_summary(tf, td, acc, fail)
                    r["round_range"] = [lo, hi]
                    r["_raw"] = (tf, td, acc, fail)
                    return r
                warm_fns = [lambda: lunit(0, C.N_LAT_WARM)]
                unit_fns = [lambda b=b: lunit(C.N_LAT_WARM + b * C.N_LAT_BLOCK, C.N_LAT_WARM + (b + 1) * C.N_LAT_BLOCK)
                            for b in range(args.lat_blocks)]
                chosen, attempts = C.run_section(unit_fns, warm_fns, cores)
                blocks = chosen["units"]
                v = [x["t_round_us"] for x in blocks]
                mu, sd = C.mean_std(v)
                tf = np.concatenate([x["_raw"][0] for x in blocks]); td = np.concatenate([x["_raw"][1] for x in blocks])
                acc = np.concatenate([x["_raw"][2] for x in blocks]); fail = np.concatenate([x["_raw"][3] for x in blocks])
                fn = os.path.join(C.RAW, f"lat_{C.METHOD_KEYS[m]}_cpu1_{args.tag}_d{d}_p{pT:.2f}.npz")
                extra = {}
                for a in attempts:
                    if a is chosen:
                        continue
                    extra[f"attempt{a['attempt']}_filter_ns"] = np.concatenate([x["_raw"][0] for x in a["units"]])
                    extra[f"attempt{a['attempt']}_decode_ns"] = np.concatenate([x["_raw"][1] for x in a["units"]])
                    extra[f"attempt{a['attempt']}_accepted"] = np.concatenate([x["_raw"][2] for x in a["units"]])
                np.savez_compressed(fn, filter_ns=tf, decode_ns=td, accepted=acc, failed=fail, **extra)
                rec = dict(kind="latency", tag=args.tag, method=m, device="cpu1", d=d, p_trans=pT, tau=tau,
                           B=1, rounds_per_block=C.N_LAT_BLOCK, n_reps=len(v), t_round_us_mean=mu,
                           t_round_us_std=sd, t_round_us_median=float(np.median(v)),
                           total_rounds=int(len(tf)), total_accepted=int(acc.sum()),
                           total_failures=int(fail[acc].sum()),
                           median_end_to_end_accepted_us=float(np.median((tf + td)[acc]) / 1e3),
                           p99_end_to_end_accepted_us=float(np.percentile((tf + td)[acc], 99) / 1e3),
                           raw_file=os.path.basename(fn), blocks=strip_raw(blocks), valid=chosen["ok"],
                           chosen_attempt=chosen["attempt"], n_attempts=len(attempts), mon=chosen["mon"],
                           core_pick=chosen["core_pick"], n_warm_units=chosen["n_warm_units"],
                           snap_before=chosen["snap"],
                           attempts=[dict({k: v for k, v in a.items() if k != "units"},
                                          units=strip_raw(a["units"])) for a in attempts if a is not chosen])
                C.append_jsonl(out, rec)
                print(f"[lat1 ] {m:10s} d={d:2d} pT={pT:.2f} t_round={mu:9.3f} +- {sd:7.3f} us  "
                      f"acc={rec['total_accepted']}/{rec['total_rounds']} fails={rec['total_failures']} "
                      f"attempts={len(attempts)} valid={chosen['ok']} sib={chosen['mon']['sibling_busy']} "
                      f"wait={chosen['mon']['wait_frac']} mhz={chosen['mon']['mhz_avg']}", flush=True)
        print(f"[point] d={d} pT={pT} done in {time.time() - t_pt:.1f} s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
