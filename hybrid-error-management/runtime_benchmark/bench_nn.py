#!/usr/bin/env python3
"""
DS+NN runtime with ONE forward pass per round (the soft output gives both the filter score |logit p|
and the decision p > 0.5), unmodified estimator_NN.NNFilterDecoder.

  --device cuda : NN on the GPU given by CUDA_VISIBLE_DEVICES (cuda:0); host side pinned to
                  one CPU core, torch.set_num_threads(1).  Every timed section/unit starts only when no
                  other compute process is on that GPU and is invalid if one appears during it.
  --device cpu  : NN on one CPU core, torch.set_num_threads(1), OMP/MKL threads = 1.

For every (d, p_trans) in --d (default all) x P_TRANS_GRID:
  est = NNFilterDecoder(circuit, Model/model_d_{d}.pt, d, num_layer=1, batch_size=512, device,
                        cal_shots_cap=30000)
  untimed checks : conf of the single-pass pipeline == est.score_batch on the same batch;
                   forward calls per batch (hook) = [500]; per single round = [1].
  batch mode     : warm-up (>= 0.5 s) on batches 0-4, then --reps timed batches of 500 rounds  (tau_b512)
  latency mode   : warm-up (>= 0.5 s) on rounds 0-49, then --lat-blocks blocks of 200 single rounds (tau_b1)
Core protocol (rt_common): wait for a physical core with both hardware threads idle, pin to one of them,
warm-up >= 0.5 s.  Validity: GPU batch mode -> per section (run_section); NN-on-CPU batches and all latency
blocks last >= 1 s -> per unit (run_units).  Criterion: other hardware thread busy <= 5%, run-queue wait
<= 2% of wall time, speed probe (pure-Python loop) before and after <= 3.9 ms; for cuda also: GPU used by
no other process.  The scaling_cur_freq clock reading is recorded but is not a criterion here: on one CPU
core the NN runs memory-bound AVX-512 kernels at a varying clock, and with the GPU the host core idles while
it waits for the GPU, so the averaged reading is low without any change in the measured time (checked on
the first GPU points: units rejected only by the clock reading took 8.3-9.4 ms, accepted units 8.5-9.4 ms).
Failing sections/units are repeated on a newly picked idle core (up to 20 attempts); all attempts are saved.
Thresholds come from results/nn_tau.jsonl (calibrate_nn.py).  Output: results/nn_<dev>_<tag>.jsonl
(restart-safe) and results/raw/lat_NN_<dev>_<tag>_d<d>_p<pT>.npz.
"""
import os
import sys
import time
import json
import argparse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import rt_common as C  # noqa: E402
from estimator_NN import NNFilterDecoder  # noqa: E402


def wait_gpu_idle(max_wait_s):
    t0 = time.time()
    while True:
        others = C.gpu_other_procs(os.getpid())
        if not others:
            return dict(waited_s=round(time.time() - t0, 1))
        if time.time() - t0 > max_wait_s:
            return dict(waited_s=round(time.time() - t0, 1), others=others, gave_up=True)
        print(f"[wait] GPU busy: {others}", flush=True)
        time.sleep(60)


def gpu_check():
    others = C.gpu_other_procs(os.getpid())
    return (len(others) == 0, dict(gpu_other_procs=others))


def strip_raw(units):
    return [{k: v for k, v in u.items() if k != "_raw"} for u in units]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--d", default=None, help="comma list of distances (default all)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lat-blocks", type=int, default=10)
    ap.add_argument("--max-wait-s", type=float, default=8 * 3600)
    ap.add_argument("--cores", required=True, help="candidate logical cpus (disjoint from other workers)")
    args = ap.parse_args()
    cores = [int(x) for x in args.cores.split(",")]
    assert args.reps + C.W_THR <= C.N_THR_BATCHES and args.lat_blocks <= C.N_LAT_BLOCKS_MAX
    dev = "cuda:0" if args.device == "cuda" else "cpu"
    devtag = "gpu" if args.device == "cuda" else "cpu1"
    on_gpu = args.device == "cuda"
    pre = (lambda: wait_gpu_idle(args.max_wait_s)) if on_gpu else None
    post = gpu_check if on_gpu else None

    def load_taus():
        return {(r["d"], r["p_trans"]): r for r in C.read_jsonl(os.path.join(C.RES, "nn_tau.jsonl"))
                if r.get("kind") == "nn_tau"}
    taus = load_taus()
    out = os.path.join(C.RES, f"nn_{devtag}_{args.tag}.jsonl")
    prev = C.read_jsonl(out)
    done = {(r["d"], r["p_trans"], r["kind"]) for r in prev if r["kind"] in ("batch", "latency")}

    info = C.env_info()
    info.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), device=dev)
    if on_gpu:
        info.update(gpu_name=torch.cuda.get_device_name(0), gpu_pci=C.set_gpu(),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cudnn_benchmark=torch.backends.cudnn.benchmark, gpu_state=C.gpu_state())
    C.append_jsonl(out, dict(kind="env", tag=args.tag, reps=args.reps, lat_blocks=args.lat_blocks,
                             B=C.B_THR, lat_block=C.N_LAT_BLOCK, lat_warm=C.N_LAT_WARM, idle_max=C.IDLE_MAX, sib_max=C.SIB_MAX,
                             wait_max=C.WAIT_MAX, max_attempts=C.MAX_ATTEMPTS, warm_min_s=C.WARM_MIN_S,
                             freq_min_mhz=None, probe_max_ms=C.PROBE_MAX_MS,
                             probe_n=C.PROBE_N,
                             cores=cores, **info))
    print(json.dumps(info), flush=True)

    d_list = [int(x) for x in args.d.split(",")] if args.d else C.D_LIST
    for d in d_list:
        for pT in C.P_TRANS_GRID:
            if (d, pT, "batch") in done and (d, pT, "latency") in done:
                print(f"[skip] d={d} pT={pT}", flush=True)
                continue
            while (d, pT) not in taus:          # thresholds are written by calibrate_nn.py
                print(f"[wait] no threshold yet for d={d} pT={pT}", flush=True)
                time.sleep(30)
                taus = load_taus()
            t_pt = time.time()
            tr = taus[(d, pT)]
            circuit, dem = C.build(d, pT)
            est = NNFilterDecoder(circuit=circuit, model_path=C.model_path(d), d=d, num_layer=C.NUM_LAYER,
                                  batch_size=C.NN_BATCH, device=dev, cal_shots_cap=C.NN_CAL_CAP)
            synd_thr, obs_thr = C.sample(circuit, d, pT, C.STREAM_THR, C.B_THR * C.N_THR_BATCHES)
            batches = [(synd_thr[k * C.B_THR:(k + 1) * C.B_THR], obs_thr[k * C.B_THR:(k + 1) * C.B_THR])
                       for k in range(C.N_THR_BATCHES)]
            synd_lat, obs_lat = C.sample(circuit, d, pT, C.STREAM_LAT, C.N_LAT_TOTAL)

            # ---- untimed checks: single pass == score_batch; one forward call per batch / round
            calls = []
            h = est.model.register_forward_hook(lambda mod, inp, outp: calls.append(int(inp[0].shape[0])))
            p0, conf0, m0 = C.nn_score_decide(est, batches[0][0], tr["tau_b512"])
            calls_batch = list(calls)
            calls.clear()
            ref = est.score_batch(batches[0][0])
            calls.clear()
            C.nn_score_decide(est, synd_lat[0:1], tr["tau_b1"])
            calls_round = list(calls)
            calls.clear()
            h.remove()
            check = dict(single_pass_vs_score_batch_maxabsdiff=float(np.max(np.abs(ref - conf0))),
                         decisions_equal=int(np.sum((ref >= tr["tau_b512"]) == m0)),
                         forward_calls_per_batch=calls_batch, forward_calls_per_round=calls_round)
            print(f"[check] d={d} pT={pT} {json.dumps(check)}", flush=True)

            def bunit(k):
                r = C.time_batch_nn(est, batches[k][0], batches[k][1], tr["tau_b512"])
                r["t_round_us"] = C.t_round_us(r["T_filter_ns"], r["T_decode_ns"], r["n"], r["n_acc"])
                r["batch_index"] = k
                return r

            def lunit(lo, hi):
                tf, td, acc, fail = C.time_rounds("nn", est, synd_lat[lo:hi], obs_lat[lo:hi], tr["tau_b1"])
                r = C.block_summary(tf, td, acc, fail)
                r["round_range"] = [lo, hi]
                r["_raw"] = (tf, td, acc, fail)
                return r

            # ---- batch mode (500 rounds per batch, tau_b512)
            if (d, pT, "batch") not in done:
                timed = [lambda k=k: bunit(k) for k in range(C.W_THR, C.W_THR + args.reps)]
                if on_gpu:
                    chosen, attempts = C.run_section(timed, [lambda k=k: bunit(k) for k in range(C.W_THR)],
                                                     cores, pre=pre, post_ok=post, freq_min=None)
                    reps = chosen["units"]
                    meta = dict(validity="section", valid=chosen["ok"], chosen_attempt=chosen["attempt"],
                                n_attempts=len(attempts), mon=chosen["mon"], core_pick=chosen["core_pick"],
                                n_warm_units=chosen["n_warm_units"], snap_before=chosen["snap"],
                                gpu_pre=chosen["pre"], gpu_post=chosen["post"],
                                attempts=[a for a in attempts if a is not chosen])
                else:
                    res = C.run_units(timed, [lambda: bunit(0), lambda: bunit(1)], [lambda: bunit(0)], cores,
                                      freq_min=None)
                    reps = res["units"]
                    meta = dict(validity="unit", valid=all(x["ok"] for x in reps),
                                n_discarded=len(res["discarded"]), discarded=res["discarded"],
                                core_picks=res["core_picks"], n_warm_units=res["n_warm_units"],
                                snap_before=res["snap"])
                v = [x["t_round_us"] for x in reps]
                mu, sd = C.mean_std(v)
                rec = dict(kind="batch", tag=args.tag, method="DS + NN", device=devtag, d=d, p_trans=pT,
                           tau=tr["tau_b512"], B=C.B_THR, n_reps=len(v), t_round_us_mean=mu, t_round_us_std=sd,
                           t_round_us_median=float(np.median(v)),
                           total_rounds=int(sum(x["n"] for x in reps)),
                           total_accepted=int(sum(x["n_acc"] for x in reps)),
                           total_failures=int(sum(x["fails"] for x in reps)),
                           reps=reps, check=check, gpu_state_after=(C.gpu_state() if on_gpu else None), **meta)
                C.append_jsonl(out, rec)
                print(f"[batch] DS + NN {devtag} d={d:2d} pT={pT:.2f} t_round={mu:10.3f} +- {sd:8.3f} us  "
                      f"acc={rec['total_accepted']}/{rec['total_rounds']} fails={rec['total_failures']} "
                      f"valid={meta['valid']}", flush=True)

            # ---- latency mode (batch size 1, tau_b1)
            if (d, pT, "latency") not in done:
                timed = [lambda b=b: lunit(C.N_LAT_WARM + b * C.N_LAT_BLOCK, C.N_LAT_WARM + (b + 1) * C.N_LAT_BLOCK)
                         for b in range(args.lat_blocks)]
                res = C.run_units(timed, [lambda: lunit(0, C.N_LAT_WARM)], [lambda: lunit(0, C.N_LAT_WARM)],
                                  cores, pre=pre, post_ok=post, freq_min=None)
                blocks = res["units"]
                v = [x["t_round_us"] for x in blocks]
                mu, sd = C.mean_std(v)
                tf = np.concatenate([x["_raw"][0] for x in blocks]); td = np.concatenate([x["_raw"][1] for x in blocks])
                acc = np.concatenate([x["_raw"][2] for x in blocks]); fail = np.concatenate([x["_raw"][3] for x in blocks])
                fn = os.path.join(C.RAW, f"lat_NN_{devtag}_{args.tag}_d{d}_p{pT:.2f}.npz")
                extra = {}
                for j, x in enumerate(res["discarded"]):
                    extra[f"discarded{j}_unit{x['unit']}_filter_ns"] = x["_raw"][0]
                np.savez_compressed(fn, filter_ns=tf, decode_ns=td, accepted=acc, failed=fail, **extra)
                rec = dict(kind="latency", tag=args.tag, method="DS + NN", device=devtag, d=d, p_trans=pT,
                           tau=tr["tau_b1"], B=1, rounds_per_block=C.N_LAT_BLOCK, n_reps=len(v),
                           t_round_us_mean=mu, t_round_us_std=sd, t_round_us_median=float(np.median(v)),
                           total_rounds=int(len(tf)), total_accepted=int(acc.sum()),
                           total_failures=int(fail[acc].sum()),
                           median_end_to_end_accepted_us=(float(np.median((tf + td)[acc]) / 1e3)
                                                          if acc.any() else None),
                           p99_end_to_end_accepted_us=(float(np.percentile((tf + td)[acc], 99) / 1e3)
                                                       if acc.any() else None),
                           raw_file=os.path.basename(fn), blocks=strip_raw(blocks), check=check,
                           validity="unit", valid=all(x["ok"] for x in blocks),
                           n_discarded=len(res["discarded"]), discarded=strip_raw(res["discarded"]),
                           core_picks=res["core_picks"], n_warm_units=res["n_warm_units"],
                           snap_before=res["snap"], gpu_pre=res["pre"],
                           gpu_state_after=(C.gpu_state() if on_gpu else None))
                C.append_jsonl(out, rec)
                print(f"[lat1 ] DS + NN {devtag} d={d:2d} pT={pT:.2f} t_round={mu:10.3f} +- {sd:8.3f} us  "
                      f"acc={rec['total_accepted']}/{rec['total_rounds']} fails={rec['total_failures']} "
                      f"valid={rec['valid']} discarded={rec['n_discarded']}", flush=True)
            del est
            if on_gpu:
                torch.cuda.empty_cache()
            print(f"[point] d={d} pT={pT} done in {time.time() - t_pt:.1f} s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
