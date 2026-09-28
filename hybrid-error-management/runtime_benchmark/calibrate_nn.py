#!/usr/bin/env python3
"""
DS+NN thresholds at 20% target acceptance (computed on a GPU; the threshold values do not depend on
GPU load, so this may run while the GPU is shared).  Unmodified estimator_NN.NNFilterDecoder.

  tau_b512 : est = NNFilterDecoder(..., batch_size=512, cal_shots_cap=30000)
             tau = est.quantile_threshold(est.score_batch(synd_cal), 0.20), synd_cal = 50,000 shots
             -> exactly the hybrid_qecc_figs.py calibration (its defaults); used for the 500-round batches.
  tau_b1   : est1 = NNFilterDecoder(..., batch_size=1): each of N_CAL_B1 = 10,000 calibration shots is
             inferred on its own (batch size 1), tau1 = est1.quantile_threshold(scores, 0.20).
             Used for the latency mode.  (The BatchNorm layers use batch statistics in eval mode, so the
             network output at batch size 1 differs from the output inside a batch of 512.)
             The same single pass also gives p, so the in-sample failure counts at batch size 1 are recorded.

Writes one line per (d, p_trans) to results/nn_tau.jsonl (restart-safe).
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python3 calibrate_nn.py
"""
import os
import sys
import time
import json

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import rt_common as C

torch.set_num_threads(1)
from estimator_NN import NNFilterDecoder  # noqa: E402


def main():
    dev = "cuda:0"
    out = os.path.join(C.RES, "nn_tau.jsonl")
    done = {(r["d"], r["p_trans"]) for r in C.read_jsonl(out) if r.get("kind") == "nn_tau"}
    gpu = dict(name=torch.cuda.get_device_name(0), pci=C.set_gpu())
    print(json.dumps(gpu), flush=True)
    for d in sorted(C.D_LIST, reverse=True):     # d=11 first: its CPU worker is the slowest
        for pT in C.P_TRANS_GRID:
            if (d, pT) in done:
                print(f"[skip] d={d} pT={pT}", flush=True)
                continue
            circuit, dem = C.build(d, pT)
            t0 = time.time()
            # ---- batch-512 threshold (hybrid_qecc_figs.py calibration)
            est = NNFilterDecoder(circuit=circuit, model_path=C.model_path(d), d=d, num_layer=C.NUM_LAYER,
                                  batch_size=C.NN_BATCH, device=dev, cal_shots_cap=C.NN_CAL_CAP)
            synd_cal, obs_cal = C.sample(circuit, d, pT, C.STREAM_CAL, C.N_CAL)
            sc = est.score_batch(synd_cal)
            tau512 = est.quantile_threshold(sc, C.TARGET_ACC)
            acc512 = float(np.mean(sc >= tau512))
            t1 = time.time()
            # ---- batch-1 threshold
            est1 = NNFilterDecoder(circuit=circuit, model_path=C.model_path(d), d=d, num_layer=C.NUM_LAYER,
                                   batch_size=1, device=dev, cal_shots_cap=C.NN_CAL_CAP)
            synd1, obs1 = C.sample(circuit, d, pT, C.STREAM_CAL_B1, C.N_CAL_B1)
            p1, conf1, _ = C.nn_score_decide(est1, synd1, np.inf)       # batch_size=1: one shot per forward
            ref = est1.score_batch(synd1[:200])                          # literal score_batch, first 200 shots
            maxdiff = float(np.max(np.abs(ref - conf1[:200])))
            tau1 = est1.quantile_threshold(conf1, C.TARGET_ACC)
            m1 = conf1 >= tau1
            pred1 = (p1 > 0.5).astype(np.uint8)
            ob1 = np.asarray(obs1).reshape(-1).astype(np.uint8)
            t2 = time.time()
            rec = dict(kind="nn_tau", d=d, p_trans=pT, gpu=gpu,
                       tau_b512=float(tau512), n_cal_b512=int(C.N_CAL), n_scored_b512=int(len(sc)),
                       cal_accept_frac_b512=acc512, n_inf_conf_b512=int(np.sum(~np.isfinite(sc))),
                       tau_b1=float(tau1), n_cal_b1=int(C.N_CAL_B1), cal_accept_frac_b1=float(m1.mean()),
                       cal_accepted_b1=int(m1.sum()),
                       cal_failures_in_sample_b1=int(np.sum(pred1[m1] != ob1[m1])),
                       cal_unfiltered_failures_b1=int(np.sum(pred1 != ob1)),
                       n_inf_conf_b1=int(np.sum(~np.isfinite(conf1))),
                       check_b1_score_batch_maxabsdiff_first200=maxdiff,
                       time_b512_s=t1 - t0, time_b1_s=t2 - t1,
                       gpu_other_procs=C.gpu_other_procs(os.getpid()), t=time.strftime("%Y-%m-%dT%H:%M:%S"))
            C.append_jsonl(out, rec)
            print(json.dumps(rec), flush=True)
            del est, est1
            torch.cuda.empty_cache()
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
