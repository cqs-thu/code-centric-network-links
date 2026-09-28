#!/usr/bin/env python3
"""
Worker of the finite-size scan (figure 11).

  worker.py --kind cpu --wid c1     SW+MWPM (SyndromeWeight) and DS+MWPM (ZCosetGap) points on one CPU core
  worker.py --kind gpu --wid g3     DS+NN (NNFilter) points on the GPU given by CUDA_VISIBLE_DEVICES (cuda:0)

Work items are claimed through files in results/claims (O_EXCL), results are written atomically:
  results/cal/<filter>_d<d>_p<p>.json (+ _scores.npz)       calibration: taus for the four targets
  results/eval/<filter>_d<d>_p<p>/cXXXX.json                 evaluation chunk k (320000 shots, seed fixed by k)
A restart loses at most the chunks that were running.
"""
import os, sys, json, time, argparse, math, traceback
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import numpy as np


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class Scheduler:
    def __init__(self, filters, wid):
        self.filters = filters
        self.wid = wid
        self.done = set()
        self.cache = {}          # (f,d,p) -> {k: record}

    def chunks(self, f, d, p):
        cd = C.chunk_dir(f, d, p)
        have = self.cache.setdefault((f, d, p), {})
        if os.path.isdir(cd):
            for fn in os.listdir(cd):
                if fn.startswith("c") and fn.endswith(".json"):
                    k = int(fn[1:5])
                    if k not in have:
                        try:
                            have[k] = json.load(open(os.path.join(cd, fn)))
                        except Exception:
                            pass
        return have

    def pick(self):
        pts = C.points(self.filters)
        # 1) calibrations
        for (f, d, p) in pts:
            if not os.path.exists(C.cal_path(f, d, p)):
                if not os.path.exists(C.claim_path(f, d, p)):
                    if C.try_claim(C.claim_path(f, d, p), C.host_info(self.wid)):
                        return ("cal", f, d, p, None)
        # 2) evaluation chunks
        claims = {}
        for fn in os.listdir(C.CLAIM_DIR):
            if fn.endswith(".claim") and "_c" in fn[-12:]:
                key, tail = fn[:-6].rsplit("_", 1)
                if tail.startswith("c") and tail[1:].isdigit():
                    claims.setdefault(key, set()).add(int(tail[1:]))
        all_done = True
        phase1, phase2 = [], []
        for (f, d, p) in pts:
            if (f, d, p) in self.done:
                continue
            all_done = False
            if not os.path.exists(C.cal_path(f, d, p)):
                continue
            ch = self.chunks(f, d, p)
            kstop, kpre, fails = C.stop_index(ch)
            if kstop is not None:
                self.done.add((f, d, p))
                continue
            claimed = claims.get(C.pkey(f, d, p), set())
            in_prog = len([k for k in claimed if k not in ch])
            extra_done = len([k for k in ch if k >= kpre])
            needed = max(1, C.K_MIN - kpre)
            if kpre > 0:
                for a in C.ACCS:
                    fa = fails[C.acc_key(a)]
                    if fa < C.TARGET_FAIL:
                        rate = fa / kpre
                        needed = max(needed, math.ceil((C.TARGET_FAIL - fa) / rate) if rate > 0 else C.K_CAP)
            else:
                needed = max(needed, C.K_MIN)
            needed = min(needed, C.K_CAP - kpre)
            if in_prog + extra_done >= needed:
                continue
            nxt = next((k for k in range(C.K_CAP) if k not in ch and k not in claimed), None)
            if nxt is None:
                continue
            (phase1 if kpre < C.K_MIN else phase2).append((f, d, p, nxt))
        if all_done:
            return "ALL_DONE"
        for (f, d, p, k) in phase1 + phase2:
            if C.try_claim(C.claim_path(f, d, p, k), C.host_info(self.wid)):
                return ("eval", f, d, p, k)
        return None


def run_cal(f, d, p, circ, est, memo, wid):
    t0 = time.time()
    smp = circ.compile_detector_sampler(seed=C.seed(d, p, 0))
    syn, _ = smp.sample(C.CAL_SHOTS, separate_observables=True)
    if memo is not None:
        memo.reset()
    scores = np.asarray(est.score_batch(syn))
    assert scores.shape == (C.CAL_SHOTS,), scores.shape
    taus, cal_acc = {}, {}
    for a in C.ACCS:
        tau = est.quantile_threshold(scores, a)
        taus[C.acc_key(a)] = float(tau)
        cal_acc[C.acc_key(a)] = float(np.mean(est.accept_mask(scores, tau)))
    flags = dict(n_nan=int(np.sum(np.isnan(scores))), n_inf=int(np.sum(np.isinf(scores))))
    if f == "ZCosetGap":
        flags["all_scores_zero"] = bool(np.all(scores == 0))      # estimator_z ValueError fallback
    np.savez_compressed(C.cal_scores_path(f, d, p), scores=scores)
    rec = dict(filter=f, label=C.LABEL[f], d=d, p_trans=p, p_local=C.P_LOCAL, num_layer=C.NUM_LAYER,
               seed=C.seed(d, p, 0), shots_cal=C.CAL_SHOTS, score_dtype=str(scores.dtype), taus=taus,
               cal_accept=cal_acc, flags=flags, nn_batch=(C.NN_BATCH if f == "NNFilter" else None),
               nn_forward_passes=(memo.n_forward if memo is not None else None),
               seconds=time.time() - t0, worker=C.host_info(wid))
    if memo is not None:
        memo.reset()
    C.atomic_write_json(C.cal_path(f, d, p), rec)
    log(f"cal {C.pkey(f, d, p)} taus={taus} cal_accept={cal_acc} {rec['seconds']:.1f}s")


def run_chunk(f, d, p, k, circ, est, memo, taus, wid):
    t0 = time.time()
    smp = circ.compile_detector_sampler(seed=C.seed(d, p, 1, k))
    syn, obs = smp.sample(C.CHUNK, separate_observables=True)
    assert obs.shape == (C.CHUNK, 1), obs.shape
    t1 = time.time()
    if memo is not None:
        memo.reset()
    res = {}
    for a in C.ACCS:
        ak = C.acc_key(a)
        ar, cl = est.evaluate_batch(syn, obs, taus[ak])
        m, fl = C.counts_from_eval(ar, cl, C.CHUNK)
        res[ak] = dict(acc_rate=float(ar), condL=float(cl), accepted=m, failures=fl)
    t2 = time.time()
    extra = dict(zero_accept=[ak for ak in res if res[ak]["accepted"] == 0])
    y_true = obs[:, 0].astype(bool)
    if memo is not None:
        pp = memo.p
        extra["nn_forward_passes"] = memo.n_forward
        extra["unfiltered_failures"] = int(np.sum((pp > 0.5) != y_true))
        extra["n_p_equal_1"] = int(np.sum(pp == 1.0))
    if k == 0:
        # verification chunk: explicit 1-D recount (and, for the NN, a fresh forward pass)
        ver = {}
        if memo is not None:
            pp = memo.p
            memo.enabled = False
            ar, cl = est.evaluate_batch(syn, obs, taus["0.10"])
            memo.enabled = True
            ver["fresh_forward_acc0.10"] = list(C.counts_from_eval(ar, cl, C.CHUNK))
            ver["fresh_forward_matches"] = ver["fresh_forward_acc0.10"] == [res["0.10"]["accepted"], res["0.10"]["failures"]]
            eps = 1e-9
            conf = np.abs(np.log((1.0 - np.clip(pp, eps, 1 - eps)) / np.clip(pp, eps, 1 - eps)))
            y_pred = pp > 0.5
            masks = {C.acc_key(a): conf >= taus[C.acc_key(a)] for a in C.ACCS}
        else:
            pred = est.matcher_std.decode_batch(syn)
            assert pred.shape == (C.CHUNK, 1), pred.shape
            y_pred = pred[:, 0].astype(bool)
            sc = np.asarray(est.score_batch(syn))
            if f == "ZCosetGap":
                ver["all_scores_zero"] = bool(np.all(sc == 0))
            masks = {C.acc_key(a): np.asarray(est.accept_mask(sc, taus[C.acc_key(a)])).reshape(-1) for a in C.ACCS}
            extra["unfiltered_failures"] = int(np.sum(y_pred != y_true))
        ok = True
        for ak, mk in masks.items():
            assert mk.shape == y_true.shape == y_pred.shape
            rc = [int(mk.sum()), int(np.sum(y_pred[mk] != y_true[mk]))]
            ver[f"recount_{ak}"] = rc
            ok &= rc == [res[ak]["accepted"], res[ak]["failures"]]
        ver["recount_matches"] = bool(ok)
        extra["verify"] = ver
        if not ok:
            log(f"RECOUNT MISMATCH {C.pkey(f, d, p)} {ver} {res}")
    if memo is not None:
        memo.reset()
    rec = dict(filter=f, d=d, p_trans=p, k=k, seed=C.seed(d, p, 1, k), shots=C.CHUNK, acc=res, extra=extra,
               sec_sample=t1 - t0, sec_eval=t2 - t1, sec_total=time.time() - t0, worker=C.host_info(wid))
    C.atomic_write_json(C.chunk_path(f, d, p, k), rec)
    fs = " ".join(f"{ak}:{res[ak]['failures']}/{res[ak]['accepted']}" for ak in res)
    log(f"chunk {C.pkey(f, d, p)} k={k} {fs} {rec['sec_total']:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--wid", required=True)
    a = ap.parse_args()
    C.ensure_dirs()
    if a.kind == "gpu":
        import torch
        torch.set_num_threads(1)
        filters, device = ["NNFilter"], "cuda:0"
        log(f"gpu worker {a.wid}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"{torch.cuda.get_device_name(0)} torch {torch.__version__} tf32(matmul)={torch.backends.cuda.matmul.allow_tf32} "
            f"tf32(cudnn)={torch.backends.cudnn.allow_tf32} cudnn.benchmark={torch.backends.cudnn.benchmark}")
    else:
        filters, device = ["SyndromeWeight", "ZCosetGap"], "cpu"
        log(f"cpu worker {a.wid}")
    sched = Scheduler(filters, a.wid)
    ests = {}
    nfail = 0
    while True:
        task = sched.pick()
        if task == "ALL_DONE":
            log("all points done; exiting")
            break
        if task is None:
            time.sleep(20)
            continue
        kind, f, d, p, k = task
        try:
            if (f, d, p) not in ests:
                circ, est = C.build(f, d, p, device)
                memo = C.MemoNN(est) if f == "NNFilter" else None
                ests[(f, d, p)] = (circ, est, memo)
            circ, est, memo = ests[(f, d, p)]
            if kind == "cal":
                run_cal(f, d, p, circ, est, memo, a.wid)
            else:
                taus = json.load(open(C.cal_path(f, d, p)))["taus"]
                run_chunk(f, d, p, k, circ, est, memo, taus, a.wid)
            nfail = 0
        except Exception:
            log(f"ERROR on {task}:\n{traceback.format_exc()}")
            try:
                os.remove(C.claim_path(f, d, p, k))
            except OSError:
                pass
            nfail += 1
            if nfail >= 5:
                log("5 consecutive errors; exiting")
                sys.exit(1)
            time.sleep(10)


if __name__ == "__main__":
    main()
