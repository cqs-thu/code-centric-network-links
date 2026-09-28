#!/usr/bin/env python3
"""
Build the figure 11 CSVs from the raw chunk counts.

For each (filter, d, p_trans) the evaluation chunks 0..K-1 are pooled, where K is the stopping index of the
sequential rule (first K >= K_MIN chunks at which every acceptance target has >= 100 failures, or the 4e8-shot cap).
condL = failures / accepted for each target.  95% intervals are Clopper-Pearson.

Outputs (in out/ , or test_out/ in test mode):
  threshold_scan_{SyndromeWeight,ZCosetGap,NNFilter}_acc{10,20,40,80}.csv  original schema + raw counts
  fig11_all_points.csv                                                   all rows in one file
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import numpy as np
import pandas as pd

OUT = os.path.join(C.ROOT, "test_out" if C.TEST else "out")
os.makedirs(OUT, exist_ok=True)
allow_incomplete = "--allow-incomplete" in sys.argv

rows = []
missing = []
for f in C.FILTERS:
    for d in C.D_LIST:
        for p in C.P_TRANS:
            if not os.path.exists(C.cal_path(f, d, p)):
                missing.append((f, d, p, "no calibration"))
                continue
            cal = json.load(open(C.cal_path(f, d, p)))
            ch = C.load_chunks(f, d, p)
            kstop, kpre, _ = C.stop_index(ch)
            if kstop is None:
                missing.append((f, d, p, f"not finished (prefix {kpre} chunks)"))
                if not allow_incomplete or kpre == 0:
                    continue
                K, complete = kpre, False
            else:
                K, complete = kstop, True
            P = C.pooled(ch, K)
            n_overshoot = len([k for k in ch if k >= K])
            recount_ok = bool(ch[0]["extra"].get("verify", {}).get("recount_matches")) if 0 in ch else None
            fresh_ok = ch[0]["extra"].get("verify", {}).get("fresh_forward_matches") if 0 in ch else None
            zero_acc_chunks = sum(1 for k in range(K) if ch[k]["extra"]["zero_accept"])
            unf = ch[0]["extra"].get("unfiltered_failures") if 0 in ch else None
            for a in C.ACCS:
                ak = C.acc_key(a)
                n, m, k = P[ak]["shots"], P[ak]["accepted"], P[ak]["failures"]
                lo, hi = C.clopper_pearson(k, m)
                hit_cap = (K == C.K_CAP) and k < C.TARGET_FAIL
                rows.append(dict(
                    estimator=f, target_accept=a, p_local=C.P_LOCAL, d=d, p_trans=p,
                    condL=(k / m if m else float("nan")),
                    filter_label=C.LABEL[f], failures=k, accepted=m, shots_eval=n,
                    achieved_accept=m / n, ci95_lo=lo, ci95_hi=hi,
                    fewer_than_10_failures=k < 10, fewer_than_100_failures=k < C.TARGET_FAIL,
                    hit_cap=hit_cap, stop=("cap" if K == C.K_CAP else "rule") if complete else "INCOMPLETE",
                    n_chunks=K, chunk_shots=C.CHUNK, overshoot_chunks_not_used=n_overshoot,
                    tau=cal["taus"][ak], shots_cal=cal["shots_cal"], cal_accept=cal["cal_accept"][ak],
                    seed_cal=cal["seed"], seed_eval_first=C.seed(d, p, 1, 0), seed_eval_last=C.seed(d, p, 1, K - 1),
                    nn_batch=(C.NN_BATCH if f == "NNFilter" else ""),
                    chunk0_recount_matches=recount_ok, chunk0_fresh_forward_matches=fresh_ok,
                    chunks_with_zero_accept=zero_acc_chunks,
                    chunk0_unfiltered_failures=unf,
                ))

df = pd.DataFrame(rows)
for f in C.FILTERS:
    for a in C.ACCS:
        sub = df[(df.estimator == f) & np.isclose(df.target_accept, a)].sort_values(["d", "p_trans"])
        sub.to_csv(os.path.join(OUT, f"threshold_scan_{f}_acc{int(round(a * 100))}.csv"), index=False,
                   float_format="%.10g")
df.sort_values(["estimator", "target_accept", "d", "p_trans"]).to_csv(
    os.path.join(OUT, "fig11_all_points.csv"), index=False, float_format="%.10g")
print(f"wrote {len(df)} rows to {OUT}")
if missing:
    print(f"{len(missing)} (filter, d, p) not complete:")
    for m_ in missing:
        print("  ", m_)
