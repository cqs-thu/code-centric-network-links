#!/usr/bin/env python3
"""Progress per (filter, d, p_trans): chunks done, stop state, failures per target."""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

filters = sys.argv[1].split(",") if len(sys.argv) > 1 else C.FILTERS
tot_done = tot = 0
for f in filters:
    print(f"== {f}")
    for d in C.D_LIST:
        for p in C.P_TRANS:
            tot += 1
            cal = os.path.exists(C.cal_path(f, d, p))
            ch = C.load_chunks(f, d, p)
            kstop, kpre, fails = C.stop_index(ch)
            st = "DONE" if kstop is not None else ("run " if cal else "cal?")
            tot_done += kstop is not None
            secs = sum(r["sec_total"] for r in ch.values())
            fs = " ".join(f"{fails[C.acc_key(a)]:>6d}" for a in C.ACCS)
            acc = ""
            if kpre:
                P = C.pooled(ch, kpre)
                acc = " ".join(f"{P[C.acc_key(a)]['failures']/max(1,P[C.acc_key(a)]['accepted']):.2e}" for a in C.ACCS)
            print(f"  d={d:2d} p={p:.2f} {st} chunks={len(ch):4d} prefix={kpre:4d} fails[10,20,40,80]={fs}  LER={acc}  cpu/gpu-s={secs:9.0f}")
print(f"points done: {tot_done}/{tot}")
