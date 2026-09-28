#!/usr/bin/env python3
"""Remove stale claims (claims without a result whose worker process is gone). Run only while no worker runs."""
import os, sys, json, socket
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def alive(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline").read()
    except OSError:
        return False
    return "worker.py" in cmd

C.ensure_dirs()
removed = kept = 0
for fn in sorted(os.listdir(C.CLAIM_DIR)):
    if not fn.endswith(".claim"):
        continue
    key, tail = fn[:-6].rsplit("_", 1)
    filt, dd, pp = key.split("_")
    d, p = int(dd[1:]), float(pp[1:])
    res = C.cal_path(filt, d, p) if tail == "cal" else C.chunk_path(filt, d, p, int(tail[1:]))
    if os.path.exists(res):
        continue
    try:
        info = json.load(open(os.path.join(C.CLAIM_DIR, fn)))
    except Exception:
        info = {}
    if info.get("host") == socket.gethostname() and alive(info.get("pid", -1)):
        kept += 1
        continue
    os.remove(os.path.join(C.CLAIM_DIR, fn))
    removed += 1
    print("removed stale claim", fn, info)
print(f"janitor: removed {removed}, kept {kept} live claims")
