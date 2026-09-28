"""
Shared definitions for the regenerated Fig 12 runtime benchmark.

All filter/decoder code is the estimator code in the parent directory (hybrid-error-management/):
  estimator_weight.SyndromeWeightEstimator   (SW + MWPM)
  estimator_z.ZCosetGapEstimator             (DS + MWPM)
  estimator_NN.NNFilterDecoder               (DS + NN, Model/model_d_{d}.pt)
  surface.State_Encoding                     (circuit; num_layer=1, p_local=0.003)

What is timed (wall clock, time.perf_counter_ns, Python GC disabled inside a timed block):
  filter stage : score computation for every round of the batch + threshold comparison
                 SW : est.score_batch (syndrome weight / n_det) + est.accept_mask
                 DS : est.score_batch (two MWPM decodings on the logical-lifted DEM) + est.accept_mask
                 NN : est._fill_event_tensor + est._predict_soft (ONE forward pass, incl. host->device
                      and device->host copies) + |logit p| (same expression as est.score_batch)
                      + est.accept_mask
  decode stage : correction for the accepted rounds only
                 SW, DS : est.matcher_std.decode_batch(synd[accepted])  (PyMatching, original DEM)
                 NN     : est._predict_hard_from_soft(p[accepted])       (reuses the soft output above)
  Not timed    : circuit/DEM construction, matcher construction, model loading, threshold calibration,
                 syndrome sampling, comparison of the correction with the true observable.

Per-round processing time of a batch (same definition as plot_running_time.py):
      t_round = T_filter / n_rounds + T_decode / n_accepted
  i.e. the time to filter and decode one accepted round.  n_rounds = 500 in the batch mode and
  n_rounds = N_LAT_BLOCK single rounds (each processed as a batch of size 1) in the latency mode.
"""
import os
import sys
import json
import time
import gc
import socket
import platform
import subprocess

sys.dont_write_bytecode = True
HQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # hybrid-error-management/
if HQ not in sys.path:
    sys.path.insert(0, HQ)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RAW = os.path.join(RES, "raw")
os.makedirs(RAW, exist_ok=True)

import numpy as np
import stim
import pymatching

from surface import State_Encoding  # noqa: E402

# ---------------------------------------------------------------- grid (as plot_running_time.py)
D_LIST = [5, 7, 9, 11]
P_LOCAL = 0.003
P_TRANS_GRID = [0.08, 0.12, 0.16, 0.20]
NUM_LAYER = 1
TARGET_ACC = 0.20

# ---------------------------------------------------------------- calibration (as hybrid_qecc_figs.py defaults)
N_CAL = 50_000        # --shots-cal default
NN_CAL_CAP = 30_000   # --nn-cal-shots-cap default (score_batch subsamples 30k of the 50k shots)
NN_BATCH = 512        # --nn-batch-size default
N_CAL_B1 = 10_000     # calibration shots for the threshold used with batch-size-1 NN inference

# ---------------------------------------------------------------- timing layout
B_THR = 500                 # rounds per batch (plot_running_time.py N_SYND)
N_THR_BATCHES = 25          # batches drawn per point (identical in every process)
W_THR = 5                   # warm-up batches (indices 0..4); timed batch r uses index W_THR + r
N_LAT_WARM = 50             # warm-up single rounds (indices 0..49)
N_LAT_BLOCK = 200           # single rounds per latency block
N_LAT_BLOCKS_MAX = 10
N_LAT_TOTAL = N_LAT_WARM + N_LAT_BLOCK * N_LAT_BLOCKS_MAX

STREAM_CAL, STREAM_THR, STREAM_LAT, STREAM_CAL_B1 = 1, 2, 3, 4

GPU_BUS_ID = None   # PCI bus id of cuda:0 (the GPU selected by CUDA_VISIBLE_DEVICES); set by set_gpu()


def set_gpu():
    """Record and return the PCI bus id of cuda:0."""
    global GPU_BUS_ID
    import torch
    props = torch.cuda.get_device_properties(0)
    GPU_BUS_ID = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
    return GPU_BUS_ID

METHOD_KEYS = {"SW + MWPM": "SW", "DS + MWPM": "DS", "DS + NN": "NN"}


def seed_for(d, pT, stream):
    return 10_000_000 * stream + 10_000 * d + int(round(pT * 1000))


def build(d, pT):
    circuit = State_Encoding(d=d, num_layer=NUM_LAYER, p_local=P_LOCAL, p_trans=pT)
    dem = circuit.detector_error_model(decompose_errors=True)
    return circuit, dem


def sample(circuit, d, pT, stream, shots):
    s = circuit.compile_detector_sampler(seed=seed_for(d, pT, stream))
    synd, obs = s.sample(shots, separate_observables=True)
    return synd, obs


def model_path(d):
    return os.path.join(HQ, "Model", f"model_d_{d}.pt")


# ---------------------------------------------------------------- environment records
def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def core_mhz(cpu):
    try:
        return int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read()) / 1000.0
    except Exception:
        return None


def sibling_list(cpu):
    try:
        return open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read().strip()
    except Exception:
        return None


def cpu_util_snapshot(cpus, dt=0.2):
    """Utilization of the given logical cpus over dt seconds (from /proc/stat)."""
    def read():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                c = int(p[0][3:])
                v = list(map(int, p[1:]))
                out[c] = (v[3] + v[4], sum(v[:8]))
        return out
    a = read()
    time.sleep(dt)
    b = read()
    res = {}
    for c in cpus:
        di = b[c][0] - a[c][0]
        dtot = max(1, b[c][1] - a[c][1])
        res[c] = round(1.0 - di / dtot, 3)
    return res


def sib_of(cpu):
    """The other hardware thread of the physical core that holds logical cpu `cpu`."""
    s = [int(x) for x in sibling_list(cpu).replace("-", ",").split(",")]
    o = [x for x in s if x != cpu]
    return o[0] if o else cpu


def _stat(cpus):
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split()
            c = int(p[0][3:])
            if c in cpus:
                v = list(map(int, p[1:]))
                out[c] = (v[3] + v[4], sum(v[:8]))
    return out


# ---------------------------------------------------------------- core choice, monitoring, validity
IDLE_MAX = 0.05         # a physical core counts as idle if both hardware threads were busy <= 5% over the poll
SIB_MAX = 0.05          # max busy fraction of the sibling hardware thread during a timed section / unit
WAIT_MAX = 0.02         # max fraction of wall time the benchmark process waited on the run queue (preempted)
FREQ_MIN_MHZ = 3000     # min average core clock over a timed section (turbo active; base clock = 2600 MHz).
                        # scaling_cur_freq (intel_cpufreq) = APERF/MPERF average since the previous read,
                        # so a read at the start and one at the end of a section give the section average.
PROBE_N = 100_000       # speed probe: pure-Python loop of PROBE_N additions (L1-resident, clock-bound), min of 3
PROBE_MAX_MS = 3.9      # probe limit: 3.4-3.6 ms on a core at turbo clock with an idle sibling, ~4.25 ms
                        # expected at the 2.6 GHz base clock, 6-9 ms with a busy sibling.  The probe runs right
                        # before and right after each timed section/unit.  (The scaling_cur_freq average can read
                        # too low on this machine, so a section needs both a clock >= FREQ_MIN_MHZ, where used,
                        # and a fast probe.)
MAX_ATTEMPTS = 20       # attempts per section (or per unit) before the least-disturbed attempt is kept
WARM_MIN_S = 0.5        # warm-up runs at least this long (frequency governor, caches, CUDA kernels)


def pick_core(cands, window=0.3, max_wait_s=600):
    """Wait until a candidate physical core has both hardware threads idle (<= IDLE_MAX over `window` s),
    then pin this process to it (sched_setaffinity).  If none is found within max_wait_s, pin to the least
    busy candidate (idle_pair = False).  cands must not overlap between concurrent benchmark processes."""
    t0 = time.time()
    cpus = set(cands) | {sib_of(c) for c in cands}
    polls = 0
    while True:
        a = _stat(cpus)
        time.sleep(window)
        b = _stat(cpus)
        polls += 1
        u = {c: 1.0 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in cpus}
        best = min(cands, key=lambda c: (max(u[c], u[sib_of(c)]), u[c] + u[sib_of(c)]))
        idle = max(u[best], u[sib_of(best)]) <= IDLE_MAX
        if idle or time.time() - t0 > max_wait_s:
            break
    os.sched_setaffinity(0, {best})
    return dict(core=best, sibling=sib_of(best), util_core_before=round(u[best], 3),
                util_sibling_before=round(u[sib_of(best)], 3), idle_pair=bool(idle),
                waited_s=round(time.time() - t0, 2), polls=polls)


def proc_sched(pid):
    v = open(f"/proc/{pid}/schedstat").read().split()
    return int(v[0]), int(v[1])


def own_sched():
    return proc_sched("self")


def mon_begin():
    c = sorted(os.sched_getaffinity(0))[0]
    s = sib_of(c)
    mhz_before = core_mhz(c)          # also starts a new APERF/MPERF averaging interval
    return (c, s, _stat({c, s}), own_sched(), time.perf_counter_ns(), mhz_before)


def mon_end(st):
    c, s, a, (r0, w0), t0, mhz_before = st
    b = _stat({c, s})
    r1, w1 = own_sched()
    wall = time.perf_counter_ns() - t0
    mhz_avg = core_mhz(c)             # average clock (C0) since the read in mon_begin
    f = lambda x: round(1.0 - (b[x][0] - a[x][0]) / max(1, b[x][1] - a[x][1]), 4)
    return dict(core=c, sibling=s, core_busy=f(c), sibling_busy=f(s), sibling_ticks=int(b[s][1] - a[s][1]),
                wait_ns=int(w1 - w0), wait_frac=round((w1 - w0) / max(1, wall), 5), wall_s=round(wall / 1e9, 4),
                mhz_warmup=mhz_before, mhz_avg=mhz_avg)


def probe_ms():
    best = None
    for _ in range(3):
        t = time.perf_counter_ns()
        x = 0
        for i in range(PROBE_N):
            x += i
        dt = time.perf_counter_ns() - t
        best = dt if best is None else min(best, dt)
    return round(best / 1e6, 4)


def mon_ok(m, freq_min=None):
    ok = (m["sibling_busy"] <= SIB_MAX and m["wait_frac"] <= WAIT_MAX
          and m["probe_before_ms"] <= PROBE_MAX_MS and m["probe_after_ms"] <= PROBE_MAX_MS)
    if freq_min is not None:
        ok = ok and (m["mhz_avg"] is not None) and m["mhz_avg"] >= freq_min
    return bool(ok)


def mon_badness(m, freq_min=None):
    return (m["sibling_busy"] > SIB_MAX, m["wait_frac"] > WAIT_MAX,
            max(m["probe_before_ms"], m["probe_after_ms"]) > PROBE_MAX_MS,
            -(m["mhz_avg"] or 0) if freq_min is not None else 0, m["sibling_busy"], m["wait_frac"])


def warm(fns, min_s=WARM_MIN_S):
    """Run the warm-up units (cyclically) for at least min_s seconds; returns the number of units run.
    Garbage is collected once here, before the timed units of a section."""
    gc.collect()
    t0 = time.perf_counter()
    n = 0
    while True:
        for fn in fns:
            fn()
            n += 1
        if time.perf_counter() - t0 >= min_s:
            return n


def run_section(unit_fns, warm_fns, cands, pre=None, post_ok=None, freq_min=FREQ_MIN_MHZ):
    """Section-level validity.  Per attempt: wait for a physical core with both hardware threads idle and
    pin to it, snapshot, warm-up (>= 0.5 s), speed probe, all timed units back to back, speed probe.
    Valid if the sibling hardware thread was busy <= 5%, the run-queue wait <= 2%, both probes
    <= PROBE_MAX_MS, the average clock >= freq_min (if not None) and post_ok().  Repeated up to MAX_ATTEMPTS times; returns (chosen attempt, all attempts)."""
    attempts = []
    for a in range(MAX_ATTEMPTS):
        pre_info = pre() if pre else None
        pick = pick_core(cands)
        snap = snapshot()
        n_warm = warm(warm_fns)
        p0 = probe_ms()
        m0 = mon_begin()
        units = [fn() for fn in unit_fns]
        mon = mon_end(m0)
        mon.update(probe_before_ms=p0, probe_after_ms=probe_ms())
        extra = post_ok() if post_ok else (True, None)
        ok = mon_ok(mon, freq_min) and extra[0]
        attempts.append(dict(attempt=a, ok=bool(ok), core_pick=pick, snap=snap, n_warm_units=n_warm, mon=mon,
                             pre=pre_info, post=extra[1], freq_min=freq_min, units=units))
        if ok:
            break
    chosen = next((x for x in attempts if x["ok"]), None)
    if chosen is None:
        chosen = min(attempts, key=lambda x: mon_badness(x["mon"], freq_min))
    return chosen, attempts


def run_units(unit_fns, warm_fns, rewarm_fns, cands, pre=None, post_ok=None, freq_min=FREQ_MIN_MHZ):
    """Unit-level validity (units of >= 1 s).  Idle core + warm-up once; each timed unit is checked on its
    own; a failing unit is repeated after a new idle-core pick and a re-warm (up to MAX_ATTEMPTS times)."""
    pre_info = pre() if pre else None
    pick = pick_core(cands)
    picks = [pick]
    snap = snapshot()
    n_warm = warm(warm_fns)
    units, discarded = [], []
    for i, fn in enumerate(unit_fns):
        tries = []
        for a in range(MAX_ATTEMPTS):
            p0 = probe_ms()
            m0 = mon_begin()
            r = fn()
            mon = mon_end(m0)
            mon.update(probe_before_ms=p0, probe_after_ms=probe_ms())
            extra = post_ok() if post_ok else (True, None)
            r = dict(r)
            r.update(unit=i, attempt=a, mon=mon, post=extra[1], freq_min=freq_min,
                     ok=bool(mon_ok(mon, freq_min) and extra[0]))
            tries.append(r)
            if r["ok"]:
                break
            if pre:
                pre()
            pick = pick_core(cands)
            picks.append(pick)
            warm(rewarm_fns)
        good = next((x for x in tries if x["ok"]), None)
        if good is None:
            good = min(tries, key=lambda x: mon_badness(x["mon"], freq_min))
        units.append(good)
        discarded.extend([x for x in tries if x is not good])
    return dict(core_picks=picks, snap=snap, n_warm_units=n_warm, pre=pre_info, units=units,
                discarded=discarded)


def env_info():
    aff = sorted(os.sched_getaffinity(0))
    sibs = sorted({int(x) for c in aff for x in sibling_list(c).replace("-", ",").split(",")})
    return dict(host=socket.gethostname(), cpu_model=cpu_model(), n_logical_cpus=os.cpu_count(),
                affinity=aff, sibling_cpus=sibs,
                python=platform.python_version(), numpy=np.__version__, stim=stim.__version__,
                pymatching=pymatching.__version__, pid=os.getpid(),
                env={k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                    "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                                                    "CUDA_DEVICE_ORDER", "PYTHONDONTWRITEBYTECODE"]})


def snapshot():
    aff = sorted(os.sched_getaffinity(0))
    sibs = sorted({int(x) for c in aff for x in sibling_list(c).replace("-", ",").split(",")})
    return dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), loadavg=[round(x, 2) for x in os.getloadavg()],
                mhz={str(c): core_mhz(c) for c in aff}, util_self_and_siblings=cpu_util_snapshot(sibs))


def gpu_other_procs(own_pid=None):
    """Compute processes on the benchmark GPU other than own_pid (nvidia-smi, PCI bus id)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and GPU_BUS_ID is not None and parts[1].lower() == GPU_BUS_ID.lower():
            if own_pid is not None and parts[0] == str(own_pid):
                continue
            rows.append(line.strip())
    return rows


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(GPU_BUS_ID),
                              "--query-gpu=name,pci.bus_id,driver_version,utilization.gpu,memory.used,"
                              "clocks.sm,clocks.mem,temperature.gpu,pstate,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        return out.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


# ---------------------------------------------------------------- jsonl helpers
def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- timed pipelines
def nn_score_decide(est, synd, tau):
    """One NN forward pass per round.  conf is computed with exactly the expression used in
    NNFilterDecoder.score_batch (estimator_NN.py:66-71), so conf == est.score_batch(synd)."""
    E = est._fill_event_tensor(synd)
    p = est._predict_soft(E)
    eps = 1e-9
    conf = abs(np.log((1.0 - np.clip(p, eps, 1 - eps)) / np.clip(p, eps, 1 - eps)))
    mask = est.accept_mask(conf, tau)
    return p, conf, mask


def time_batch_mwpm(est, synd, obs, tau):
    gc.disable()
    t0 = time.perf_counter_ns()
    scores = est.score_batch(synd)
    mask = est.accept_mask(scores, tau)
    t1 = time.perf_counter_ns()
    pred = est.matcher_std.decode_batch(synd[mask])
    t2 = time.perf_counter_ns()
    gc.enable()
    pred = np.asarray(pred).reshape(-1)
    o = np.asarray(obs[mask]).reshape(-1)
    return dict(T_filter_ns=t1 - t0, T_decode_ns=t2 - t1, n=int(len(synd)), n_acc=int(mask.sum()),
                fails=int(np.sum(pred.astype(np.uint8) != o.astype(np.uint8))),
                n_zero_score=int(np.sum(np.asarray(scores) == 0)))


def time_batch_nn(est, synd, obs, tau):
    gc.disable()
    t0 = time.perf_counter_ns()
    p, conf, mask = nn_score_decide(est, synd, tau)
    t1 = time.perf_counter_ns()
    pred = est._predict_hard_from_soft(p[mask])
    t2 = time.perf_counter_ns()
    gc.enable()
    o = np.asarray(obs).reshape(-1)[mask]
    return dict(T_filter_ns=t1 - t0, T_decode_ns=t2 - t1, n=int(len(synd)), n_acc=int(mask.sum()),
                fails=int(np.sum(pred.astype(np.uint8) != o.astype(np.uint8))),
                n_inf_conf=int(np.sum(~np.isfinite(conf))))


def time_rounds(kind, est, synd, obs, tau):
    """Process len(synd) rounds one at a time (batch size 1).  Returns per-round arrays."""
    n = len(synd)
    tf = np.zeros(n, dtype=np.int64)
    td = np.zeros(n, dtype=np.int64)
    acc = np.zeros(n, dtype=bool)
    fail = np.zeros(n, dtype=bool)
    ob = np.asarray(obs).reshape(-1).astype(np.uint8)
    gc.collect()
    gc.disable()
    if kind == "mwpm":
        for i in range(n):
            s = synd[i:i + 1]
            t0 = time.perf_counter_ns()
            sc = est.score_batch(s)
            m = est.accept_mask(sc, tau)
            t1 = time.perf_counter_ns()
            if m[0]:
                pr = est.matcher_std.decode_batch(s)
                t2 = time.perf_counter_ns()
                td[i] = t2 - t1
                acc[i] = True
                fail[i] = int(np.asarray(pr).reshape(-1)[0]) != ob[i]
            tf[i] = t1 - t0
    else:
        for i in range(n):
            s = synd[i:i + 1]
            t0 = time.perf_counter_ns()
            p, conf, m = nn_score_decide(est, s, tau)
            t1 = time.perf_counter_ns()
            if m[0]:
                pr = est._predict_hard_from_soft(p[m])
                t2 = time.perf_counter_ns()
                td[i] = t2 - t1
                acc[i] = True
                fail[i] = int(pr[0]) != ob[i]
            tf[i] = t1 - t0
    gc.enable()
    return tf, td, acc, fail


def t_round_us(T_filter_ns, T_decode_ns, n, n_acc):
    """plot_running_time.py definition: filter time per round + decode time per accepted round."""
    return (T_filter_ns / n + T_decode_ns / max(1, n_acc)) / 1e3


def block_summary(tf, td, acc, fail):
    n = len(tf)
    n_acc = int(acc.sum())
    T_f = int(tf.sum())
    T_d = int(td[acc].sum())
    return dict(T_filter_ns=T_f, T_decode_ns=T_d, n=n, n_acc=n_acc, fails=int(fail[acc].sum()),
                t_round_us=t_round_us(T_f, T_d, n, n_acc),
                filter_us_per_round=T_f / n / 1e3,
                decode_us_per_accepted=(T_d / n_acc / 1e3) if n_acc else None,
                median_filter_us=float(np.median(tf) / 1e3),
                median_end_to_end_accepted_us=(float(np.median((tf + td)[acc]) / 1e3) if n_acc else None),
                p99_end_to_end_accepted_us=(float(np.percentile((tf + td)[acc], 99) / 1e3) if n_acc else None))


def mean_std(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
