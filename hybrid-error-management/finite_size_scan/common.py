"""
Shared configuration and helpers for the finite-size scan (figure 11 of the paper).

The circuit, the DEM and the three estimators are built with the code in the parent directory:
  threshold_scan.build_circuit_and_sampler(cfg)   -> State_Encoding(d, num_layer=1, p_local, p_trans), DEM
  threshold_scan.make_estimator(name, circ, dem, cfg, args)
and every filter decision / failure count comes from the estimator's own
score_batch / quantile_threshold / evaluate_batch.

Sampling differs from threshold_scan.run_once_for_estimator as follows:
  * calibration batch = 2**18 = 262144 shots (>= 2e5), NN calibration cap disabled (--nn-cal-shots-cap 0),
  * evaluation in chunks of 320000 shots (= 625 NN batches of 512) with seeded stim samplers,
    until every acceptance target has >= 100 failures (minimum 4 chunks = 1.28e6 shots, cap 1250 chunks = 4e8 shots),
  * the four acceptance targets of one (filter, d, p_trans) share the calibration batch and the evaluation chunks
    (evaluate_batch is called once per target on the same chunk); for the NN the forward pass of a chunk is
    computed once and re-used by the four evaluate_batch calls (MemoNN below),
  * the three filters use the same seeds, so they see the same calibration and evaluation shots.
"""
import os, sys, json, time, socket, types, errno
sys.dont_write_bytecode = True
HQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # hybrid-error-management/
if HQ not in sys.path:
    sys.path.insert(0, HQ)
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST = os.environ.get("FIG11_TEST", "") == "1"
RES = os.path.join(ROOT, "test_results" if TEST else "results")
CAL_DIR = os.path.join(RES, "cal")
EVAL_DIR = os.path.join(RES, "eval")
CLAIM_DIR = os.path.join(RES, "claims")
LOG_DIR = os.path.join(ROOT, "test_logs" if TEST else "logs")

FILTERS = ["SyndromeWeight", "ZCosetGap", "NNFilter"]          # names used by threshold_scan.py
LABEL = {"SyndromeWeight": "SW + MWPM", "ZCosetGap": "DS + MWPM", "NNFilter": "DS + NN"}
D_LIST = [5, 7, 9, 11]
P_TRANS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
ACCS = [0.10, 0.20, 0.40, 0.80]
P_LOCAL = 0.003
NUM_LAYER = 1
NN_BATCH = 512            # threshold_scan.py default (--nn-batch-size 512)
CAL_SHOTS = 262144        # 2**18 = 512 * 512 (>= 2e5; every NN batch is full)
CHUNK = 320000            # 625 * 512 (every NN batch is full)
K_MIN = 4                 # at least 1.28e6 evaluation shots per point
K_CAP = 1250              # 1250 * 320000 = 4e8 evaluation shots per point
TARGET_FAIL = 100
SEED_BASE = 20260925
if TEST:   # small sizes for the pipeline test only (multiples of 512 kept)
    D_LIST = [5, 11]
    P_TRANS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    CAL_SHOTS = 8192
    CHUNK = 10240
    K_MIN = 2
    K_CAP = 6
    TARGET_FAIL = 20


def seed(d, p, stream, k=0):
    """stim seed; stream 0 = calibration, 1 = evaluation chunk k. Same seeds for all three filters."""
    pidx = P_TRANS.index(p)
    return (((SEED_BASE * 100 + d) * 10 + pidx) * 10 + stream) * 10000 + k


def pkey(filt, d, p):
    return f"{filt}_d{d}_p{p:.2f}"


def acc_key(acc):
    return f"{acc:.2f}"


def cal_path(filt, d, p):
    return os.path.join(CAL_DIR, pkey(filt, d, p) + ".json")


def cal_scores_path(filt, d, p):
    return os.path.join(CAL_DIR, pkey(filt, d, p) + "_scores.npz")


def chunk_dir(filt, d, p):
    return os.path.join(EVAL_DIR, pkey(filt, d, p))


def chunk_path(filt, d, p, k):
    return os.path.join(chunk_dir(filt, d, p), f"c{k:04d}.json")


def claim_path(filt, d, p, k=None):
    tail = "cal" if k is None else f"c{k:04d}"
    return os.path.join(CLAIM_DIR, f"{pkey(filt, d, p)}_{tail}.claim")


def ensure_dirs():
    for dd in (CAL_DIR, EVAL_DIR, CLAIM_DIR, LOG_DIR):
        os.makedirs(dd, exist_ok=True)


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def try_claim(path, info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    return True


def make_args(device="cuda:0"):
    # same fields that threshold_scan.make_estimator reads from the CLI namespace
    return types.SimpleNamespace(nn_model_dir=os.path.join(HQ, "Model"), nn_batch_size=NN_BATCH,
                                 nn_device=device, nn_cal_shots_cap=0)   # 0 -> None -> no cap


def build(filt, d, p, device="cuda:0"):
    import threshold_scan as H
    cfg = H.ExperimentConfig(d=d, p_local=P_LOCAL, p_trans=p, shots_cal=CAL_SHOTS, shots_eval=CHUNK,
                             num_layer=NUM_LAYER)
    circ, dem, _unseeded_sampler = H.build_circuit_and_sampler(cfg)
    est = H.make_estimator(filt, circ, dem, cfg, make_args(device))
    return circ, est


def counts_from_eval(acc_rate, condL, n):
    """Exact integer counts from evaluate_batch's (acceptance, conditional LER) on n shots."""
    n_acc = int(round(acc_rate * n))
    assert abs(acc_rate * n - n_acc) < 1e-6, (acc_rate, n)
    if n_acc == 0:
        assert condL == 1.0, condL          # evaluate_batch returns (0.0, 1.0) when nothing is accepted
        return 0, 0
    n_fail = int(round(condL * n_acc))
    assert abs(condL * n_acc - n_fail) < 1e-6, (condL, n_acc)
    return n_acc, n_fail


class MemoNN:
    """Caches the event tensor and the NN soft output of the current chunk on the estimator instance,
    so that the four evaluate_batch calls (one per acceptance target) on the same chunk run the
    network once.  evaluate_batch itself runs unmodified."""

    def __init__(self, est):
        self.est = est
        self.orig_fill = est._fill_event_tensor
        self.orig_pred = est._predict_soft
        self.enabled = True
        self.reset()
        est._fill_event_tensor = self.fill
        est._predict_soft = self.pred

    def reset(self):
        self.key = None
        self.E = None
        self.p = None
        self.n_forward = 0

    def fill(self, syndromes):
        if not self.enabled:
            return self.orig_fill(syndromes)
        if self.key is syndromes and self.E is not None:
            return self.E
        self.key, self.E, self.p = syndromes, self.orig_fill(syndromes), None
        return self.E

    def pred(self, E):
        if not self.enabled:
            return self.orig_pred(E)
        if E is self.E and self.p is not None:
            return self.p
        p = self.orig_pred(E)
        self.n_forward += 1
        if E is self.E:
            self.p = p
        return p


def load_chunks(filt, d, p):
    """dict k -> chunk record for all finished chunks."""
    out = {}
    cd = chunk_dir(filt, d, p)
    if not os.path.isdir(cd):
        return out
    for fn in os.listdir(cd):
        if fn.startswith("c") and fn.endswith(".json"):
            try:
                r = json.load(open(os.path.join(cd, fn)))
            except Exception:
                continue
            out[int(r["k"])] = r
    return out


def stop_index(chunks):
    """Sequential stopping rule on the contiguous prefix of finished chunks 0..K-1.
    Returns (K_stop or None, K_prefix, cumulative failures per acc at K_prefix).
    K_stop = first K >= K_MIN at which every acceptance target has >= TARGET_FAIL failures, or K_CAP."""
    fails = {acc_key(a): 0 for a in ACCS}
    K = 0
    while K in chunks and K < K_CAP:
        for a in ACCS:
            fails[acc_key(a)] += chunks[K]["acc"][acc_key(a)]["failures"]
        K += 1
        if K >= K_MIN and min(fails.values()) >= TARGET_FAIL:
            return K, K, fails
    if K >= K_CAP:
        return K_CAP, K, fails
    return None, K, fails


def pooled(chunks, K):
    """Pooled counts over chunks 0..K-1."""
    res = {}
    for a in ACCS:
        ak = acc_key(a)
        n = sum(chunks[k]["shots"] for k in range(K))
        m = sum(chunks[k]["acc"][ak]["accepted"] for k in range(K))
        f = sum(chunks[k]["acc"][ak]["failures"] for k in range(K))
        res[ak] = dict(shots=n, accepted=m, failures=f)
    return res


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def points(filters):
    """(filt, d, p) in scheduling order: cheap first (small d, large p_trans)."""
    out = []
    for d in D_LIST:
        for p in sorted(P_TRANS, reverse=True):
            for f in filters:
                out.append((f, d, p))
    return out


def host_info(wid):
    return dict(pid=os.getpid(), host=socket.gethostname(), wid=wid, time=time.strftime("%Y-%m-%d %H:%M:%S"))
