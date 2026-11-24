"""Dataset access.

Real data come from the Gagolewski benchmark suite (clustering-data-v1, cloned into data/).
Label files `<name>.labelsN.gz` hold one reference partition each; label 0 = noise.
Synthetic generators reproduce the pilot scenarios (S1-S3) and add controllable variants.
"""
import gzip, os, glob
import numpy as np

SUITE = os.path.join(os.path.dirname(__file__), "..", "data", "clustering-data-v1")
BATTERIES = ["cdc", "fcps", "g2mg", "graves", "h2mg", "mnist", "other", "sipu", "uci", "wut"]


def _read(path):
    with gzip.open(path, "rt") as f:
        return np.loadtxt(f)


def load(battery, name, suite=SUITE):
    """Return X (n,d) and a list of label arrays (0 = noise) for every reference labeling."""
    base = os.path.join(suite, battery, name)
    X = _read(base + ".data.gz")
    X = X.reshape(len(X), -1)
    labs = []
    for p in sorted(glob.glob(base + ".labels*.gz")):
        labs.append(_read(p).astype(int))
    return X, labs


def catalog(max_n=1000, max_d=3, suite=SUITE, batteries=BATTERIES):
    """List all datasets with n<=max_n and d<=max_d. Returns list of dicts."""
    rows = []
    for b in batteries:
        for p in sorted(glob.glob(os.path.join(suite, b, "*.data.gz"))):
            name = os.path.basename(p)[:-len(".data.gz")]
            X, labs = load(b, name, suite)
            n, d = X.shape
            if n > max_n or d > max_d or not labs:
                continue
            for li, y in enumerate(labs):
                K = len(set(y[y > 0]))
                rows.append(dict(battery=b, name=name, labeling=li, n=n, d=d, K=K,
                                 noise=float(np.mean(y == 0)),
                                 sizes_ratio=float(np.bincount(y[y > 0])[1:].max() /
                                                   max(1, np.bincount(y[y > 0])[1:].min()))))
    return rows


def tier(n):
    return 1 if n <= 400 else (2 if n <= 1000 else 3)


def add_uniform_noise(X, y, frac, seed=0):
    """Inject uniform background noise (frac * n points) in the bounding box; new points get label 0."""
    rng = np.random.default_rng(seed)
    m = int(round(frac * len(X)))
    if m == 0:
        return X, y
    lo, hi = X.min(0), X.max(0)
    Z = rng.uniform(lo, hi, (m, X.shape[1]))
    return np.vstack([X, Z]), np.concatenate([y, np.zeros(m, int)])


# ---------------- synthetic generators (noise/tail = label 0) ----------------
def gen_bridge(seed, nA=35, nB=35):
    """S1: two blobs + 2-point noise bridge in the valley + sparse 6-point tail on A (RG1)."""
    rng = np.random.default_rng(seed)
    A = rng.normal([0, 0], 0.33, (nA, 2)); B = rng.normal([3.3, 0], 0.33, (nB, 2))
    bridge = np.array([[1.45, 0.12], [1.85, -0.10]])
    tail = np.array([[-1.55, .05], [-1.9, .12], [-2.25, 0], [-2.6, .1], [-2.95, -.05], [-3.3, .08]])
    tail = tail + rng.normal(0, 0.04, tail.shape)
    X = np.vstack([A, B, bridge, tail])
    y = np.array([1] * nA + [2] * nB + [0] * 8)
    return X, y


def gen_imb(seed, nA=70, nB=12):
    """S2: imbalanced truth (70 vs 12) + 2-point bridge (RG2)."""
    rng = np.random.default_rng(seed)
    A = rng.normal([0, 0], 0.45, (nA, 2)); B = rng.normal([3.6, 0], 0.18, (nB, 2))
    bridge = np.array([[2.05, .10], [2.5, -.08]])
    return np.vstack([A, B, bridge]), np.array([1] * nA + [2] * nB + [0] * 2)


def gen_comb(seed, nA=70, nB=12):
    """S3: S2 plus a sparse 6-point tail on the large cluster."""
    rng = np.random.default_rng(seed)
    A = rng.normal([0, 0], 0.45, (nA, 2)); B = rng.normal([3.6, 0], 0.18, (nB, 2))
    bridge = np.array([[2.05, .10], [2.5, -.08]])
    tail = np.array([[-2.3 - 0.38 * k, 0.05 * (-1) ** k] for k in range(6)]) + rng.normal(0, .04, (6, 2))
    return np.vstack([A, B, bridge, tail]), np.array([1] * nA + [2] * nB + [0] * 8)


def gen_bridge_k(seed, K=3, n_per=40, n_bridge=2, n_tail=0, sep=3.0, sd=0.35):
    """Controllable generator: K blobs on a line, `n_bridge` points in each valley, optional tail."""
    rng = np.random.default_rng(seed)
    pts, lab = [], []
    for k in range(K):
        pts.append(rng.normal([k * sep, 0], sd, (n_per, 2))); lab += [k + 1] * n_per
    for k in range(K - 1):
        xs = np.linspace(k * sep + sd * 3.2, (k + 1) * sep - sd * 3.2, n_bridge + 2)[1:-1]
        pts.append(np.c_[xs, rng.normal(0, 0.05, n_bridge)]); lab += [0] * n_bridge
    for i in range(n_tail):
        pts.append([[-1.2 * sd * 3 - 0.45 * i, rng.normal(0, .05)]]); lab += [0]
    return np.vstack(pts), np.array(lab)


GENERATORS = dict(S1=gen_bridge, S2=gen_imb, S3=gen_comb)
