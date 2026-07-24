"""
SOTA synergy baselines for benchmarking ANTE.

All operate on *aligned* triples (a, b, y) of equal length (the caller handles
lag alignment: lagged -> a=A_{t-1}, b=B_{t-1}, y=Y_t; contemporaneous -> same t).

Baselines
---------
* interaction_information : classical co-information synergy
      II = I(A,B;Y) - I(A;Y) - I(B;Y)   (>0 synergy, <0 redundancy),
  with MI computed in the Gaussian (covariance) approximation.
* surd_synergy : the SURD decomposition (Martinez-Sanchez et al., Nat. Commun.
  2024; reference code cloned from GitHub) -> total synergistic info I_S, binned.
* oinfo : O-information (Rosas et al.) via the `hoi` package -> negative
  O-information indicates synergy-dominated triples.
* permutation_pvalue / peeking_permutation : batch significance for a synergy
  statistic and its (invalid) continuously-monitored counterpart, used to
  contrast with ANTE's anytime-valid control.
"""
import os, sys, subprocess
import numpy as np

# ------------------------------------------------------------------ Gaussian MI
def _gauss_mi(cov, ix, iy):
    """Gaussian mutual information between variable blocks ix, iy from covariance."""
    ixy = ix + iy
    def logdet(idx):
        sub = cov[np.ix_(idx, idx)]
        sign, ld = np.linalg.slogdet(sub)
        return ld
    return 0.5 * (logdet(ix) + logdet(iy) - logdet(ixy))


def interaction_information(a, b, y):
    """Co-information / interaction information (Gaussian). >0 => synergy."""
    X = np.vstack([np.asarray(a, float), np.asarray(b, float), np.asarray(y, float)])
    cov = np.cov(X) + 1e-9 * np.eye(3)
    I_AB_Y = _gauss_mi(cov, [0, 1], [2])
    I_A_Y = _gauss_mi(cov, [0], [2])
    I_B_Y = _gauss_mi(cov, [1], [2])
    return float(I_AB_Y - I_A_Y - I_B_Y)


# ------------------------------------------------------------------ kNN (KSG) MI
def ksg_mi(X, Y, k=5):
    """Kraskov-Stogbauer-Grassberger (estimator 1) mutual information I(X;Y), nats.

    X, Y are (n,) or (n,d) arrays. Nonparametric; captures arbitrary (incl.
    non-monotonic, e.g. product) dependence -- unlike Gaussian/copula MI.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    n = len(X)
    # small jitter to break ties / degeneracies
    rng = np.random.default_rng(0)
    X = X + 1e-10 * rng.standard_normal(X.shape)
    Y = Y + 1e-10 * rng.standard_normal(Y.shape)
    Z = np.hstack([X, Y])
    dz = cKDTree(Z).query(Z, k=k + 1, p=np.inf)[0][:, k]   # dist to k-th neighbour (max-norm)
    tx = cKDTree(X); ty = cKDTree(Y)
    nx = np.array([len(tx.query_ball_point(X[i], dz[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(Y[i], dz[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


# ------------------------------------------------------------------ SURD
_SURD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "SURD")

def _ensure_surd():
    utils = os.path.join(_SURD_DIR, "utils")
    if not os.path.isdir(utils):
        os.makedirs(os.path.dirname(_SURD_DIR), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Computational-Turbulence-Group/SURD.git",
                        _SURD_DIR], check=True, capture_output=True)
    # stub pymp (only the serial paths are used)
    stub = os.path.join(utils, "pymp.py")
    if not os.path.exists(stub):
        with open(stub, "w") as f:
            f.write("class _S:\n def array(self,shape,dtype=float):\n"
                    "  import numpy as np; return np.zeros(shape,dtype=dtype)\n"
                    "shared=_S()\n"
                    "class Parallel:\n def __init__(self,*a,**k):pass\n"
                    " def __enter__(self):return self\n def __exit__(self,*a):return False\n"
                    " def range(self,n):return range(n)\n"
                    " @property\n def thread_num(self):return 0\n")
    if utils not in sys.path:
        sys.path.insert(0, utils)


def surd_synergy(a, b, y, nbins=8):
    """Total synergistic information I_S from SURD (binned histogram)."""
    _ensure_surd()
    import surd as S
    data = np.vstack([np.asarray(y, float), np.asarray(a, float), np.asarray(b, float)])
    hist, _ = np.histogramdd(data.T, bins=(nbins, nbins, nbins))
    I_R, I_S, MI, leak = S.surd(hist)
    return float(sum(I_S.values()))


# ------------------------------------------------------------------ O-information
def oinfo(a, b, y):
    """O-information of the triple via `hoi` (negative => synergy-dominated).

    Returned as -O so that larger = more synergistic, matching the other scores.
    """
    try:
        from hoi.metrics import Oinfo
        X = np.vstack([np.asarray(a, float), np.asarray(b, float), np.asarray(y, float)]).T
        model = Oinfo(X)
        o = np.asarray(model.fit(minsize=3, maxsize=3))
        return float(-np.ravel(o)[0])
    except Exception as e:  # noqa
        return float("nan")


# ------------------------------------------------------------------ discrete PID / PEID
def _joint_pmf(*cols, states=2):
    """Empirical joint pmf over discrete columns (values in 0..states-1)."""
    cols = [np.asarray(c, int) for c in cols]
    shape = tuple(states for _ in cols)
    p = np.zeros(shape)
    for idx in zip(*cols):
        p[idx] += 1
    return p / p.sum()


def _mi_from_joint(pxy):
    px = pxy.sum(axis=1, keepdims=True); py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pxy / (px * py)
        return float(np.nansum(pxy * np.log2(np.where(pxy > 0, r, 1))))


def pid_wb_synergy(a, b, y, states=2):
    """Williams-Beer PID synergy (I_min redundancy), discrete. Syn = I(Y;A,B)
    - I(Y;A) - I(Y;B) + I_min.  (2 binary sources.)"""
    a = np.asarray(a, int); b = np.asarray(b, int); y = np.asarray(y, int)
    # joint source state s in 0..(states^2-1)
    s = a * states + b
    # build P(y, s) with s having states^2 values
    ns = states * states
    pys = np.zeros((states, ns))
    for yy, ss in zip(y, s):
        pys[yy, ss] += 1
    pys /= pys.sum()
    I_Y_AB = _mi_from_joint(pys)
    I_Y_A = _mi_from_joint(_joint_pmf(y, a, states=states))
    I_Y_B = _mi_from_joint(_joint_pmf(y, b, states=states))
    # I_min redundancy: sum_y p(y) min_i I_spec(y;A_i)
    py = np.bincount(y, minlength=states) / len(y)
    def spec_info(src):
        src = np.asarray(src, int)
        pjoint = _joint_pmf(y, src, states=states)  # (states_y, states_src)
        ps = pjoint.sum(axis=0, keepdims=True); pY = pjoint.sum(axis=1, keepdims=True)
        pY_s = pjoint / np.where(ps > 0, ps, 1)      # p(y|src)
        out = np.zeros(states)
        for yy in range(states):
            val = 0.0
            for ss in range(pjoint.shape[1]):
                if pjoint[yy, ss] > 0 and pY[yy, 0] > 0 and pY_s[yy, ss] > 0:
                    pa_y = pjoint[yy, ss] / pY[yy, 0]
                    val += pa_y * (np.log2(1/pY[yy, 0]) - np.log2(1/pY_s[yy, ss]))
            out[yy] = val
        return out
    iA = spec_info(a); iB = spec_info(b)
    Imin = float(np.sum(py * np.minimum(iA, iB)))
    return float(I_Y_AB - I_Y_A - I_Y_B + Imin)


def _EI_discrete(src, y, n_src_states):
    """Effective information EI(src->y) under max-entropy (uniform) intervention
    on the source, per PEID: EI = (1/M) sum_i sum_j p(j|i) log2( M p(j|i) / sum_k p(j|k) )."""
    src = np.asarray(src, int); y = np.asarray(y, int)
    ny = int(y.max()) + 1; M = n_src_states
    # conditional p(y=j | src=i)
    pj_i = np.zeros((M, ny))
    for ii, jj in zip(src, y):
        pj_i[ii, jj] += 1
    row = pj_i.sum(axis=1, keepdims=True)
    pj_i = np.divide(pj_i, np.where(row > 0, row, 1))
    colavg = pj_i.mean(axis=0, keepdims=True)          # (1/M) sum_k p(j|k)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pj_i / np.where(colavg > 0, colavg, 1)
        term = pj_i * np.log2(np.where(r > 0, r, 1))
    return float(term.sum() / M)


def peid_synergy(a, b, y, states=2):
    """PEID SynEID (Yang, Wang & Zhang 2026), discrete:
    Syn = EI({A,B}->Y) - EI(A->Y) - EI(B->Y)  under max-entropy source intervention."""
    a = np.asarray(a, int); b = np.asarray(b, int); y = np.asarray(y, int)
    s = a * states + b
    ei_joint = _EI_discrete(s, y, states * states)
    ei_a = _EI_discrete(a, y, states)
    ei_b = _EI_discrete(b, y, states)
    return float(ei_joint - ei_a - ei_b)


# ------------------------------------------------------------------ faithful continuous PEID
def _gauss_mi_blocks(U, V):
    """Affine-Gaussian mutual information I(U;V) from the joint sample covariance
    (this is PEID's affine transport-map MI: a Gaussian density on the features)."""
    U = np.atleast_2d(U); V = np.atleast_2d(V)
    if U.shape[0] < U.shape[1]: pass
    UV = np.hstack([U, V])
    du = U.shape[1]
    C = np.cov(UV, rowvar=False) + 1e-8 * np.eye(UV.shape[1])
    def ld(M): s, v = np.linalg.slogdet(M); return v
    return 0.5 * (ld(C[:du, :du]) + ld(C[du:, du:]) - ld(C))


def peid_synergy_continuous(a, b, y, n_mc=None, seed=0, intervene=True):
    """Faithful continuous PEID Syn^EID (Yang, Wang & Zhang 2026, Appendix F).

    Uses the paper's *affine-Gaussian transport map* MI on POLYNOMIAL-LIFTED
    features -- one source lifted to (x,x^2,x^3), the pair to (x,y,xy,x^2,y^2) --
    the cross term xy being what makes joint nonlinear (product) mechanisms
    detectable.  Synergy (Eq. 94):
        Syn = I(phi12(A,B);Y) - I(phi1(A);Y) - I(phi2(B);Y).
    With intervene=True the sources are first replaced by an independent
    max-entropy (uniform) sample pushed through a fitted polynomial mechanism
    (PEID's do(X~U) intervention that zeroes source-side redundancy).
    """
    a = np.asarray(a, float); b = np.asarray(b, float); y = np.asarray(y, float)
    n = len(y); n_mc = n_mc or n; rng = np.random.default_rng(seed)
    if intervene:
        # fit polynomial mechanism E[Y|A,B] and resample under uniform sources
        Phi = np.column_stack([np.ones(n), a, b, a*b, a*a, b*b])
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        resid = y - Phi @ w
        Ap = rng.uniform(a.min(), a.max(), n_mc); Bp = rng.uniform(b.min(), b.max(), n_mc)
        Yp = (np.column_stack([np.ones(n_mc), Ap, Bp, Ap*Bp, Ap*Ap, Bp*Bp]) @ w
              + rng.choice(resid, size=n_mc, replace=True))
        a, b, y = Ap, Bp, Yp
    phi1 = np.column_stack([a, a*a, a*a*a])
    phi2 = np.column_stack([b, b*b, b*b*b])
    phi12 = np.column_stack([a, b, a*b, a*a, b*b])
    Y = y[:, None]
    return float(_gauss_mi_blocks(phi12, Y) - _gauss_mi_blocks(phi1, Y) - _gauss_mi_blocks(phi2, Y))


# ------------------------------------------------------------------ batch tests
def permutation_pvalue(a, b, y, stat=interaction_information, n_perm=200, seed=0):
    """Two-sided-ish permutation p-value for a synergy statistic (batch)."""
    rng = np.random.default_rng(seed)
    obs = stat(a, b, y)
    a = np.asarray(a); b = np.asarray(b); y = np.asarray(y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(y))
        null[k] = stat(a, b[perm], y)      # break the joint (A,B)->Y structure
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, float(p)
