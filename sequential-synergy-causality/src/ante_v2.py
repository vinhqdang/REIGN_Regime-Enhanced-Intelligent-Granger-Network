"""
ANTE-SG v2: predictive synergy test with an *exactly* valid bounded score.

Differences from ``synergistic_granger.ante_sg`` (kept untouched for reproducing
the original submission):

1.  Bounded-loss score (no clipping of the score).  Out-of-sample losses are
    truncated, ell_c(e) = min(e^2, c^2).  Because ell_c is 2c-Lipschitz in e and
    both competing predictions p_t^1, p_t^2 are F_{t-1}-measurable, the score
    gap of a nested pair obeys the *predictable* bound
        |ell_c(Y-p^1) - ell_c(Y-p^2)| <= 2c |p^1 - p^2| =: b_t .
    Dividing by the predictable positive b_t gives u_t in [-1, 1] with
    E[u_t | F_{t-1}] = E[s_t | F_{t-1}] / b_t, so the sign of the conditional
    mean is preserved EXACTLY (no clipping, no symmetry assumption) and the
    betting factor 1 + lambda_t u_t is positive for any predictable
    lambda_t in [0, gamma], gamma < 1.
2.  Two score definitions:
      'addjoint' : s = ell(add) - ell(joint)           (removes redundancy)
      'four'     : s = ell(A) + ell(B) - ell(joint) - ell(base)   (original)
3.  Predictable standardisation (no look-ahead from full-sample moments).
4.  Reverse-hypothesis (exclusion) e-process for H_delta: E[s_t | F_{t-1}] >= delta.
5.  Everything is derived from a single pass of forgetting-RLS errors so that
    many variants can be evaluated from the same stored errors.
"""
import numpy as np
from itertools import combinations  # noqa: F401  (kept for callers)

try:
    from synergistic_granger import ForgettingRLS
except ImportError:  # pragma: no cover
    from .synergistic_granger import ForgettingRLS


# ---------------------------------------------------------------------------
# Predictable standardisation
# ---------------------------------------------------------------------------
def predictable_standardise(x, warmup=30, zclip=None):
    """z_t = (x_t - m_{t-1}) / s_{t-1}, with m, s the expanding mean / std of
    x_1..x_{t-1}.  Entries t < warmup are set to 0.  Uses only the past, so the
    standardised series is adapted to the natural filtration."""
    x = np.asarray(x, float)
    T = len(x)
    z = np.zeros(T)
    csum = 0.0
    csq = 0.0
    for t in range(T):
        if t >= warmup:
            m = csum / t
            v = max(csq / t - m * m, 1e-12)
            z[t] = (x[t] - m) / np.sqrt(v)
        csum += x[t]
        csq += x[t] * x[t]
    if zclip is not None:
        z = np.clip(z, -zclip, zclip)
    return z


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
def _lags(series, t, p):
    return [series[t - 1 - j] for j in range(p)]


def make_features(Y, A, B, cond, p, contemp=False, feat_clip=None, extra=None):
    """Return t -> dict of feature vectors for the models
    base, A, B, add, joint.
      base  : 1, Y-lags, conditioning-lags
      A / B : base + that source's own linear+squared lags
      add   : base + A-own + B-own           (additive in A,B)
      joint : add + same-lag cross-products  (A*B)
    ``extra(t, A, B)`` may return additional joint-only features (e.g. random
    Fourier features of (A,B)) and additive-only features via a tuple
    (add_extra, joint_extra).
    """
    cond = list(cond or [])

    def f(t):
        base = [1.0] + _lags(Y, t, p)
        for z in cond:
            base += _lags(z, t, p)
        la = _lags(A, t, p)
        lb = _lags(B, t, p)
        if contemp:
            la = [A[t]] + la
            lb = [B[t]] + lb
        if feat_clip is not None:
            la = [float(np.clip(v, -feat_clip, feat_clip)) for v in la]
            lb = [float(np.clip(v, -feat_clip, feat_clip)) for v in lb]
        la2 = [v * v for v in la]
        lb2 = [v * v for v in lb]
        cross = [a * b for a, b in zip(la, lb)]
        ownA = la + la2
        ownB = lb + lb2
        add_extra = []
        joint_extra = []
        if extra is not None:
            add_extra, joint_extra = extra(t, la, lb)
        addv = base + ownA + ownB + list(add_extra)
        return {
            "base": np.array(base),
            "A": np.array(base + ownA),
            "B": np.array(base + ownB),
            "add": np.array(addv),
            "joint": np.array(addv + cross + list(joint_extra)),
        }

    return f


MODELS = ("base", "A", "B", "add", "joint")


def run_errors(Y, A, B, cond=None, p=1, contemp=False, forget=0.995, ridge=1e-2,
               feat_clip=None, extra=None, models=MODELS):
    """One pass of forgetting-RLS predictors.  Returns (pred, err) arrays of shape
    (T, len(models)); row t holds predictions made BEFORE seeing Y_t
    (F_{t-1}-measurable) and the realised errors Y_t - pred_t.  Rows t < p are 0."""
    Y = np.asarray(Y, float)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    cond = [np.asarray(z, float) for z in (cond or [])]
    T = len(Y)
    feats = make_features(Y, A, B, cond, p, contemp, feat_clip, extra)
    f0 = feats(p)
    rls = {k: ForgettingRLS(len(f0[k]), forget=forget, ridge=ridge) for k in models}
    pred = np.zeros((T, len(models)))
    for t in range(p, T):
        f = feats(t)
        for j, k in enumerate(models):
            pred[t, j] = rls[k].predict(f[k])
        for k in models:
            rls[k].update(f[k], Y[t])
    err = Y[:, None] - pred
    err[:p] = 0.0
    return pred, err


# ---------------------------------------------------------------------------
# Scores from stored predictions
# ---------------------------------------------------------------------------
def _idx(models, name):
    return list(models).index(name)


def bounded_score(Y, pred, models=MODELS, kind="addjoint", cap=3.0):
    """Truncated-loss score s_t and its predictable bound b_t.
    kind='addjoint': s = l(add) - l(joint),  b = 2c|p_add - p_joint|
    kind='four'    : s = l(A)+l(B)-l(joint)-l(base),
                     b = 2c(|p_A - p_joint| + |p_B - p_base|)
    ell(e) = min(e^2, cap^2).
    """
    Y = np.asarray(Y, float)
    e = Y[:, None] - pred
    L = np.minimum(e ** 2, cap ** 2)
    g = lambda n: _idx(models, n)
    if kind == "addjoint":
        s = L[:, g("add")] - L[:, g("joint")]
        b = 2 * cap * np.abs(pred[:, g("add")] - pred[:, g("joint")])
    elif kind == "four":
        s = L[:, g("A")] + L[:, g("B")] - L[:, g("joint")] - L[:, g("base")]
        b = 2 * cap * (np.abs(pred[:, g("A")] - pred[:, g("joint")])
                       + np.abs(pred[:, g("B")] - pred[:, g("base")]))
    else:
        raise ValueError(kind)
    return s, b


def raw_score(Y, pred, models=MODELS, kind="addjoint"):
    """Untruncated squared-error score (legacy loss)."""
    Y = np.asarray(Y, float)
    L = (Y[:, None] - pred) ** 2
    g = lambda n: _idx(models, n)
    if kind == "addjoint":
        return L[:, g("add")] - L[:, g("joint")]
    return L[:, g("A")] + L[:, g("B")] - L[:, g("joint")] - L[:, g("base")]


def cw_adjustment(pred, models=MODELS):
    """Clark-West (2007) adjustment term (p_add - p_joint)^2 for the nested pair."""
    g = lambda n: _idx(models, n)
    return (pred[:, g("add")] - pred[:, g("joint")]) ** 2


# ---------------------------------------------------------------------------
# Betting e-processes on a normalised bounded score u_t in [-1, 1]
# ---------------------------------------------------------------------------
def bet_process(u, gamma=0.9, warmup=50, start=0, ceil=1e15):
    """One-sided plug-in Kelly bet on u_t in [-1,1].  lambda_t = clip(mu/v, 0, gamma)
    from the expanding running mean / second moment of u_s, s<t (predictable);
    lambda_t = 0 during warm-up.  Returns (E trajectory, lambda trajectory)."""
    T = len(u)
    E = 1.0
    traj = np.ones(T)
    lam_tr = np.zeros(T)
    s1 = s2 = 0.0
    cnt = 0
    for t in range(start, T):
        if cnt >= warmup:
            mu = s1 / cnt
            v = s2 / cnt + 1e-12
            lam = min(max(mu / v, 0.0), gamma)
        else:
            lam = 0.0
        E = min(E * (1.0 + lam * u[t]), ceil)
        traj[t] = E
        lam_tr[t] = lam
        s1 += u[t]
        s2 += u[t] * u[t]
        cnt += 1
    return traj, lam_tr


def normalise(s, b):
    """u = s / b with u = 0 where b = 0 (then s = 0 exactly)."""
    u = np.zeros_like(s)
    ok = b > 1e-12
    u[ok] = s[ok] / b[ok]
    return np.clip(u, -1.0, 1.0)  # numerical safety only: |s|<=b holds analytically


def exclusion_bounds(s, b, deltas, thr, gamma=0.9, warmup=50, start=0):
    """Reverse-hypothesis e-processes.  For each delta, K_t(delta) is the capital
    of a bettor against H_delta: E[s_t | F_{t-1}] >= delta for all t, betting on
    w_t = (delta - s_t)/(b_t + delta) in [-1,1] (E[w_t|F]<=0 under H_delta).
    The bet lambda_t is computed separately for each delta, so error control over a
    finite grid G needs the Bonferroni level alpha/|G| (pass thr = |G|/alpha).
    Returns, per delta, max_t K_t(delta) and the smallest delta whose running
    maximum crossed ``thr`` (np.nan if none)."""
    deltas = np.asarray(deltas, float)
    T = len(s)
    out_max = np.ones(len(deltas))
    for i, d in enumerate(deltas):
        den = b + d
        w = np.where(den > 1e-12, (d - s) / np.where(den > 1e-12, den, 1.0), 0.0)
        w = np.clip(w, -1.0, 1.0)
        K = 1.0
        s1 = s2 = 0.0
        cnt = 0
        mx = 1.0
        for t in range(start, T):
            if cnt >= warmup:
                mu = s1 / cnt
                v = s2 / cnt + 1e-12
                lam = min(max(mu / v, 0.0), gamma)
            else:
                lam = 0.0
            K *= (1.0 + lam * w[t])
            mx = max(mx, K)
            s1 += w[t]
            s2 += w[t] * w[t]
            cnt += 1
        out_max[i] = mx
    crossed = np.where(out_max >= thr)[0]
    return out_max, (float(deltas[crossed[0]]) if len(crossed) else np.nan)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def ante_sg_v2(Y, A, B, cond=None, p=1, contemp=False, forget=0.995, ridge=1e-2,
               kind="addjoint", cap=3.0, gamma=0.9, warmup=50, alpha=0.05,
               feat_clip=None, extra=None):
    Y = np.asarray(Y, float)
    pred, _ = run_errors(Y, A, B, cond, p, contemp, forget, ridge, feat_clip, extra)
    s, b = bounded_score(Y, pred, MODELS, kind, cap)
    u = normalise(s, b)
    E, lam = bet_process(u, gamma, warmup, start=p)
    rej = np.where(E >= 1.0 / alpha)[0]
    return dict(E=E, lam=lam, s=s, b=b, u=u, final_E=float(E[-1]), max_E=float(E.max()),
                reject_time=int(rej[0]) if len(rej) else None,
                rejected=bool(len(rej)), mean_s=float(np.mean(s[p:])), pred=pred)


# ---------------------------------------------------------------------------
# k-source group test (add vs joint with all same-lag interaction orders 2..k)
# ---------------------------------------------------------------------------
def ante_group_v2(Y, sources, cond=None, p=1, contemp=False, forget=0.995, ridge=1e-2,
                  cap=3.0, gamma=0.9, warmup=50, alpha=0.05, feat_clip=None):
    """Exactly valid group-synergy test for k sources acting on Y.
    additive model : base + each source's own linear+squared lags
    joint model    : additive + all same-lag products of order 2..k
    Score = truncated-loss gap, normalised by the predictable bound b_t."""
    from itertools import combinations
    Y = np.asarray(Y, float)
    srcs = [np.asarray(s, float) for s in sources]
    cond = [np.asarray(z, float) for z in (cond or [])]
    k = len(srcs)
    T = len(Y)

    def feats(t):
        base = [1.0] + _lags(Y, t, p)
        for z in cond:
            base += _lags(z, t, p)
        lags = []
        own = []
        for s in srcs:
            ls = _lags(s, t, p)
            if contemp:
                ls = [s[t]] + ls
            if feat_clip is not None:
                ls = [float(np.clip(v, -feat_clip, feat_clip)) for v in ls]
            lags.append(ls)
            own += ls + [v * v for v in ls]
        cross = []
        for order in range(2, k + 1):
            for combo in combinations(range(k), order):
                prod = [1.0] * len(lags[0])
                for idx in combo:
                    prod = [a * b for a, b in zip(prod, lags[idx])]
                cross += prod
        return np.array(base + own), np.array(base + own + cross)

    fa, fj = feats(p)
    r_add = ForgettingRLS(len(fa), forget, ridge)
    r_joint = ForgettingRLS(len(fj), forget, ridge)
    pred = np.zeros((T, 2))
    for t in range(p, T):
        fa, fj = feats(t)
        pred[t, 0] = r_add.predict(fa)
        pred[t, 1] = r_joint.predict(fj)
        r_add.update(fa, Y[t])
        r_joint.update(fj, Y[t])
    L = np.minimum((Y[:, None] - pred) ** 2, cap ** 2)
    s = L[:, 0] - L[:, 1]
    b = 2 * cap * np.abs(pred[:, 0] - pred[:, 1])
    s[:p] = 0.0
    b[:p] = 0.0
    u = normalise(s, b)
    E, lam = bet_process(u, gamma, warmup, start=p)
    rej = np.where(E >= 1.0 / alpha)[0]
    return dict(E=E, lam=lam, s=s, b=b, u=u, k=k, final_E=float(E[-1]), max_E=float(E.max()),
                reject_time=int(rej[0]) if len(rej) else None, rejected=bool(len(rej)),
                mean_s=float(np.mean(s[p:])))
