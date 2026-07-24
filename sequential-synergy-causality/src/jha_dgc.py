"""
Faithful (simplified) re-implementation of Distributional Granger Causality
(Jha 2026, arXiv:2606.22230) -- the closest *sequential* competitor to ANTE.

DGC tests whether a SINGLE source X distributionally Granger-causes a target Y
by examining several distributional channels of Y_t (mean, scale, lower/upper
tail quantiles, 3rd/4th cumulants), calibrating each with a circular-shift
permutation p-value, and combining them by alpha-investing (anytime-valid via a
test supermartingale + Ville's inequality).

Crucially, DGC is single-source: it has NO notion of a joint/synergistic effect
of a *group* of predictors.  This module reproduces the paper's size/power
behaviour and is used to show that DGC misses group synergy that ANTE detects.
"""
import numpy as np


def _resid_own(Y, p=1):
    """Residual of Y_t regressed on its own p lags (+ intercept)."""
    T = len(Y)
    yy = Y[p:]
    X = np.column_stack([np.ones(T - p)] + [Y[p - 1 - j:T - 1 - j] for j in range(p)])
    w, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return yy - X @ w


def _channel_stats(Xlag, r):
    """Association of source lag Xlag with distributional features of the
    own-past residual r (all as |correlation|, a scale-free statistic)."""
    def abscorr(u, v):
        u = u - u.mean(); v = v - v.mean()
        d = np.sqrt((u @ u) * (v @ v))
        return abs(float(u @ v) / d) if d > 0 else 0.0
    rl = r - np.median(r)
    qL = np.quantile(r, 0.1); qU = np.quantile(r, 0.9)
    return {
        "mean":  abscorr(Xlag, r),
        "scale": abscorr(Xlag, r ** 2),
        "tailL": abscorr(Xlag, (r < qL).astype(float)),
        "tailU": abscorr(Xlag, (r > qU).astype(float)),
        "cum3":  abscorr(Xlag, r ** 3),
        "cum4":  abscorr(Xlag, r ** 4),
    }


def dgc_test(Y, X, p=1, alpha=0.05, B=200, psi=None, seed=0):
    """Distributional Granger test: does single source X distributionally
    Granger-cause Y?  Returns dict(rejected, channel, wealth_path, pvalues).

    Alpha-investing (Foster-Stine, per the paper): W0=alpha; commit a_k, reject
    channel iff P_k<=a_k; W <- W - a_k/(1-a_k) + 1{reject}*psi; global reject iff
    any channel rejects before wealth exhausts.  P_k via circular-shift perms.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float); X = np.asarray(X, float)
    psi = psi if psi is not None else alpha
    r = _resid_own(Y, p)
    Xlag = X[p - 1:len(X) - 1][:len(r)]                 # X_{t-1} aligned to r
    n = len(r)
    obs = _channel_stats(Xlag, r)
    # circular-shift permutation null for each channel (shift X only)
    channels = ["mean", "scale", "tailL", "tailU", "cum3", "cum4"]
    null = {c: np.empty(B) for c in channels}
    for bi in range(B):
        s = rng.integers(1, n)
        xs = np.roll(Xlag, s)
        st = _channel_stats(xs, r)
        for c in channels:
            null[c][bi] = st[c]
    pval = {c: (1 + np.sum(null[c] >= obs[c])) / (1 + B) for c in channels}
    # order channels by a G_r-like diagnostic (bigger cumulants -> tails/cumulants first)
    g = (np.mean(((r - r.mean()) / r.std()) ** 3) ** 2 / 6
         + np.mean(((r - r.mean()) / r.std()) ** 4) ** 2 / 24) ** 0.5
    order = (["mean", "scale", "tailL", "tailU", "cum3", "cum4"] if g < 1.0
             else ["scale", "tailL", "tailU", "cum3", "cum4", "mean"])
    W = alpha; rejected = False; which = None; path = [W]
    for c in order:
        a_k = W / 2.0                                    # admissible level to spend
        if pval[c] <= a_k:
            rejected = True; which = c
            W = W - a_k / (1 - a_k) + psi
            path.append(W); break
        W = W - a_k / (1 - a_k)
        path.append(W)
        if W <= 1e-6: break
    return dict(rejected=rejected, channel=which, pvalues=pval, wealth_path=path, Gr=float(g))
