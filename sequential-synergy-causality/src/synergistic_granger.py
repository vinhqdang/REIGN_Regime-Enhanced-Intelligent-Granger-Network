"""
ANTE-SG : Anytime-valid Nonparametric Test of synErgy for Synergistic Granger
causality in (financial) time series.

Idea
----
A *group* {A,B} is a synergistic Granger cause of target Y if the joint lagged
history of A and B predicts Y beyond the SUM of what each adds individually.
Define one-step-ahead predictive losses of four nested online predictors of Y_t
(all using only information up to t-1):

    base : uses  Y-own lags (+ optional conditioning set Z lags)
    +A   : base + A lags
    +B   : base + B lags
    +AB  : base + A lags + B lags

Per-step synergy score (realized, out-of-sample):

    s_t = l^A_t + l^B_t - l^{AB}_t - l^{base}_t
        = ( l^A_t - l^{AB}_t )  -  ( l^{base}_t - l^B_t )
          \_______________/       \_________________/
          gain from adding B       gain from adding B
          on top of A              on top of nothing

E[s_t] equals the interaction of predictive gains,
    Syn(A,B->Y) = Delta({A,B}) - Delta({A}) - Delta({B}),
the "joint minus sum of parts" on the predictive-information scale (the Gaussian
case coincides with an interaction of conditional transfer entropies).

Null (prequential / game-theoretic):  H0 :  E[s_t | F_{t-1}] <= 0  for all t
("the joint predictor has no super-additive advantage given the past").  This
needs NO stationarity and NO correct model specification -- it is a statement
about realized out-of-sample predictability, which is what makes it suitable for
nonstationary financial data.

Test by betting: with a predictable fraction lambda_t >= 0 and a bounded score,

    E_t = prod_{s<=t} ( 1 + lambda_s * u_s ),   u_s = clip(s_s / scale_{s-1}, -b, b),

is a nonnegative supermartingale under H0, so Ville's inequality gives
P_{H0}(exists t : E_t >= 1/alpha) <= alpha : anytime-valid, monitor/stop anytime.
Regime shifts are absorbed by a forgetting factor in the online predictors.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Forgetting recursive least squares (one online linear predictor)
# ---------------------------------------------------------------------------
class ForgettingRLS:
    """Recursive least squares with exponential forgetting (regime adaptivity)."""

    def __init__(self, dim, forget=0.995, ridge=1e-2):
        self.d = dim
        self.lam = forget
        self.w = np.zeros(dim)
        self.P = np.eye(dim) / ridge

    def predict(self, x):
        return float(self.w @ x)

    def update(self, x, y):
        P, lam = self.P, self.lam
        Px = P @ x
        denom = lam + x @ Px
        k = Px / denom
        err = y - self.w @ x
        self.w = self.w + k * err
        self.P = (P - np.outer(k, Px)) / lam


# ---------------------------------------------------------------------------
# Lag-feature construction
# ---------------------------------------------------------------------------
def _lagblock(series, t, p):
    return [series[t - 1 - j] for j in range(p)]


def build_feature_fn(Y, sources, cond, p, contemp=False):
    """Return a function t -> {model_name: feature_vector} for base/+A/+B/+AB.

    To measure *synergy* (irreducible joint structure) rather than within-variable
    nonlinearity, each marginal model gets its source's own linear AND squared
    terms, while ONLY the joint model additionally gets the cross-products.  Thus
    a DGP like Y=A^2 is absorbed by the A-model (no false synergy), whereas Y=A*B
    is captured only by the joint model (true synergy).

    contemp=False -> lagged (Granger) synergy: features are A_{t-i},B_{t-i}, i=1..p
                     (does the joint PAST predict Y_t beyond parts).
    contemp=True  -> contemporaneous synergy: additionally include the time-t
                     source values A_t,B_t and the joint model's cross A_t*B_t
                     (does the joint CONTEMPORANEOUS state explain Y_t beyond
                     parts).  Predictors are still fit only on the past, so the
                     betting martingale stays valid.
    """
    A = sources["A"]; B = sources["B"]

    def feats(t):
        base = [1.0] + _lagblock(Y, t, p)
        for z in cond:
            base += _lagblock(z, t, p)
        la = _lagblock(A, t, p); la2 = [v * v for v in la]
        lb = _lagblock(B, t, p); lb2 = [v * v for v in lb]
        cross = [a * b for a, b in zip(la, lb)]          # same-lag interaction
        if contemp:
            a0, b0 = A[t], B[t]
            la = [a0] + la; la2 = [a0 * a0] + la2
            lb = [b0] + lb; lb2 = [b0 * b0] + lb2
            cross = [a0 * b0] + cross
        return {
            "base": np.array(base),
            "A":    np.array(base + la + la2),
            "B":    np.array(base + lb + lb2),
            "AB":   np.array(base + la + la2 + lb + lb2 + cross),
        }
    return feats


# ---------------------------------------------------------------------------
# ANTE-SG detector
# ---------------------------------------------------------------------------
def ante_sg(Y, A, B, cond=None, p=1, alpha=0.05, forget=0.995, ridge=1e-2,
            warmup=60, clip_b=10.0, cap_frac=0.9, ceil=1e15, contemp=False):
    """Sequential anytime-valid test for positive synergy of {A,B} on Y.

    contemp=False tests lagged (Granger) synergy; contemp=True tests
    contemporaneous synergy (time-t joint state). Returns dict with the e-value
    trajectory, rejection decision/time, running synergy-score mean, per-step scores.
    """
    Y = np.asarray(Y, float)
    cond = [np.asarray(z, float) for z in (cond or [])]
    feats = build_feature_fn(Y, {"A": np.asarray(A, float), "B": np.asarray(B, float)},
                             cond, p, contemp=contemp)
    T = len(Y)

    dims = {k: len(feats(p)[k]) for k in ("base", "A", "B", "AB")}
    rls = {k: ForgettingRLS(dims[k], forget=forget, ridge=ridge) for k in dims}

    lam_max = cap_frac / clip_b
    E = 1.0
    E_traj = np.ones(T)
    scores = np.full(T, np.nan)
    # predictable running stats of the (unnormalised) score and its scale
    s1 = s2 = 0.0; cnt = 0
    ew_sq = 1e-8; ew_beta = 0.98
    reject_time = None

    for t in range(p, T):
        f = feats(t)
        # out-of-sample (predictable) losses: predict BEFORE updating
        losses = {k: (Y[t] - rls[k].predict(f[k])) ** 2 for k in rls}
        s = losses["A"] + losses["B"] - losses["AB"] - losses["base"]
        scores[t] = s
        # predictable scale (uses info up to t-1 only)
        scale = np.sqrt(ew_sq)
        u = np.clip(s / scale, -clip_b, clip_b)
        # predictable one-sided betting fraction (GRAPA, tests synergy > 0)
        if cnt >= max(warmup, 1):
            mu = s1 / cnt
            v = s2 / cnt + 1e-12
            lam = max(mu / v, 0.0) / scale        # scale-match to normalised u
            lam = min(lam, lam_max)
        else:
            lam = 0.0
        E = min(E * (1.0 + lam * u), ceil)
        E_traj[t] = E
        if reject_time is None and E >= 1.0 / alpha:
            reject_time = t
        # ---- updates (make everything above predictable) ----
        for k in rls:
            rls[k].update(f[k], Y[t])
        s1 += s; s2 += s * s; cnt += 1
        ew_sq = ew_beta * ew_sq + (1 - ew_beta) * s * s

    return dict(E=E_traj, reject_time=reject_time, rejected=reject_time is not None,
                final_E=float(E_traj[-1]), scores=scores,
                mean_score=float(np.nanmean(scores[p:])))


# ---------------------------------------------------------------------------
# Multi-source (group) synergy: does a GROUP {X1..Xk} jointly cause Y beyond the
# sum of parts?  Synergy = out-of-sample gain of the JOINT model (own terms + all
# pairwise cross-products) over the ADDITIVE model (own terms only).
# ---------------------------------------------------------------------------
def ante_group(Y, sources, cond=None, p=1, alpha=0.05, forget=0.995, ridge=1e-2,
               warmup=60, clip_b=10.0, cap_frac=0.9, ceil=1e15, contemp=False):
    """Sequential anytime-valid test for GROUP synergy of {X1..Xk} on Y.

    ADDITIVE model: base(Y,Z lags) + each source's own linear+squared terms.
    JOINT model:    additive + all pairwise same-lag cross-products X_i*X_j.
    Per-step score s_t = loss_additive_t - loss_joint_t (>0 => the group predicts
    Y super-additively).  Under H0 (no group synergy) E[s_t|F_{t-1}] <= 0, so the
    betting process is an e-process (Ville anytime-valid).
    """
    Y = np.asarray(Y, float)
    srcs = [np.asarray(s, float) for s in sources]
    cond = [np.asarray(z, float) for z in (cond or [])]
    k = len(srcs); T = len(Y)

    def feats(t):
        base = [1.0] + _lagblock(Y, t, p)
        for z in cond:
            base += _lagblock(z, t, p)
        own = []
        lags = []
        for s in srcs:
            ls = _lagblock(s, t, p)
            if contemp:
                ls = [s[t]] + ls
            lags.append(ls)
            own += ls + [v * v for v in ls]
        cross = []
        from itertools import combinations as _cmb
        for order in range(2, k + 1):                # all interactions order 2..k
            for combo in _cmb(range(k), order):
                prod = [1.0] * p
                for idx in combo:
                    prod = [pp * v for pp, v in zip(prod, lags[idx])]
                cross += prod
        add = np.array(base + own)
        joint = np.array(base + own + cross)
        return add, joint

    d_add = len(feats(p)[0]); d_joint = len(feats(p)[1])
    rls_add = ForgettingRLS(d_add, forget, ridge)
    rls_joint = ForgettingRLS(d_joint, forget, ridge)
    lam_max = cap_frac / clip_b
    E = 1.0; E_traj = np.ones(T); scores = np.full(T, np.nan)
    s1 = s2 = 0.0; cnt = 0; ew = 1e-8; reject_time = None
    for t in range(p, T):
        fa, fj = feats(t)
        la = (Y[t] - rls_add.predict(fa)) ** 2
        lj = (Y[t] - rls_joint.predict(fj)) ** 2
        s = la - lj                                  # >0: joint (group) beats additive
        scores[t] = s
        scale = np.sqrt(ew)
        u = np.clip(s / scale, -clip_b, clip_b)
        lam = (max(s1 / cnt, 0.0) / (s2 / cnt + 1e-12) / scale) if cnt >= max(warmup, 1) else 0.0
        lam = min(lam, lam_max)
        E = min(E * (1.0 + lam * u), ceil); E_traj[t] = E
        if reject_time is None and E >= 1.0 / alpha:
            reject_time = t
        rls_add.update(fa, Y[t]); rls_joint.update(fj, Y[t])
        s1 += s; s2 += s * s; cnt += 1; ew = 0.98 * ew + 0.02 * s * s
    return dict(E=E_traj, reject_time=reject_time, rejected=reject_time is not None,
                final_E=float(E_traj[-1]), k=k, mean_score=float(np.nanmean(scores[p:])))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def batch_synergy_estimate(Y, A, B, cond=None, p=1, ridge=1e-2, contemp=False):
    """In-sample OLS/ridge estimate of the synergy Syn(A,B->Y) (fixed-sample)."""
    Y = np.asarray(Y, float)
    cond = [np.asarray(z, float) for z in (cond or [])]
    feats = build_feature_fn(Y, {"A": np.asarray(A, float), "B": np.asarray(B, float)},
                             cond, p, contemp=contemp)
    T = len(Y)
    rows = {k: [] for k in ("base", "A", "B", "AB")}
    ys = []
    for t in range(p, T):
        f = feats(t)
        for k in rows:
            rows[k].append(f[k])
        ys.append(Y[t])
    ys = np.array(ys)
    def mse(k):
        X = np.array(rows[k]); d = X.shape[1]
        w = np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ ys)
        return float(np.mean((ys - X @ w) ** 2))
    L = {k: mse(k) for k in rows}
    return L["A"] + L["B"] - L["AB"] - L["base"]   # >0 => super-additive synergy


def peeking_synergy_test(Y, A, B, cond=None, p=1, alpha=0.05, warmup=60):
    """Fixed-sample one-sided z-test on the synergy score, evaluated at EVERY t.

    This is the *invalid* continuously-monitored classical test used to show
    type-I inflation vs. the anytime-valid e-process.
    """
    from scipy.stats import norm
    res = ante_sg(Y, A, B, cond=cond, p=p, alpha=alpha, warmup=0)  # reuse scores
    s = res["scores"]
    s = s[~np.isnan(s)]
    zc = norm.ppf(1 - alpha)
    n = len(s)
    cummean = np.cumsum(s) / np.arange(1, n + 1)
    cumvar = np.cumsum(s ** 2) / np.arange(1, n + 1) - cummean ** 2
    se = np.sqrt(np.maximum(cumvar, 1e-18) / np.arange(1, n + 1))
    z = cummean / se
    z[:warmup] = 0.0
    crossed = np.where(z >= zc)[0]
    return dict(rejected=len(crossed) > 0,
                reject_time=int(crossed[0]) if len(crossed) else None, z=z)


def pairwise_granger_f(Y, X, p=1):
    """Classical Granger F-test that X Granger-causes Y (individual, not joint)."""
    from scipy.stats import f as fdist
    Y = np.asarray(Y, float); X = np.asarray(X, float)
    T = len(Y)
    yy, Xr, Xf = [], [], []
    for t in range(p, T):
        yy.append(Y[t])
        base = [1.0] + [Y[t - 1 - j] for j in range(p)]
        Xr.append(base)
        Xf.append(base + [X[t - 1 - j] for j in range(p)])
    yy = np.array(yy); Xr = np.array(Xr); Xf = np.array(Xf)
    def rss(Xm):
        w, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
        return float(np.sum((yy - Xm @ w) ** 2))
    r0, r1 = rss(Xr), rss(Xf)
    n = len(yy); q = p; k = Xf.shape[1]
    F = ((r0 - r1) / q) / (r1 / (n - k))
    pval = 1 - fdist.cdf(F, q, n - k)
    return dict(F=float(F), pval=float(pval))
