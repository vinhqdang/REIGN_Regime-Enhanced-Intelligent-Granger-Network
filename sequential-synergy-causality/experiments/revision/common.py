"""Shared helpers for the revision experiments (simulation generators, finance
loader, summary statistics).  Import with:
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(__file__))
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def rate_ci(flags):
    flags = np.asarray(flags, bool)
    k, n = int(flags.sum()), len(flags)
    lo, hi = wilson(k, n)
    return dict(rate=k / n if n else np.nan, k=k, n=n, lo=lo, hi=hi)


def auc_ci(pos, neg, n_boot=1000, seed=0):
    """Mann-Whitney AUC with percentile-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)

    def auc(p, n):
        s = np.concatenate([p, n])
        order = np.argsort(s, kind="mergesort")
        r = np.empty(len(s))
        r[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
        return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

    a = auc(pos, neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return dict(auc=float(a), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


# ------------------------------------------------------------------ simulation
def gen_var(kind, T, rng, rho=0.2, strength=0.9, heavy=False, skew=False,
            garch=False, ar=0.3):
    """Common-factor VAR with two sources A,B (correlation controlled by rho via a
    shared innovation F), market factor Z, target Y.

    kinds:
      'null'   : Y_t = .5 A_{t-1} + .5 B_{t-1} + .3 Z_t + e   (additive)
      'syn'    : Y_t = strength*A_{t-1}B_{t-1} + .2 A_{t-1} + .3 Z_t + e
      'own'    : Y_t = .6 A_{t-1}^2 + .4 B_{t-1} + .3 Z_t + e   (own nonlinearity)
      'cubic'  : Y_t = .5 A_{t-1}^3/3 + .5 B_{t-1} + .3 Z_t + e (additive, f outside span)
      'break'  : additive, coefficients flip sign at T/2 (nonstationary, still additive)
      'a2b'    : Y_t = strength*A_{t-1}^2 B_{t-1} + .2 A + .3 Z + e (interaction outside span)
      'tanh'   : Y_t = strength*tanh(2A_{t-1})*tanh(2B_{t-1}) + .2 A + .3 Z + e
      'sinab'  : Y_t = strength*sin(A_{t-1}*B_{t-1}) + .2 A + .3 Z + e
    heavy: Student-t(3) innovations (unit variance); skew: add exp-skewed innovation;
    garch: GARCH(1,1)-type conditional variance for the target innovation.
    corr(A_t,B_t) ~= rho / (1 + ...) roughly rho in the stationary innovation scale.
    """
    def innov(n):
        if heavy:
            x = rng.standard_t(3, n) / np.sqrt(3.0)
        else:
            x = rng.standard_normal(n)
        if skew:
            x = x + (rng.exponential(1.0, n) - 1.0)
            x = x / np.sqrt(2.0)
        return x

    F = rng.standard_normal(T)
    Z = rng.standard_normal(T)
    ea, eb, ey = innov(T), innov(T), innov(T)
    if garch:
        z = ey.copy()
        h = 1.0
        eps_prev = 0.0
        for t in range(T):
            h = 0.05 + 0.10 * eps_prev ** 2 + 0.85 * h
            eps_prev = z[t] * np.sqrt(h)
            ey[t] = eps_prev
        ey = ey / ey.std()
    A = np.zeros(T)
    B = np.zeros(T)
    Y = np.zeros(T)
    for t in range(1, T):
        A[t] = ar * A[t - 1] + 0.5 * Z[t] + np.sqrt(rho) * F[t] + np.sqrt(1 - rho) * ea[t]
        B[t] = ar * B[t - 1] + 0.5 * Z[t] + np.sqrt(rho) * F[t] + np.sqrt(1 - rho) * eb[t]
    for t in range(1, T):
        a, b, c = A[t - 1], B[t - 1], 0.3 * Z[t]
        if kind == "null":
            Y[t] = 0.5 * a + 0.5 * b + c + ey[t]
        elif kind == "syn":
            Y[t] = strength * a * b + 0.2 * a + c + ey[t]
        elif kind == "own":
            Y[t] = 0.6 * a ** 2 + 0.4 * b + c + ey[t]
        elif kind == "cubic":
            Y[t] = 0.5 * a ** 3 / 3 + 0.5 * b + c + ey[t]
        elif kind == "break":
            s = 1.0 if t < T // 2 else -1.0
            Y[t] = s * (0.5 * a + 0.5 * b) + c + ey[t]
        elif kind == "a2b":
            Y[t] = strength * a * a * b + 0.2 * a + c + ey[t]
        elif kind == "tanh":
            Y[t] = strength * np.tanh(2 * a) * np.tanh(2 * b) + 0.2 * a + c + ey[t]
        elif kind == "sinab":
            Y[t] = strength * np.sin(a * b) + 0.2 * a + c + ey[t]
        else:
            raise ValueError(kind)
    return Y, A, B, Z


# ------------------------------------------------------------------ finance
PANELS = {
    "etf16": ("returns_panel.csv", "SPY"),         # daily, 16 assets, cond on SPY
    "stocks13": ("famous_returns.csv", "SPY"),     # daily, 13 assets, cond on SPY
    "world6": ("world_indices_returns.csv", "US_SP500"),
    "hourly12": ("famous_hourly_returns.csv", "SPY"),
    "crypto10": ("crypto_hourly_returns.csv", "BTC-USD"),
}


def load_panel(name):
    f, cond = PANELS[name]
    r = pd.read_csv(os.path.join(DATA, f), index_col=0, parse_dates=True)
    return r, cond


def target_series(r, kind, tail_q=0.90, tail_min=100):
    """Raw (un-standardised) variable panel built from returns.
    ret : r_t ;  vol : |r_t| ;  tail : 1{|r_t| > expanding q90 of |r|_{<t}} (predictable)"""
    if kind == "ret":
        return r.copy()
    if kind == "vol":
        return r.abs()
    if kind == "tail":
        a = r.abs()
        q = a.shift(1).expanding(min_periods=tail_min).quantile(tail_q)
        return (a > q).astype(float).where(q.notna()).dropna()
    raise ValueError(kind)


def weekly(r, k=5):
    """Non-overlapping k-day sums of log returns."""
    n = (len(r) // k) * k
    return r.iloc[:n].groupby(np.arange(n) // k).sum()
