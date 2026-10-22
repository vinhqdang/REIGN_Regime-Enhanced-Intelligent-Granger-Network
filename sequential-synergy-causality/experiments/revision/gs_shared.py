"""Shared helpers for the R9-R11 experiments.

* gs_boundaries : group-sequential boundary constants for Pocock and O'Brien-Fleming
  type Lan-DeMets alpha spending, found by Monte Carlo over Brownian look statistics
  (correlated looks, sequential crossing rule).  The spending functions are
      Pocock : alpha * log(1 + (e-1) t)
      OBF    : 2 - 2 Phi(z_{alpha/2} / sqrt(t))
  The Monte Carlo total crossing probability is returned as a check (should be ~alpha).
* v2_addjoint : one ante_v2 pass (run_errors -> bounded addjoint score -> normalise ->
  bet_process) returning the e-trajectory and summaries.
* legacy_run  : wrapper around synergistic_granger.ante_sg.
"""
import os, sys
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from ante_v2 import run_errors, bounded_score, normalise, bet_process, raw_score  # noqa: E402
from synergistic_granger import ante_sg  # noqa: E402


def spend(t, kind, alpha):
    """Cumulative alpha spent at information fraction t."""
    t = np.asarray(t, float)
    if kind == "pocock":
        return alpha * np.log(1.0 + (np.e - 1.0) * t)
    if kind == "obf":
        return 2.0 - 2.0 * norm.cdf(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(t))
    raise ValueError(kind)


def gs_boundaries(info, kind, alpha=0.05, sided="two", n_paths=200000, seed=12345):
    """Boundary constants c_k so that P(exists k: S_k >= c_k) ~= alpha under the
    Brownian model Z_k = W(t_k)/sqrt(t_k), S = |Z| (two-sided) or Z (one-sided).
    Crossing is checked sequentially: at look k, the alive paths (not stopped
    earlier) are cut at the quantile giving the spent increment alpha(t_k)-alpha(t_{k-1}).
    Returns (c, mc_total_alpha)."""
    info = np.asarray(info, float)
    K = len(info)
    rng = np.random.default_rng(seed)
    dt = np.diff(np.concatenate([[0.0], info]))
    W = np.cumsum(rng.standard_normal((n_paths, K)) * np.sqrt(dt), axis=1)
    Z = W / np.sqrt(info)
    S = np.abs(Z) if sided == "two" else Z
    alive = np.ones(n_paths, bool)
    c = np.full(K, np.inf)
    prev = 0.0
    for k in range(K):
        cum = float(spend(info[k], kind, alpha))
        m = int(round((cum - prev) * n_paths))
        prev = cum
        idx = np.where(alive)[0]
        if m > 0 and len(idx) > 0:
            vals = np.sort(S[idx, k])[::-1]
            c[k] = vals[min(m, len(vals)) - 1]
        alive &= ~(S[:, k] >= c[k])
    mc_total = float(1.0 - alive.mean())
    return c, mc_total


def v2_addjoint(Y, A, B, cond=None, p=1, kind="addjoint", alpha=0.05, cap=3.0,
                gamma=0.9, warmup=50, forget=0.995, ridge=1e-2):
    """ante_v2 test with the settings used throughout the revision (defaults of ante_v2)."""
    Y = np.asarray(Y, float)
    pred, _ = run_errors(Y, A, B, cond, p, False, forget, ridge)
    s, b = bounded_score(Y, pred, kind=kind, cap=cap)
    u = normalise(s, b)
    E, lam = bet_process(u, gamma, warmup, start=p)
    rej = np.where(E >= 1.0 / alpha)[0]
    return dict(E=E, lam=lam, s=s, pred=pred,
                final_log10=float(np.log10(max(E[-1], 1e-300))),
                max_log10=float(np.log10(max(E.max(), 1e-300))),
                max_E=float(E.max()),
                reject_time=int(rej[0]) if len(rej) else None,
                rejected=bool(len(rej)))


def legacy_run(Y, A, B, cond=None, p=1, alpha=0.05):
    r = ante_sg(Y, A, B, cond=cond, p=p, alpha=alpha)
    return dict(final_log10=float(np.log10(max(r["final_E"], 1e-300))),
                max_log10=float(np.log10(max(r["E"].max(), 1e-300))),
                reject_time=r["reject_time"], rejected=bool(r["rejected"]),
                E=r["E"], scores=r["scores"])
