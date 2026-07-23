"""
Sequential, anytime-valid tests for group (synergistic) causality.

Core object: a betting e-process (test martingale) for the null

    H0 :  theta = 0 ,   theta = mu_11 - mu_10 - mu_01 + mu_00 ,

the additive causal-interaction contrast ("joint effect = sum of parts").

Why a martingale.  Under randomization with known propensities pi(a,b), the
inverse-probability-weighted score

    psi_t = c(A_t,B_t) * C_t / pi(A_t,B_t),      c(a,b) = (-1)^{(1-a)+(1-b)}

is unbiased for theta:  E[psi_t] = sum_{a,b} c(a,b) mu_ab = theta.  Hence under
H0, E[psi_t]=0.  We then "bet" on the sequence with a predictable fraction
lambda_t:

    E_t = prod_{s<=t} (1 + lambda_s * psi_s),   |lambda_s * psi_s| < 1.

Because lambda_s is predictable and E[psi_s | F_{s-1}] = 0 under H0,
E[E_t | F_{t-1}] = E_{t-1}: (E_t) is a nonnegative martingale with E_0 = 1.
Ville's inequality gives the anytime-valid guarantee

    P_{H0}( exists t : E_t >= 1/alpha ) <= alpha,

so rejecting the first time E_t crosses 1/alpha controls type-I error uniformly
over time, with no penalty for continuous monitoring or optional stopping.

A two-sided test averages a capital process that bets on theta>0 with one that
bets on theta<0 (an average of e-processes is an e-process).
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Per-sample interaction score
# ---------------------------------------------------------------------------
def interaction_scores(A, B, C, piA=0.5, piB=0.5):
    """IPW per-observation scores psi_t with E[psi_t] = additive interaction theta.

    Randomized design => pi(a,b) = piA^a (1-piA)^(1-a) * piB^b (1-piB)^(1-b).
    With piA=piB=0.5, psi_t = 4 * c(A_t,B_t) * C_t  in [-4, 4].
    """
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C).astype(float)
    piab = (piA ** A * (1 - piA) ** (1 - A)) * (piB ** B * (1 - piB) ** (1 - B))
    c = (-1.0) ** ((1 - A) + (1 - B))          # +1 on (1,1)&(0,0), -1 on (1,0)&(0,1)
    return c * C / piab


def score_bound(piA=0.5, piB=0.5):
    """Almost-sure bound c on |psi_t| (C in [0,1]) = 1 / min-cell-propensity."""
    cells = [piA * piB, piA * (1 - piB), (1 - piA) * piB, (1 - piA) * (1 - piB)]
    return 1.0 / min(cells)


# ---------------------------------------------------------------------------
# 2. Betting e-process (two-sided) for the mean-zero null of psi
# ---------------------------------------------------------------------------
def betting_eprocess(psi, cbound, warmup=20, cap_frac=0.9):
    """Two-sided GRAPA-style betting e-process for E[psi]=0.

    Returns E_t (the two-sided e-value trajectory) and the one-sided pieces.
    lambda_t is a *predictable* truncated Kelly fraction built only from psi_{<t}.
    """
    psi = np.asarray(psi, dtype=float)
    n = len(psi)
    lam_max = cap_frac / cbound                 # keeps 1 + lam*psi > 0
    CEIL = 1e15                                 # numeric ceiling (>> any 1/alpha)

    E_plus = np.ones(n + 1)                     # bets on theta > 0
    E_minus = np.ones(n + 1)                    # bets on theta < 0
    s1 = 0.0          # running sum   sum psi
    s2 = 0.0          # running sum   sum psi^2
    cnt = 0
    for t in range(n):
        # predictable plug-in mean / second-moment estimates from psi_{<t}
        if cnt >= warmup:
            mu = s1 / cnt
            v = s2 / cnt + 1e-8
            lam = mu / v                        # Kelly-ish fraction ~ mu / E[psi^2]
        else:
            lam = 0.0
        lam_p = min(max(lam, 0.0), lam_max)
        lam_m = max(min(lam, 0.0), -lam_max)
        # capital is monotone once it crosses any sane 1/alpha, so capping at a
        # large ceiling avoids float overflow without affecting the reject decision
        E_plus[t + 1] = min(E_plus[t] * (1.0 + lam_p * psi[t]), CEIL)
        E_minus[t + 1] = min(E_minus[t] * (1.0 + lam_m * psi[t]), CEIL)
        # update running stats AFTER using them (predictability)
        s1 += psi[t]; s2 += psi[t] ** 2; cnt += 1

    E_two = 0.5 * (E_plus + E_minus)
    return E_two[1:], E_plus[1:], E_minus[1:]


def run_eprocess(A, B, C, alpha=0.05, piA=0.5, piB=0.5, **kw):
    """Convenience wrapper: scores -> e-process -> rejection time."""
    psi = interaction_scores(A, B, C, piA, piB)
    cb = score_bound(piA, piB)
    E, Ep, Em = betting_eprocess(psi, cb, **kw)
    thresh = 1.0 / alpha
    crossed = np.where(E >= thresh)[0]
    reject_time = int(crossed[0]) + 1 if len(crossed) else None
    return dict(E=E, E_plus=Ep, E_minus=Em, threshold=thresh,
                reject_time=reject_time, rejected=reject_time is not None)


# ---------------------------------------------------------------------------
# 3. Fixed-n Wald RERI test with repeated peeking (the baseline that mis-behaves)
# ---------------------------------------------------------------------------
def wald_peeking(A, B, C, alpha=0.05, piA=0.5, piB=0.5, min_n=30):
    """Classical IPW estimate of theta + Wald z-test evaluated at every t.

    Returns the trajectory of |z| and the first crossing of z_{alpha/2}, i.e.
    what a fixed-sample test does when (invalidly) monitored continuously.
    """
    from scipy.stats import norm
    psi = interaction_scores(A, B, C, piA, piB)
    n = len(psi)
    zcrit = norm.ppf(1 - alpha / 2)
    zt = np.full(n, np.nan)
    running_mean = np.cumsum(psi) / np.arange(1, n + 1)
    running_var = np.cumsum(psi ** 2) / np.arange(1, n + 1) - running_mean ** 2
    se = np.sqrt(np.maximum(running_var, 1e-12) / np.arange(1, n + 1))
    zt = np.abs(running_mean) / se
    zt[:min_n] = 0.0
    crossed = np.where(zt >= zcrit)[0]
    reject_time = int(crossed[0]) + 1 if len(crossed) else None
    return dict(z=zt, zcrit=zcrit, reject_time=reject_time,
                rejected=reject_time is not None, theta_hat=running_mean)


# ---------------------------------------------------------------------------
# 4. Subset-refinement search over candidate cause-pairs
# ---------------------------------------------------------------------------
def subset_refinement(data, variables, alpha=0.05, **kw):
    """Run an anytime-valid e-process for every candidate pair of causes.

    Family-wise validity: with a per-pair threshold 1/alpha and m pairs, a union
    bound gives FWER <= m*alpha; dividing alpha by m (e-value Bonferroni) or
    e-BH restores control.  We report per-pair e-values so either calibration can
    be applied downstream.
    """
    from itertools import combinations
    C = data["C"]
    pairs = list(combinations(variables, 2))
    m = len(pairs)
    results = {}
    for (u, v) in pairs:
        res = run_eprocess(data[u], data[v], C, alpha=alpha / m, **kw)  # Bonferroni-e
        results[(u, v)] = dict(final_E=float(res["E"][-1]),
                               reject_time=res["reject_time"],
                               rejected=res["rejected"],
                               E=res["E"])
    return dict(pairs=results, n_pairs=m, alpha_each=alpha / m)
