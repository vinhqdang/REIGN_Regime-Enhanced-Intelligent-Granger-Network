"""
Synthetic data generators for sequential synergistic-causality experiments.

Each generator returns a stream of i.i.d. observations (A, B, [D,] C) where A, B
(and optionally D) are randomized binary "candidate causes" and C is a binary
effect. Randomization (known propensity pi=0.5 per factor) makes the additive
interaction contrast identified without confounding, which keeps the per-sample
IPW interaction score unbiased under the null.

The additive interaction contrast (RERI on the risk-difference scale) for two
binary causes is

    theta = mu_11 - mu_10 - mu_01 + mu_00 ,   mu_ab = E[C | do(A=a, B=b)].

theta = 0 is the "no synergy / joint effect = sum of parts" null.
"""
import numpy as np


def _bernoulli(rng, p, size):
    return (rng.random(size) < p).astype(int)


def gen_null(n, rng, base=0.15, ea=0.25, eb=0.20):
    """Additive main effects, NO interaction  ->  theta = 0 (the null).

    p(C=1 | A,B) = base + ea*A + eb*B   (clipped to [0,1]); theta == 0 exactly.
    """
    A = _bernoulli(rng, 0.5, n)
    B = _bernoulli(rng, 0.5, n)
    p = np.clip(base + ea * A + eb * B, 0.0, 1.0)
    C = (rng.random(n) < p).astype(int)
    theta = 0.0
    return dict(A=A, B=B, C=C, theta=theta)


def gen_and(n, rng, base=0.15, ea=0.15, eb=0.15, syn=0.35):
    """AND-gate positive synergy: extra risk only when A=B=1  ->  theta = syn > 0."""
    A = _bernoulli(rng, 0.5, n)
    B = _bernoulli(rng, 0.5, n)
    p = np.clip(base + ea * A + eb * B + syn * (A * B), 0.0, 1.0)
    C = (rng.random(n) < p).astype(int)
    # theta = mu11-mu10-mu01+mu00 = (base+ea+eb+syn)-(base+ea)-(base+eb)+base = syn
    return dict(A=A, B=B, C=C, theta=syn)


def gen_xor(n, rng, flip=0.05):
    """XOR mechanism: pure informational synergy.

    C = A XOR B with a small label-flip. Pairwise MI I(A;C)=I(B;C)=0 but the
    joint I(A,B;C) is large. On the additive scale theta = mu11-mu10-mu01+mu00
    = flip - (1-flip) - (1-flip) + flip = 2*(2*flip - 1)  (strongly negative),
    so a two-sided additive-interaction e-process still detects it.
    """
    A = _bernoulli(rng, 0.5, n)
    B = _bernoulli(rng, 0.5, n)
    pure = np.bitwise_xor(A, B)
    flips = rng.random(n) < flip
    C = np.where(flips, 1 - pure, pure)
    theta = 2.0 * (2.0 * flip - 1.0)
    return dict(A=A, B=B, C=C, theta=theta)


def gen_triple(n, rng, base=0.12, main=0.10, syn=0.4, synergistic_pair=("A", "B")):
    """Three candidate causes A,B,D; a synergistic AND only on one *pair*.

    Used to test subset-refinement: the method should localise synergy to the
    true interacting pair and not fire on the innocent third variable.
    """
    A = _bernoulli(rng, 0.5, n)
    B = _bernoulli(rng, 0.5, n)
    D = _bernoulli(rng, 0.5, n)
    idx = {"A": A, "B": B, "D": D}
    u, v = synergistic_pair
    p = np.clip(base + main * A + main * B + main * D + syn * (idx[u] * idx[v]),
                0.0, 1.0)
    C = (rng.random(n) < p).astype(int)
    return dict(A=A, B=B, D=D, C=C, theta_pair={synergistic_pair: syn})


GENERATORS = {
    "null": gen_null,
    "and": gen_and,
    "xor": gen_xor,
    "triple": gen_triple,
}
