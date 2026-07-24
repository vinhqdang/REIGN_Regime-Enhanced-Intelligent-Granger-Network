"""
Sample-efficiency benchmark: how much data does each method need to detect
genuine synergy, and does ANTE's sequential design detect earlier?

Addresses the "SURD is data-starved" caveat by *quantifying* it: sweep sample
size N and plot detection power (on Y=A*B synergy) and type-I (on additive null)
for ANTE vs batch SOTA given a calibrated permutation test at each N.

  - ANTE      : sequential -- reject if the e-value crosses 1/alpha within N.
  - SURD      : batch synergy + block-permutation null, tested once at N.
  - InterInfo : batch Gaussian co-information + permutation null at N.

Shows (i) batch methods need larger N (data-hungry, esp. SURD), (ii) ANTE reaches
power at least as fast while being usable at *every* N without recalibration, and
(iii) at large N all methods converge -- SURD is not broken, just data-hungry.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from synergistic_granger import ante_sg
from baselines import surd_synergy, interaction_information
OUT = os.path.join(HERE, "results")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
ALPHA = 0.05


def gen(kind, T, rng, strength=0.8):
    Z = rng.standard_normal(T); A = np.zeros(T); B = np.zeros(T); Y = np.zeros(T)
    for t in range(1, T):
        A[t] = 0.3*A[t-1] + 0.5*Z[t] + rng.standard_normal()
        B[t] = 0.3*B[t-1] + 0.5*Z[t] + rng.standard_normal()
    for t in range(1, T):
        c = 0.3*Z[t]
        if kind == "synergistic": Y[t] = strength*A[t-1]*B[t-1] + 0.2*A[t-1] + c + rng.standard_normal()
        else:                     Y[t] = 0.5*A[t-1] + 0.5*B[t-1] + c + rng.standard_normal()
    return Y, A, B, Z


def block_perm_pvalue(a, b, y, stat, n_perm=60, block=25, seed=0):
    """Circular-block-permutation p-value for a batch synergy statistic."""
    rng = np.random.default_rng(seed)
    obs = stat(a, b, y); n = len(y); nb = int(np.ceil(n / block))
    null = np.empty(n_perm)
    for k in range(n_perm):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s+block) % n for s in starts])[:n]
        null[k] = stat(a, b[idx], y)
    return (1 + np.sum(null >= obs)) / (1 + n_perm)


def run(Ns=(250, 500, 1000, 2000, 4000), n_rep=80, seed=0):
    rng = np.random.default_rng(seed)
    methods = ["ANTE", "SURD", "InteractionInfo"]
    power = {m: [] for m in methods}; type1 = {m: [] for m in methods}
    ante_latency = []
    for N in Ns:
        pw = {m: 0 for m in methods}; t1 = {m: 0 for m in methods}; lat = []
        for _ in range(n_rep):
            # power (synergy present)
            Y, A, B, Z = gen("synergistic", N + 1, rng)
            r = ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
            if r["rejected"]: pw["ANTE"] += 1; lat.append(r["reject_time"])
            a, b, y = A[:-1], B[:-1], Y[1:]
            if block_perm_pvalue(a, b, y, lambda u, v, w: surd_synergy(u, v, w, nbins=6)) < ALPHA: pw["SURD"] += 1
            if block_perm_pvalue(a, b, y, interaction_information) < ALPHA: pw["InteractionInfo"] += 1
            # type-I (null / additive)
            Y, A, B, Z = gen("additive", N + 1, rng)
            if ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA)["rejected"]: t1["ANTE"] += 1
            a, b, y = A[:-1], B[:-1], Y[1:]
            if block_perm_pvalue(a, b, y, lambda u, v, w: surd_synergy(u, v, w, nbins=6)) < ALPHA: t1["SURD"] += 1
            if block_perm_pvalue(a, b, y, interaction_information) < ALPHA: t1["InteractionInfo"] += 1
        for m in methods:
            power[m].append(pw[m] / n_rep); type1[m].append(t1[m] / n_rep)
        ante_latency.append(float(np.median(lat)) if lat else None)
        print(f"N={N:5d}  power ANTE={power['ANTE'][-1]:.2f} SURD={power['SURD'][-1]:.2f} "
              f"II={power['InteractionInfo'][-1]:.2f} | ANTE median latency={ante_latency[-1]}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, m in enumerate(methods):
        ax[0].plot(Ns, power[m], "o-", color=PAL[i], lw=2, label=m)
    ax[0].axhline(0.8, color="gray", ls=":", lw=1, label="80% power")
    ax[0].set_xscale("log"); ax[0].set_xlabel("sample size N"); ax[0].set_ylabel("detection power (Y=A·B)")
    ax[0].set_title("(a) Sample efficiency: power vs N"); ax[0].legend(fontsize=8)
    for i, m in enumerate(methods):
        ax[1].plot(Ns, type1[m], "s-", color=PAL[i], lw=2, label=m)
    ax[1].axhline(ALPHA, color="black", ls="--", lw=1.2, label=r"$\alpha$")
    ax[1].set_xscale("log"); ax[1].set_xlabel("sample size N"); ax[1].set_ylabel("type-I (additive null)")
    ax[1].set_title("(b) Type-I at each fixed N (calibrated batch tests)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/bench_sample_efficiency.png"); plt.close()

    res = dict(Ns=list(Ns), power=power, type1=type1, ante_median_latency=ante_latency, n_rep=n_rep)
    json.dump(res, open(f"{OUT}/sample_efficiency.json", "w"), indent=2)
    print("sample-efficiency done.")
    return res


if __name__ == "__main__":
    np.seterr(all="ignore")
    run()
