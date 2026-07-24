"""
(1) Validate the re-implemented 2026 baselines against their papers' own numbers.
(2) Show the only other sequential/anytime-valid method (Jha 2026 DGC) is
    single-source and misses GROUP synergy that ANTE detects.

Saves validation_dgc.json + bench_dgc_vs_ante.png.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from baselines import peid_synergy, peid_synergy_continuous
from jha_dgc import dgc_test
import synergy_eprocess as sep_int
OUT = os.path.join(HERE, "results")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def validate_peid(seed=0):
    rng = np.random.default_rng(seed); n = 60000
    A = rng.integers(0, 2, n); B = rng.integers(0, 2, n)
    xor = peid_synergy(A, B, A ^ B)
    AND = peid_synergy(A, B, A & B)
    # continuous trend on the paper's own system: X1 = a*sin(X2*X3)+(1-a)X2+noise
    trend = {}
    for a in [0.0, 0.5, 1.0]:
        X2 = rng.uniform(-1, 1, 6000); X3 = rng.uniform(-1, 1, 6000)
        Y = a*np.sin(X2*X3) + (1-a)*X2 + 0.05*rng.standard_normal(6000)
        trend[a] = peid_synergy_continuous(X2, X3, Y)
    out = dict(discrete_XOR=xor, discrete_AND=AND, paper_AND_target=0.189,
               continuous_trend={f"alpha={k}": v for k, v in trend.items()})
    print("PEID validation: XOR=%.3f (paper 1.0), AND=%.3f (paper 0.189); "
          "continuous alpha 0->1: %s" % (xor, AND, {k: round(v, 3) for k, v in trend.items()}))
    return out


def validate_dgc(seed=1):
    rng = np.random.default_rng(seed)
    def s1(T, s):
        X = np.zeros(T); Y = np.zeros(T)
        for t in range(1, T):
            X[t] = 0.5*X[t-1] + rng.standard_normal()
            Y[t] = 0.3*Y[t-1] + s*X[t-1] + rng.standard_normal()
        return Y, X
    res = {}
    for s, key in [(0.0, "size_s0"), (0.3, "power_s0.3")]:
        c = 0
        for _ in range(200):
            Y, X = s1(500, s)
            c += dgc_test(Y, X, B=150)["rejected"]
        res[key] = c / 200
    print(f"DGC validation (S1, T=500): size={res['size_s0']:.3f} (paper ~0.05), "
          f"power={res['power_s0.3']:.3f} (paper ~1.0)")
    return res


def dgc_vs_ante_synergy(seed=2, R=150, T=3000):
    rng = np.random.default_rng(seed)
    dgcA = dgcB = ante = 0
    for _ in range(R):
        A = rng.integers(0, 2, T); B = rng.integers(0, 2, T)
        Y = np.zeros(T, int); Y[1:] = A[:-1] ^ B[:-1]
        dgcA += dgc_test(Y.astype(float), A.astype(float), B=120)["rejected"]
        dgcB += dgc_test(Y.astype(float), B.astype(float), B=120)["rejected"]
        ante += sep_int.run_eprocess(A[:-1], B[:-1], Y[1:], alpha=0.05)["rejected"]
    res = dict(dgc_A=dgcA/R, dgc_B=dgcB/R, ante_group=ante/R, R=R)
    print(f"Group synergy (Y=A XOR B): DGC(A)={res['dgc_A']:.2f} DGC(B)={res['dgc_B']:.2f} "
          f"ANTE(group)={res['ante_group']:.2f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labs = ["Jha DGC\n(single-source A)", "Jha DGC\n(single-source B)", "ANTE\n(group {A,B})"]
    vals = [res["dgc_A"], res["dgc_B"], res["ante_group"]]
    cols = [PAL[3], PAL[3], PAL[2]]
    ax.bar(labs, vals, color=cols, edgecolor="black", lw=0.6)
    ax.axhline(0.05, color="k", ls="--", lw=1.2, label=r"$\alpha=0.05$")
    ax.set_ylabel("detection rate (Y = A XOR B, pure group synergy)")
    ax.set_title("Only ANTE detects group synergy;\nthe sequential SOTA (Jha 2026) is single-source and misses it")
    for i, v in enumerate(vals): ax.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/bench_dgc_vs_ante.png"); plt.close()
    return res


if __name__ == "__main__":
    np.seterr(all="ignore")
    out = dict(peid_validation=validate_peid(), dgc_validation=validate_dgc(),
               dgc_vs_ante_group_synergy=dgc_vs_ante_synergy())
    json.dump(out, open(f"{OUT}/validation_dgc.json", "w"), indent=2, default=str)
    print("done.")
