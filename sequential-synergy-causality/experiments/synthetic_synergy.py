"""
Synthetic validation of ANTE-SG (synergistic Granger causality).

S1  Type-I / discrimination: additive and own-nonlinear DGPs are NOT flagged;
    genuine cross-interaction IS.  (rejection rates)
S2  Anytime-valid vs continuously-monitored fixed-sample test (type-I inflation).
S3  Power and median detection time vs synergy strength.
S4  e-process trajectories: null vs synergistic.

Saves JSON + PNG to experiments/results/.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synergistic_granger import ante_sg, peeking_synergy_test

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05


def gen(kind, T, rng, strength=0.9, ar=0.3):
    """Common-factor VAR variants. Sources A,B have own AR dynamics; Y depends
    on their lags additively / own-nonlinearly / synergistically (+ market Z)."""
    Z = rng.standard_normal(T)                      # common market factor
    A = np.zeros(T); B = np.zeros(T); Y = np.zeros(T)
    for t in range(1, T):
        A[t] = ar * A[t - 1] + 0.5 * Z[t] + rng.standard_normal()
        B[t] = ar * B[t - 1] + 0.5 * Z[t] + rng.standard_normal()
    for t in range(1, T):
        common = 0.3 * Z[t]
        if kind == "additive":
            Y[t] = 0.4 * A[t - 1] + 0.4 * B[t - 1] + common + rng.standard_normal()
        elif kind == "ownnonlin":
            Y[t] = 0.5 * A[t - 1] ** 2 + 0.4 * B[t - 1] + common + rng.standard_normal()
        elif kind == "synergistic":
            Y[t] = strength * A[t - 1] * B[t - 1] + 0.2 * A[t - 1] + common + rng.standard_normal()
        else:
            raise ValueError(kind)
    return dict(Y=Y, A=A, B=B, Z=Z)


def s1_discrimination(n_rep=300, T=2500, seed=0):
    rng = np.random.default_rng(seed)
    kinds = ["additive", "ownnonlin", "synergistic"]
    rej = {k: 0 for k in kinds}
    for _ in range(n_rep):
        for k in kinds:
            d = gen(k, T, rng)
            r = ante_sg(d["Y"], d["A"], d["B"], cond=[d["Z"]], p=1, alpha=ALPHA)
            rej[k] += r["rejected"]
    res = {k: rej[k] / n_rep for k in kinds}
    print("S1 rejection rates:", res)
    return dict(rejection_rate=res, n_rep=n_rep, T=T, alpha=ALPHA)


def s2_anytime_vs_peeking(n_rep=400, T=2500, seed=1):
    rng = np.random.default_rng(seed)
    e_false = w_false = 0
    for _ in range(n_rep):
        d = gen("additive", T, rng)                 # null for synergy
        e = ante_sg(d["Y"], d["A"], d["B"], cond=[d["Z"]], p=1, alpha=ALPHA)
        w = peeking_synergy_test(d["Y"], d["A"], d["B"], cond=[d["Z"]], p=1, alpha=ALPHA)
        e_false += e["rejected"]; w_false += w["rejected"]
    res = dict(eprocess_type1=e_false / n_rep, peeking_type1=w_false / n_rep,
               n_rep=n_rep, T=T, alpha=ALPHA)
    print("S2:", res)
    return res


def s3_power_latency(n_rep=250, T=2500, seed=2):
    rng = np.random.default_rng(seed)
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    power, det = [], []
    for s in strengths:
        rej, times = 0, []
        for _ in range(n_rep):
            d = gen("synergistic", T, rng, strength=s)
            r = ante_sg(d["Y"], d["A"], d["B"], cond=[d["Z"]], p=1, alpha=ALPHA)
            if r["rejected"]:
                rej += 1; times.append(r["reject_time"])
        power.append(rej / n_rep)
        det.append(float(np.median(times)) if times else None)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(strengths, power, "o-", color=PAL[2], lw=2)
    ax[0].axhline(ALPHA, color="k", ls=":", lw=1.2, label=r"$\alpha$")
    ax[0].set_xlabel("synergy strength (A·B coefficient)")
    ax[0].set_ylabel(f"rejection rate within T={T}")
    ax[0].set_title("(a) power vs synergy strength"); ax[0].legend()
    xs = [s for s, d_ in zip(strengths, det) if d_]
    ys = [d_ for d_ in det if d_]
    ax[1].plot(xs, ys, "s-", color=PAL[0], lw=2)
    ax[1].set_xlabel("synergy strength (A·B coefficient)")
    ax[1].set_ylabel("median detection time (steps)")
    ax[1].set_title("(b) detection latency")
    plt.tight_layout(); plt.savefig(f"{OUT}/sg_s3_power_latency.png"); plt.close()
    res = dict(strength=strengths, power=power, median_detection=det, n_rep=n_rep, T=T)
    print("S3 power:", [round(p, 3) for p in power])
    return res


def s4_paths(T=2500, seed=3, n_show=12):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, col in [("additive", PAL[0]), ("ownnonlin", PAL[4]), ("synergistic", PAL[3])]:
        for k in range(n_show):
            d = gen(kind, T, rng, strength=0.7)
            r = ante_sg(d["Y"], d["A"], d["B"], cond=[d["Z"]], p=1, alpha=ALPHA)
            ax.plot(np.log10(np.maximum(r["E"], 1e-3)), color=col, alpha=0.35, lw=1.0,
                    label=kind if k == 0 else None)
    ax.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.6,
               label=r"reject threshold $\log_{10}(1/\alpha)$")
    ax.set_xlabel("trading steps seen $t$"); ax.set_ylabel(r"$\log_{10} E_t$")
    ax.set_title("ANTE-SG: capital grows only under genuine synergy")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/sg_s4_paths.png"); plt.close()
    print("S4 saved")


if __name__ == "__main__":
    np.seterr(all="ignore")
    res = {}
    res["S1_discrimination"] = s1_discrimination()
    res["S2_anytime_vs_peeking"] = s2_anytime_vs_peeking()
    res["S3_power_latency"] = s3_power_latency()
    s4_paths()
    with open(f"{OUT}/synthetic_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSynthetic experiments complete.")
