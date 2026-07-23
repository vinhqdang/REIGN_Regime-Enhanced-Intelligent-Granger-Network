"""
Experiments for sequential synergistic-causality testing.

E1  Anytime-valid type-I control vs. repeated-peeking Wald test (null data).
E2  Sample-paths of the e-process under null / AND / XOR mechanisms.
E3  Power & detection-time vs. synergy strength.
E4  Subset refinement: localise a synergistic pair among three candidate causes.

All results saved to experiments/results/ (JSON + PNG, 300 dpi).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_generators as dg
import synergy_eprocess as se

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05
N = 4000


# ---------------------------------------------------------------------------
def e1_type_one_error(n_rep=2000, n=N, seed=0):
    """False-positive rate under H0 for (a) e-process, (b) peeking Wald test."""
    rng = np.random.default_rng(seed)
    e_false, wald_false = 0, 0
    e_traj_max, wald_first = [], []
    for _ in range(n_rep):
        d = dg.gen_null(n, rng)
        er = se.run_eprocess(d["A"], d["B"], d["C"], alpha=ALPHA)
        wr = se.wald_peeking(d["A"], d["B"], d["C"], alpha=ALPHA)
        e_false += er["rejected"]
        wald_false += wr["rejected"]
        e_traj_max.append(float(np.max(er["E"])))
        if wr["rejected"]:
            wald_first.append(wr["reject_time"])
    res = dict(
        n_rep=n_rep, n=n, alpha=ALPHA,
        eprocess_type1=e_false / n_rep,
        wald_peeking_type1=wald_false / n_rep,
        eprocess_max_E_median=float(np.median(e_traj_max)),
    )
    # figure: distribution of max E vs 1/alpha, and cumulative false-positive curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(np.clip(e_traj_max, 0, 40), bins=40, color=PAL[0], alpha=0.8)
    ax[0].axvline(1 / ALPHA, color="red", ls="--", lw=1.6, label=r"reject threshold $1/\alpha=20$")
    ax[0].set_xlabel(r"$\max_t E_t$ over the stream (null data)")
    ax[0].set_ylabel("frequency")
    ax[0].set_title("(a) e-process never systematically crosses under $H_0$")
    ax[0].legend()

    # cumulative type-I as a function of monitoring horizon
    rng2 = np.random.default_rng(seed + 1)
    horizons = np.arange(50, n + 1, 50)
    e_cum = np.zeros(len(horizons)); w_cum = np.zeros(len(horizons))
    reps2 = 400
    for _ in range(reps2):
        d = dg.gen_null(n, rng2)
        er = se.run_eprocess(d["A"], d["B"], d["C"], alpha=ALPHA)
        wr = se.wald_peeking(d["A"], d["B"], d["C"], alpha=ALPHA)
        eE = er["E"]; wz = wr["z"]; zc = wr["zcrit"]
        for i, h in enumerate(horizons):
            if np.any(eE[:h] >= 1 / ALPHA): e_cum[i] += 1
            if np.any(wz[:h] >= zc): w_cum[i] += 1
    e_cum /= reps2; w_cum /= reps2
    ax[1].plot(horizons, w_cum, color=PAL[3], lw=2, label="fixed-$n$ Wald, monitored continuously")
    ax[1].plot(horizons, e_cum, color=PAL[2], lw=2, label="e-process (ours)")
    ax[1].axhline(ALPHA, color="black", ls=":", lw=1.4, label=r"nominal $\alpha=0.05$")
    ax[1].set_xlabel("monitoring horizon (samples seen)")
    ax[1].set_ylabel("cumulative false-positive rate")
    ax[1].set_title("(b) continuous monitoring inflates the classical test")
    ax[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/e1_type_one_error.png"); plt.close()
    res["wald_peeking_type1_at_horizon"] = dict(zip([int(h) for h in horizons], list(w_cum)))
    res["eprocess_type1_at_horizon"] = dict(zip([int(h) for h in horizons], list(e_cum)))
    print("E1 done:", {k: v for k, v in res.items() if not isinstance(v, dict)})
    return res


# ---------------------------------------------------------------------------
def e2_sample_paths(n=N, seed=3, n_show=12):
    """Overlay e-process trajectories for null / AND / XOR mechanisms."""
    rng = np.random.default_rng(seed)
    specs = [("null", dg.gen_null, PAL[0]),
             ("AND synergy", dg.gen_and, PAL[2]),
             ("XOR (pure synergy)", dg.gen_xor, PAL[3])]
    fig, ax = plt.subplots(figsize=(9, 5))
    summary = {}
    for name, gen, col in specs:
        rts = []
        for k in range(n_show):
            d = gen(n, rng)
            er = se.run_eprocess(d["A"], d["B"], d["C"], alpha=ALPHA)
            ax.plot(np.arange(1, n + 1), np.log10(np.maximum(er["E"], 1e-3)),
                    color=col, alpha=0.35, lw=1.0,
                    label=name if k == 0 else None)
            if er["rejected"]:
                rts.append(er["reject_time"])
        summary[name] = dict(n_rejected=len(rts), n_paths=n_show,
                             median_detection=float(np.median(rts)) if rts else None)
    ax.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.6,
               label=r"reject threshold $\log_{10}(1/\alpha)$")
    ax.set_xlabel("samples seen $t$")
    ax.set_ylabel(r"$\log_{10} E_t$")
    ax.set_title("e-process trajectories: capital grows only under genuine synergy")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/e2_sample_paths.png"); plt.close()
    print("E2 done:", summary)
    return summary


# ---------------------------------------------------------------------------
def e3_power_detection(n=N, seed=7, n_rep=300):
    """Rejection rate and median detection time vs. AND-synergy strength."""
    rng = np.random.default_rng(seed)
    syns = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    power, det = [], []
    for s in syns:
        rej, times = 0, []
        for _ in range(n_rep):
            d = dg.gen_and(n, rng, base=0.15, ea=0.15, eb=0.15, syn=s)
            er = se.run_eprocess(d["A"], d["B"], d["C"], alpha=ALPHA)
            if er["rejected"]:
                rej += 1; times.append(er["reject_time"])
        power.append(rej / n_rep)
        det.append(float(np.median(times)) if times else None)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(syns, power, "o-", color=PAL[2], lw=2)
    ax[0].axhline(ALPHA, color="black", ls=":", lw=1.2, label=r"$\alpha$ (at syn=0)")
    ax[0].set_xlabel("synergy strength (AND coefficient)")
    ax[0].set_ylabel(f"rejection rate within {n} samples")
    ax[0].set_title("(a) power increases with synergy strength")
    ax[0].legend()
    xs = [s for s, d_ in zip(syns, det) if d_ is not None]
    ys = [d_ for d_ in det if d_ is not None]
    ax[1].plot(xs, ys, "s-", color=PAL[0], lw=2)
    ax[1].set_xlabel("synergy strength (AND coefficient)")
    ax[1].set_ylabel("median detection time (samples)")
    ax[1].set_title("(b) stronger synergy is detected sooner")
    plt.tight_layout(); plt.savefig(f"{OUT}/e3_power_detection.png"); plt.close()
    res = dict(synergy=syns, power=power, median_detection=det, n_rep=n_rep, n=n)
    print("E3 done: power=", [round(p, 3) for p in power])
    return res


# ---------------------------------------------------------------------------
def e4_subset_refinement(n=N, seed=11, n_rep=200):
    """Among {A,B,D}, only (A,B) is synergistic: does the search localise it?"""
    rng = np.random.default_rng(seed)
    fire = {("A", "B"): 0, ("A", "D"): 0, ("B", "D"): 0}
    example = None
    for r in range(n_rep):
        d = dg.gen_triple(n, rng, syn=0.4, synergistic_pair=("A", "B"))
        out = se.subset_refinement(d, ["A", "B", "D"], alpha=ALPHA)
        for pair, info in out["pairs"].items():
            if info["rejected"]:
                fire[pair] += 1
        if r == 0:
            example = out
    fire_rate = {f"{u}-{v}": fire[(u, v)] / n_rep for (u, v) in fire}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = list(fire_rate.keys()); vals = list(fire_rate.values())
    cols = [PAL[2] if lab == "A-B" else PAL[3] for lab in labels]
    ax.bar(labels, vals, color=cols, edgecolor="black", lw=0.6)
    ax.axhline(ALPHA, color="black", ls=":", lw=1.3, label=r"$\alpha$ (false-positive target)")
    ax.set_ylabel(f"detection rate within {n} samples")
    ax.set_xlabel("candidate cause-pair")
    ax.set_title("(a) refinement fires on the true synergistic pair (A-B) only")
    ax.legend()
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/e4_subset_refinement.png"); plt.close()
    res = dict(detection_rate=fire_rate, n_rep=n_rep, n=n,
               alpha_each=example["alpha_each"], n_pairs=example["n_pairs"])
    print("E4 done:", fire_rate)
    return res


if __name__ == "__main__":
    np.seterr(all="ignore")
    all_res = {}
    all_res["E1_type_one_error"] = e1_type_one_error()
    all_res["E2_sample_paths"] = e2_sample_paths()
    all_res["E3_power_detection"] = e3_power_detection()
    all_res["E4_subset_refinement"] = e4_subset_refinement()
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print("\nAll experiments complete. Results in", OUT)
