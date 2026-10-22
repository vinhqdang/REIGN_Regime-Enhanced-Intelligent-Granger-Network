"""R10: power and detection-time sweep over the interaction coefficient (Fig. 2 reconciliation).

DGP: gen_var('syn', T=2500, rho=0.2, strength=c) for c in {0,0.1,0.2,0.3,0.45,0.6,0.75,0.9}.
Tests: ANTE-legacy (synergistic_granger.ante_sg, alpha=0.05) and ANTE-v2 addjoint
(max E >= 20).  500 replications per c, 2 worker processes.
Outputs: results/r10_power_sweep.json, results/r10_power_sweep.png, results/r10_power_sweep_summary.md
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from common import gen_var, wilson, RES  # noqa: E402
from gs_shared import v2_addjoint, legacy_run  # noqa: E402

C_GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
N_REP = 500
T = 2500
RHO = 0.2
ALPHA = 0.05
SEED = 10007
N_BOOT = 1000
PAL = {"legacy": "#4c72b0", "v2": "#c44e52"}


def worker(args):
    ci, rep = args
    c = C_GRID[ci]
    rng = np.random.default_rng([SEED, ci, rep])
    Y, A, B, Z = gen_var("syn", T, rng, rho=RHO, strength=c)
    leg = legacy_run(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
    v2 = v2_addjoint(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
    return ci, rep, leg["rejected"], leg["reject_time"], v2["rejected"], v2["reject_time"]


def boot_median_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(median=np.nan, lo=np.nan, hi=np.nan, n=0)
    bs = [np.median(rng.choice(x, len(x))) for _ in range(n_boot)]
    return dict(median=float(np.median(x)), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n=int(len(x)))


def violations(est, lo, hi, increasing):
    """Adjacent-pair violations of monotonicity in which the CIs do not overlap.
    increasing=True: est should not decrease significantly; False: should not increase."""
    v = []
    for i in range(len(est) - 1):
        if np.isnan(est[i]) or np.isnan(est[i + 1]):
            continue
        if increasing and est[i + 1] < est[i] and hi[i + 1] < lo[i]:
            v.append((i, i + 1))
        if (not increasing) and est[i + 1] > est[i] and lo[i + 1] > hi[i]:
            v.append((i, i + 1))
    return v


def main():
    np.seterr(all="ignore")
    t0 = time.time()
    tasks = [(ci, r) for ci in range(len(C_GRID)) for r in range(N_REP)]
    out = {ci: {"legacy": [], "v2": [], "legacy_t": [], "v2_t": []} for ci in range(len(C_GRID))}
    done = 0
    with Pool(2) as pool:
        for ci, rep, lr, lt, vr, vt in pool.imap_unordered(worker, tasks, chunksize=8):
            out[ci]["legacy"].append(lr); out[ci]["v2"].append(vr)
            if lr: out[ci]["legacy_t"].append(lt)
            if vr: out[ci]["v2_t"].append(vt)
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(tasks)}  t={time.time()-t0:.0f}s", flush=True)

    rng_b = np.random.default_rng(0)
    rows = []
    for ci, c in enumerate(C_GRID):
        r = {"c": c}
        for key in ["legacy", "v2"]:
            flags = np.array(out[ci][key], bool)
            k, n = int(flags.sum()), len(flags)
            lo, hi = wilson(k, n)
            r[key] = dict(power=k / n, k=k, n=n, lo=lo, hi=hi)
            tt = out[ci][key + "_t"]
            r[key]["detect_time"] = boot_median_ci(tt, rng_b)
            r[key]["detect_time_samples"] = int(len(tt))
        rows.append(r)

    # monotonicity checks (power non-decreasing in c; detection time non-increasing, c>0 only)
    mono = {}
    for key in ["legacy", "v2"]:
        pw = np.array([r[key]["power"] for r in rows]); plo = np.array([r[key]["lo"] for r in rows])
        phi = np.array([r[key]["hi"] for r in rows])
        vp = violations(pw, plo, phi, increasing=True)
        strict_p = bool(np.all(np.diff(pw) >= 0))
        dt = np.array([r[key]["detect_time"]["median"] for r in rows[1:]])
        dlo = np.array([r[key]["detect_time"]["lo"] for r in rows[1:]])
        dhi = np.array([r[key]["detect_time"]["hi"] for r in rows[1:]])
        vd = violations(dt, dlo, dhi, increasing=False)
        mono[key] = dict(power_monotone_raw=strict_p,
                         power_monotone_within_CI=len(vp) == 0,
                         power_significant_violations=[[C_GRID[i], C_GRID[j]] for i, j in vp],
                         latency_monotone_raw=bool(np.all(np.diff(dt[~np.isnan(dt)]) <= 0)),
                         latency_monotone_within_CI=len(vd) == 0,
                         latency_significant_violations=[[C_GRID[i + 1], C_GRID[j + 1]] for i, j in vd])

    res = dict(script="r10_power_sweep.py", n_rep=N_REP, T=T, rho=RHO, alpha=ALPHA, seed=SEED,
               c_grid=C_GRID, dgp="gen_var('syn', T, rng, rho=0.2, strength=c)  (0.9 ->c*A_{t-1}B_{t-1}+0.2A_{t-1}+0.3Z+e)",
               settings=dict(legacy="ante_sg(p=1, alpha=.05, forget=.995, ridge=1e-2, cond=[Z]); reject if E>=20 (first time)",
                             v2="ante_v2 addjoint, cap=3, gamma=.9, warmup=50, forget=.995, ridge=1e-2, cond=[Z]; reject if max E>=20",
                             detection_time="time index of first rejection (0-based), median among rejections; bootstrap CI 1000 resamples; latency monotonicity excludes c=0",
                             power_ci="Wilson 95%", seeds=f"default_rng([{SEED}, c_index, rep])"),
               rows=rows, monotonicity=mono, runtime_s=time.time() - t0)
    json.dump(res, open(os.path.join(RES, "r10_power_sweep.json"), "w"), indent=2, default=str)

    # figure
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax2 = ax.twinx()
    x = np.array(C_GRID)
    for key, lab in [("legacy", "ANTE-legacy"), ("v2", "ANTE-v2 addjoint")]:
        pw = np.array([r[key]["power"] for r in rows]); lo = np.array([r[key]["lo"] for r in rows])
        hi = np.array([r[key]["hi"] for r in rows])
        ax.plot(x, pw, "-o", color=PAL[key], label=f"power: {lab}")
        ax.fill_between(x, lo, hi, color=PAL[key], alpha=0.2)
        dm = np.array([r[key]["detect_time"]["median"] for r in rows], float)
        dl = np.array([r[key]["detect_time"]["lo"] for r in rows], float)
        dh = np.array([r[key]["detect_time"]["hi"] for r in rows], float)
        ax2.plot(x, dm, "--s", color=PAL[key], alpha=0.8, ms=4, label=f"median detection time: {lab}")
        ax2.fill_between(x, dl, dh, color=PAL[key], alpha=0.1)
    ax.axhline(ALPHA, color="k", ls=":", lw=1)
    ax.set_xlabel("interaction coefficient c (syn DGP, rho=0.2, T=2500)")
    ax.set_ylabel("power (rejection rate, Wilson 95% CI)")
    ax2.set_ylabel("median detection time (samples), bootstrap CI")
    ax.set_title("Power and detection time vs interaction strength (500 reps)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "r10_power_sweep.png"), dpi=200); plt.close()

    # summary
    L = ["# R10 summary: power sweep (syn DGP, rho=0.2, T=2500, 500 reps)\n",
         "| c | legacy power [CI] | legacy median detect (CI) | v2 power [CI] | v2 median detect (CI) |",
         "|---|---|---|---|---|"]
    for r in rows:
        a, b = r["legacy"], r["v2"]
        ad = a["detect_time"]; bd = b["detect_time"]
        L.append(f"| {r['c']} | {a['power']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | "
                 f"{ad['median']:.0f} ({ad['lo']:.0f}, {ad['hi']:.0f}) | "
                 f"{b['power']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] | "
                 f"{bd['median']:.0f} ({bd['lo']:.0f}, {bd['hi']:.0f}) |")
    L.append("")
    for key in ["legacy", "v2"]:
        m = mono[key]
        L.append(f"- {key}: power monotone (raw) {m['power_monotone_raw']}, within CI {m['power_monotone_within_CI']}; "
                 f"latency non-increasing (raw, c>0) {m['latency_monotone_raw']}, within CI {m['latency_monotone_within_CI']}")
    open(os.path.join(RES, "r10_power_sweep_summary.md"), "w").write("\n".join(L) + "\n")
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
