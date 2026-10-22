"""R2: correlated sources (R2 comment 2.3, R1 point 1).

Additive mechanism ('null') and synergistic mechanism ('syn', strength 0.9) for
rho in {0, .3, .6, .8, .9, .95, .99} (rho set via gen_var; the realised sample
corr(A_t, B_t) is also reported).  n_rep = 150 per cell.  For every cell the v2
addjoint and four tests and the legacy ante_sg test are reported (rejection rate
with Wilson CI, median final log10 E, median max log10 E, mean score).  The
population-like check averages the raw squared-error scores (raw_score) over time
and replications on the additive DGP.

Run from the repository directory:  python3 experiments/revision/r2_correlated_sources.py
"""
import os
import sys
import json
import time

import numpy as np

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from multiprocessing import Pool  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import rate_ci, gen_var, RES  # noqa: E402
from ante_v2 import (run_errors, bounded_score, normalise, bet_process,  # noqa: E402
                     raw_score, MODELS)
from synergistic_granger import ante_sg  # noqa: E402

SCRIPT = "r2_correlated_sources"
ALPHA = 0.05
THR = 1.0 / ALPHA
T = 2500
N_REP = 150
CAP = 3.0
GAMMA = 0.9
WARM = 50
STRENGTH = 0.9
RHOS = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]
DGPS = ["null", "syn"]
METHODS = ("addjoint", "four", "legacy")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else None
    return o


def rep_task(args):
    dgp, rho, seed = args
    rng = np.random.default_rng(seed)
    Y, A, B, Z = gen_var(dgp, T, rng, rho=rho, strength=STRENGTH)
    pred, _ = run_errors(Y, A, B, [Z], p=1, forget=0.995)
    out = dict(corr=float(np.corrcoef(A[1:], B[1:])[0, 1]))
    for kind in ("addjoint", "four"):
        s, b = bounded_score(Y, pred, MODELS, kind, cap=CAP)
        u = normalise(s, b)
        E, _ = bet_process(u, GAMMA, WARM, start=1)
        out[kind] = dict(final=float(E[-1]), maxE=float(E.max()), mean_s=float(np.mean(s[1:])))
        raw = raw_score(Y, pred, MODELS, kind)
        out[kind]["raw_mean"] = float(np.mean(raw[1:]))
    leg = ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA, forget=0.995)
    out["legacy"] = dict(final=float(leg["final_E"]), maxE=float(np.max(leg["E"])),
                         mean_s=float(leg["mean_score"]))
    # legacy scores are already raw squared-error scores (unclipped)
    out["legacy"]["raw_mean"] = float(leg["mean_score"])
    return out


def summarise(rows):
    res = {"realised_corr_mean": float(np.mean([r["corr"] for r in rows]))}
    for m in METHODS:
        fin = np.array([r[m]["final"] for r in rows], float)
        mx = np.array([r[m]["maxE"] for r in rows], float)
        ms = np.array([r[m]["mean_s"] for r in rows], float)
        rw = np.array([r[m]["raw_mean"] for r in rows], float)
        d = rate_ci(mx >= THR)
        d.update(median_log10_final=float(np.median(np.log10(fin))),
                 median_log10_max=float(np.median(np.log10(mx))),
                 mean_score=float(np.mean(ms)),
                 mean_raw_score=float(np.mean(rw)),
                 mean_raw_score_se=float(np.std(rw, ddof=1) / np.sqrt(len(rw))))
        res[m] = d
    return res


def main():
    t0 = time.time()
    tasks = []
    keys = []
    for di, dgp in enumerate(DGPS):
        for ri, rho in enumerate(RHOS):
            keys.append((dgp, rho))
            for r in range(N_REP):
                tasks.append((dgp, rho, 200000 * (di + 1) + 1000 * (ri + 1) + r))
    print(f"R2: {len(keys)} cells x {N_REP} reps = {len(tasks)} tasks, 2 workers", flush=True)
    buckets = {k: [] for k in keys}
    done = 0
    with Pool(2) as pool:
        for (dgp, rho, _), out in zip(tasks, pool.imap(rep_task, tasks, chunksize=4)):
            buckets[(dgp, rho)].append(out)
            done += 1
            if done % 150 == 0 or done == len(tasks):
                print(f"  progress {done}/{len(tasks)}  elapsed {time.time() - t0:.0f}s", flush=True)

    cells = []
    for dgp, rho in keys:
        agg = summarise(buckets[(dgp, rho)])
        cells.append(dict(dgp=dgp, rho=rho, strength=STRENGTH if dgp == "syn" else None,
                          n_rep=N_REP, methods=agg))
        print(f"  {dgp} rho={rho}: corr={agg['realised_corr_mean']:.3f} " + " | ".join(
            f"{m} {agg[m]['rate']:.3f} [{agg[m]['lo']:.3f},{agg[m]['hi']:.3f}]" for m in METHODS),
            flush=True)

    runtime = time.time() - t0
    out = dict(script=SCRIPT, n_rep=N_REP, T=T, alpha=ALPHA, rejection_rule="max_E >= 20",
               settings=dict(gamma=GAMMA, warmup=WARM, cap=CAP, forget=0.995, p=1, cond="[Z]",
                             strength_syn=STRENGTH, rhos=RHOS, feat_clip=None,
                             seeds="200000*(dgp index+1)+1000*(rho index+1)+rep",
                             legacy="synergistic_granger.ante_sg defaults"),
               cells=cells, runtime_sec=runtime)
    with open(os.path.join(RES, f"{SCRIPT}.json"), "w") as fh:
        json.dump(clean(out), fh, indent=2)

    L = ["# R2 correlated sources (auto-generated)", "",
         f"n_rep = {N_REP} per cell, T = {T}, alpha = {ALPHA}, rejection = max E >= 20, "
         f"syn strength = {STRENGTH}.", "",
         "## Rejection, log10 E and score per cell", "",
         "| DGP | rho | realised corr(A,B) | method | rejection | 95% Wilson CI | median log10 final E | "
         "median log10 max E | mean score (bounded for v2) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for m in METHODS:
            d = c["methods"][m]
            L.append(f"| {c['dgp']} | {c['rho']} | {c['methods']['realised_corr_mean']:.3f} | {m} | "
                     f"{d['rate']:.3f} ({d['k']}/{d['n']}) | [{d['lo']:.3f}, {d['hi']:.3f}] | "
                     f"{d['median_log10_final']:.3f} | {d['median_log10_max']:.3f} | {d['mean_score']:.4f} |")
    L += ["", "## Population-like check: mean raw squared-error score (raw_score), additive DGP", "",
          "| rho | addjoint mean raw s (+-SE) | four mean raw s (+-SE) |", "|---|---|---|"]
    for c in cells:
        if c["dgp"] != "null":
            continue
        a, f = c["methods"]["addjoint"], c["methods"]["four"]
        L.append(f"| {c['rho']} | {a['mean_raw_score']:.4f} (+-{a['mean_raw_score_se']:.4f}) | "
                 f"{f['mean_raw_score']:.4f} (+-{f['mean_raw_score_se']:.4f}) |")
    L += ["", "## Population-like check on the synergy DGP (raw squared-error scores)", "",
          "| rho | addjoint mean raw s | four mean raw s |", "|---|---|---|"]
    for c in cells:
        if c["dgp"] != "syn":
            continue
        L.append(f"| {c['rho']} | {c['methods']['addjoint']['mean_raw_score']:.4f} | "
                 f"{c['methods']['four']['mean_raw_score']:.4f} |")
    L += ["", f"Runtime: {runtime / 60:.1f} min. Figure: `{os.path.join(RES, SCRIPT + '.png')}`", ""]
    with open(os.path.join(RES, f"{SCRIPT}_summary.md"), "w") as fh:
        fh.write("\n".join(L))

    # ---------------------------------------------------------- figure
    colors = {"addjoint": "#1f77b4", "four": "#ff7f0e", "legacy": "#7f7f7f"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    rhos = np.array(RHOS)
    for ax, dgp, title in ((axes[0], "null", "Additive: false rejection vs rho"),
                           (axes[1], "syn", "Synergy (strength 0.9): power vs rho")):
        cs = [c for c in cells if c["dgp"] == dgp]
        for m in METHODS:
            r = np.array([c["methods"][m]["rate"] for c in cs])
            lo = np.array([c["methods"][m]["lo"] for c in cs])
            hi = np.array([c["methods"][m]["hi"] for c in cs])
            ax.plot(rhos, r, "o-", color=colors[m], label=m)
            ax.fill_between(rhos, lo, hi, color=colors[m], alpha=0.2)
        if dgp == "null":
            ax.axhline(ALPHA, color="red", ls="--", lw=1)
        ax.set_xlabel("rho (source correlation)")
        ax.set_ylabel("rejection rate (max E >= 20)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
    ax = axes[2]
    for dgp, ls in (("null", "-"), ("syn", "--")):
        cs = [c for c in cells if c["dgp"] == dgp]
        for m in ("addjoint", "four"):
            y = np.array([c["methods"][m]["mean_raw_score"] for c in cs])
            se = np.array([c["methods"][m]["mean_raw_score_se"] for c in cs])
            ax.errorbar(rhos, y, yerr=1.96 * se, fmt="o" + ls, color=colors[m], capsize=2,
                        label=f"{m} ({dgp})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("rho (source correlation)")
    ax.set_ylabel("mean raw score (squared-error gap)")
    ax.set_title("Mean score vs rho (95% CI bars)")
    ax.legend(fontsize=8)
    fig.suptitle(f"R2: correlated sources (n_rep={N_REP} per cell, Wilson / CI bands)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"{SCRIPT}.png"), dpi=200)
    print(f"Done in {runtime / 60:.1f} min. JSON: {os.path.join(RES, SCRIPT + '.json')}", flush=True)


if __name__ == "__main__":
    main()
