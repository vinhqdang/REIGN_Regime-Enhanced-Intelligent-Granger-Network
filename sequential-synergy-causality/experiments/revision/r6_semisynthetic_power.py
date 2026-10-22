#!/usr/bin/env python3
"""R6: semi-synthetic power on real financial noise (v2 addjoint test).

Real triples (C; A,B) are drawn (seed 0) from two R5 settings: stocks13 target 'ret' (lag1,
conditioned on SPY) and etf16 target 'vol' (lag1, conditioned on |SPY|).  The target is replaced by

    Y'_t = z(C)_t + beta * w(A)_{t-1} * w(B)_{t-1},   w = clip(z, -5, 5)  (feature winsorisation
                                                                          identical to the test)

and Y' is re-standardised predictably before the v2 addjoint test is run on the REAL A, B, SPY
series (lag1, forget=0.995, feat_clip=5).  beta = 0 gives the Type-I rate on real noise.
The injected term is a known LAGGED interaction at the population level.

The population variance share is var(beta*w_A*w_B)/var(Y') computed on the realised series
(Y' before the final re-standardisation, rows t >= 30).

Run from the repository directory:  python3 experiments/revision/r6_semisynthetic_power.py
Outputs in experiments/revision/results/: r6_semisynthetic_power.json, r6_semisynthetic_power.png,
r6_semisynthetic_power_summary.md
"""
import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from common import wilson, RES  # noqa: E402
from ante_v2 import (run_errors, bounded_score, normalise, bet_process,  # noqa: E402
                     predictable_standardise, MODELS)
from r5_master_scan import build_matrix, hypotheses  # identical construction to R5  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

np.seterr(all="ignore")

BETAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
N_TRIPLES = 40
SEED = 0
FEAT_CLIP = 5.0
FORGET = 0.995
RIDGE = 1e-2
CAP = 3.0
GAMMA = 0.9
WARM = 50
ALPHA = 0.05
STD_WARM = 30
M_FAMILY = [660, 4400]
DATASETS = [("stocks13", "ret", "stocks13 ret lag1 (cond SPY)"),
            ("etf16", "vol", "etf16 vol lag1 (cond |SPY|)")]
THRESH = {"per_hyp_20": 1.0 / ALPHA,
          "fw_m660": 660 / ALPHA,
          "fw_m4400": 4400 / ALPHA}


def _eval(args):
    beta, zC, zA, zB, zS = args
    T = len(zC)
    fa = np.clip(zA, -FEAT_CLIP, FEAT_CLIP)
    fb = np.clip(zB, -FEAT_CLIP, FEAT_CLIP)
    term = np.zeros(T)
    term[1:] = beta * fa[:-1] * fb[:-1]
    Yp = zC + term
    if beta > 0:
        share = float(np.var(term[STD_WARM:]) / np.var(Yp[STD_WARM:]))
    else:
        share = 0.0
    Y = predictable_standardise(Yp, warmup=STD_WARM)
    pred, _ = run_errors(Y, zA, zB, cond=[zS], p=1, forget=FORGET, ridge=RIDGE,
                         feat_clip=FEAT_CLIP)
    s, b = bounded_score(Y, pred, MODELS, "addjoint", cap=CAP)
    u = normalise(s, b)
    E, _ = bet_process(u, gamma=GAMMA, warmup=WARM, start=1)
    mx = float(E.max())
    return dict(beta=beta, share=share, final_log10E=float(np.log10(E[-1])),
                max_log10E=float(np.log10(mx)),
                rej_per_hyp_20=bool(mx >= THRESH["per_hyp_20"]),
                rej_fw_m660=bool(mx >= THRESH["fw_m660"]),
                rej_fw_m4400=bool(mx >= THRESH["fw_m4400"]))


def _power_rows(recs, betas):
    rows = []
    for beta in betas:
        rr = [r for r in recs if r["beta"] == beta]
        n = len(rr)
        row = dict(beta=beta, n=n,
                   mean_share=float(np.mean([r["share"] for r in rr])),
                   min_share=float(np.min([r["share"] for r in rr])),
                   max_share=float(np.max([r["share"] for r in rr])),
                   median_max_log10E=float(np.median([r["max_log10E"] for r in rr])))
        for key, label in [("rej_per_hyp_20", "power_per_hyp_20"),
                           ("rej_fw_m660", "power_fw_m660"),
                           ("rej_fw_m4400", "power_fw_m4400")]:
            k = int(sum(r[key] for r in rr))
            lo, hi = wilson(k, n)
            row[label] = dict(k=k, rate=k / n, lo=lo, hi=hi)
        rows.append(row)
    return rows


def _min_detectable(rows, key, level=0.8):
    for r in rows:
        if r["beta"] > 0 and r[key]["rate"] >= level:
            return dict(beta=r["beta"], mean_share=r["mean_share"], power=r[key]["rate"],
                        ci=[r[key]["lo"], r[key]["hi"]])
    return None


def main():
    t0 = time.time()
    results = {"settings": dict(betas=BETAS, n_triples=N_TRIPLES, seed=SEED, feat_clip=FEAT_CLIP,
                                forget=FORGET, ridge=RIDGE, cap=CAP, gamma=GAMMA, warmup=WARM,
                                alpha=ALPHA, std_warmup=STD_WARM, test="v2 addjoint",
                                thresholds={"per_hyp_20": THRESH["per_hyp_20"],
                                            "fw_m660": THRESH["fw_m660"],
                                            "fw_m4400": THRESH["fw_m4400"]},
                                injected="beta*w(A)_{t-1}*w(B)_{t-1}, w=clip(z,-5,5)",
                                share_definition="var(beta*w_A*w_B)/var(Y') on realised series, rows t>=30, "
                                                 "Y' before final predictable re-standardisation"),
               "datasets": {}}
    with Pool(2) as pool:
        for panel, target, label in DATASETS:
            Z, cols, cond, _ = build_matrix(panel, target, "lag1")
            H = hypotheses(cols, cond)
            rng = np.random.default_rng(SEED)
            sel = np.sort(rng.choice(len(H), N_TRIPLES, replace=False))
            zS = np.ascontiguousarray(Z[:, cols.index(cond)])
            tasks, meta = [], []
            for i in sel:
                C, A, B = H[i]
                zC = np.ascontiguousarray(Z[:, cols.index(C)])
                zA = np.ascontiguousarray(Z[:, cols.index(A)])
                zB = np.ascontiguousarray(Z[:, cols.index(B)])
                for beta in BETAS:
                    tasks.append((beta, zC, zA, zB, zS))
                    meta.append(dict(triple_index=int(i), C=C, A=A, B=B))
            print(f"{panel} {target}: T={Z.shape[0]} triples={len(sel)} runs={len(tasks)}", flush=True)
            out = pool.map(_eval, tasks, chunksize=4)
            recs = []
            for m, o in zip(meta, out):
                recs.append({**m, **o})
            rows = _power_rows(recs, BETAS)
            mins = {k: _min_detectable(rows, k) for k in
                    ["power_per_hyp_20", "power_fw_m660", "power_fw_m4400"]}
            results["datasets"][f"{panel}_{target}"] = dict(
                label=label, T=int(Z.shape[0]), conditioner=cond,
                triples=[dict(C=r["C"], A=r["A"], B=r["B"], triple_index=r["triple_index"])
                         for r in recs if r["beta"] == 0.0],
                table=rows, minimum_detectable_power_0_8=mins, per_run=recs)
            print(f"done {panel} {target} in {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(RES, "r6_semisynthetic_power.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    styles = [("power_per_hyp_20", "max E >= 20 (per hypothesis)", "C0", "o"),
              ("power_fw_m660", "max E >= 660/alpha (m=660)", "C1", "s"),
              ("power_fw_m4400", "max E >= 4400/alpha (m=4400)", "C2", "^")]
    for ax, (panel, target, label) in zip(axes, DATASETS):
        ds = results["datasets"][f"{panel}_{target}"]
        rows = ds["table"]
        x = np.array([r["mean_share"] for r in rows])
        for key, name, col, mk in styles:
            y = np.array([r[key]["rate"] for r in rows])
            lo = np.array([r[key]["lo"] for r in rows])
            hi = np.array([r[key]["hi"] for r in rows])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk + "-", color=col, capsize=3,
                        label=name, ms=5)
        ax.axhline(0.8, color="grey", ls=":", lw=1)
        ax.axhline(ALPHA, color="grey", ls="--", lw=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("population variance share of injected term (mean over 40 triples)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        for r in rows:
            ax.annotate(f"b={r['beta']:g}", (r["mean_share"], r["power_per_hyp_20"]["rate"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=6)
    axes[0].set_ylabel("power (Wilson 95% CI over 40 triples)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "r6_semisynthetic_power.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------ markdown
    L = ["# R6 semi-synthetic power summary\n",
         f"Triples: {N_TRIPLES} per dataset (seed {SEED}); v2 addjoint; beta = 0 is Type-I on real noise. "
         "Thresholds: max E >= 20 (per hypothesis), >= 660/0.05 = 13200 and >= 4400/0.05 = 88000 (family-wise).\n"]
    for panel, target, label in DATASETS:
        ds = results["datasets"][f"{panel}_{target}"]
        L.append(f"## {label}\n")
        L.append("| beta | mean var share | power max E>=20 [95% CI] | power >=13200 [CI] | power >=88000 [CI] | median max log10E |")
        L.append("|---|---|---|---|---|---|")
        for r in ds["table"]:
            def f(d):
                return f"{d['rate']:.3f} [{d['lo']:.2f}, {d['hi']:.2f}]"
            L.append(f"| {r['beta']:g} | {r['mean_share']:.4f} (range {r['min_share']:.4f}-{r['max_share']:.4f}) | "
                     f"{f(r['power_per_hyp_20'])} | {f(r['power_fw_m660'])} | {f(r['power_fw_m4400'])} | "
                     f"{r['median_max_log10E']:.3f} |")
        L.append("")
        md = ds["minimum_detectable_power_0_8"]
        L.append("Minimum detectable (first grid value with point-estimate power >= 0.8):\n")
        for key, name in [("power_per_hyp_20", "max E >= 20"), ("power_fw_m660", "m=660 family"),
                          ("power_fw_m4400", "m=4400 family")]:
            v = md[key]
            if v is None:
                L.append(f"- {name}: none on the grid")
            else:
                L.append(f"- {name}: beta = {v['beta']:g}, mean variance share = {v['mean_share']:.4f} "
                         f"(power {v['power']:.3f})")
        L.append("")
    with open(os.path.join(RES, "r6_semisynthetic_power_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"R6 finished in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
