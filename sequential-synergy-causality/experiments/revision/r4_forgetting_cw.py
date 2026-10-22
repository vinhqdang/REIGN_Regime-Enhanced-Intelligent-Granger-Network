#!/usr/bin/env python3
"""R4: forgetting sensitivity, Clark-West-adjusted score, and the null-vs-finance gap.

(a) forget in {0.98, 0.99, 0.995, 0.999} on
      - additive null DGP (gen_var 'null', rho=0.5, T=2500, n_rep=100),
      - 150 financial triples (seed 0) from panel stocks13, target 'vol' (|r|),
        lag 1, conditioned on SPY, predictable standardisation (warm-up 30),
        feat_clip=5.
    Reports median final / running-max log10 E, rejection rate (max_E >= 20) with
    Wilson CI, mean fraction of steps with lambda > 0, mean raw bounded score s
    (addjoint), mean Clark-West-adjusted score s + (p_add - p_joint)^2, mean s^2.
(b) Diagnostics of the null-vs-finance gap, for the same finance triples and for
    the garch+heavy null (rho=0.5, n_rep=100): fraction of steps with lambda > 0,
    median log10 product over active-bet episodes (contiguous runs of lambda>0),
    decomposition of log10 final E into thirds of the sample, and the fraction of
    triples with running-max log10 E > +0.3 and final log10 E < -0.3 (with the
    sign pattern of the running mean of s around the peak).
(c) Table (JSON + markdown) and figure (final log10 E distributions per forget).

Outputs (experiments/revision/results/): r4_forgetting_cw.json,
r4_forgetting_cw_summary.md, r4_forgetting_cw.png, r4_finance_triples.csv
"""
import os
import sys
import json
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
np.seterr(all="ignore")

from common import gen_var, rate_ci, load_panel, target_series, RES  # noqa: E402
from ante_v2 import (run_errors, bounded_score, normalise, bet_process,  # noqa: E402
                     cw_adjustment, predictable_standardise, MODELS)

FORGETS = [0.98, 0.99, 0.995, 0.999]
N_REP = 100
N_TRIPLES = 150
N_PROC = 2
RHO = 0.5
RIDGE = 1e-2
CAP = 3.0
GAMMA = 0.9
WARMUP = 50
P = 1
ALPHA = 0.05
THR = 1.0 / ALPHA
FIN_CLIP = 5.0
SIM_T = 2500
SEED_NULL = 401
SEED_GARCH = 402


# ---------------------------------------------------------------- core run
def run_all_forgets(Y, A, B, Z, clip):
    """Returns {forget: metrics dict} for one (Y, A, B, cond) series."""
    T = len(Y)
    out = {}
    for f in FORGETS:
        pred, _ = run_errors(Y, A, B, cond=[Z], p=P, forget=f, ridge=RIDGE,
                             feat_clip=clip, models=MODELS)
        s, b = bounded_score(Y, pred, MODELS, "addjoint", CAP)
        u = normalise(s, b)
        E, lam = bet_process(u, GAMMA, WARMUP, start=P)
        cw = cw_adjustment(pred)
        ss, uu, ll = s[P:], u[P:], lam[P:]
        inc = np.log10(1.0 + ll * uu)            # log10 factor per step (0 if lam=0)
        logE = np.log10(np.maximum(E, 1e-300))
        runmax = np.maximum.accumulate(E)
        log_runmax = np.log10(runmax)
        final = float(logE[-1])
        maxl = float(log_runmax[-1])
        tstar = int(np.argmax(E))
        n = len(inc)
        c1, c2 = n // 3, 2 * n // 3
        thirds = [float(inc[:c1].sum()), float(inc[c1:c2].sum()), float(inc[c2:].sum())]
        # contiguous episodes of lambda > 0
        act = (ll > 0).astype(int)
        d = np.diff(np.concatenate([[0], act, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        episodes = [float(inc[i:j].sum()) for i, j in zip(starts, ends)]
        before = ss[: max(tstar - P, 0) + 1]
        after = ss[max(tstar - P, 0) + 1:]
        reversal = bool(maxl > 0.3 and final < -0.3)
        out[str(f)] = dict(
            final_log10E=final,
            max_log10E=maxl,
            reject=bool(E.max() >= THR),
            argmax_frac=float(tstar / max(T - 1, 1)),
            frac_lam_pos=float(np.mean(ll > 0)),
            mean_s=float(np.mean(ss)),
            mean_cw_adj=float(np.mean(ss + cw[P:])),
            mean_s2=float(np.mean(ss ** 2)),
            thirds=thirds,
            sum_inc_check=float(inc.sum() - final),
            episodes=episodes,
            n_episodes=len(episodes),
            reversal=reversal,
            mean_s_before_peak=float(np.mean(before)) if len(before) else np.nan,
            mean_s_after_peak=float(np.mean(after)) if len(after) else np.nan,
            reversal_pos_then_neg=bool(reversal and np.mean(before) > 0
                                       and len(after) and np.mean(after) < 0),
        )
    return out


# ---------------------------------------------------------------- workers
def task_null(args):
    kind, rep = args
    if kind == "null":
        rng = np.random.default_rng([SEED_NULL, rep])
        Y, A, B, Z = gen_var("null", SIM_T, rng, rho=RHO)
    else:
        rng = np.random.default_rng([SEED_GARCH, rep])
        Y, A, B, Z = gen_var("null", SIM_T, rng, rho=RHO, heavy=True, garch=True)
    return dict(group=kind, rep=rep, res=run_all_forgets(Y, A, B, Z, clip=None))


def task_fin(args):
    idx, C, A, B, Y, Ya, Yb, Yc = args
    return dict(group="finance", idx=idx, C=C, A=A, B=B,
                res=run_all_forgets(Y, Ya, Yb, Yc, clip=FIN_CLIP))


# ---------------------------------------------------------------- aggregation
def agg(recs, f):
    """Aggregate one group at one forget value."""
    key = str(f)
    rows = [r["res"][key] for r in recs]
    fin = np.array([x["final_log10E"] for x in rows])
    mx = np.array([x["max_log10E"] for x in rows])
    rej = rate_ci([x["reject"] for x in rows])
    thirds = np.array([x["thirds"] for x in rows])
    eps = np.concatenate([np.array(x["episodes"]) for x in rows]) if any(x["episodes"] for x in rows) else np.array([])
    rev = np.array([x["reversal"] for x in rows])
    neg_final = fin < 0
    return dict(
        n=len(rows),
        median_final_log10E=float(np.median(fin)),
        median_max_log10E=float(np.median(mx)),
        mean_final_log10E=float(np.mean(fin)),
        q10_final=float(np.quantile(fin, 0.1)), q90_final=float(np.quantile(fin, 0.9)),
        reject_rate=rej["rate"], reject_ci=[rej["lo"], rej["hi"]], reject_k=rej["k"],
        mean_frac_lam_pos=float(np.mean([x["frac_lam_pos"] for x in rows])),
        mean_s=float(np.mean([x["mean_s"] for x in rows])),
        mean_cw_adj=float(np.mean([x["mean_cw_adj"] for x in rows])),
        mean_s2=float(np.mean([x["mean_s2"] for x in rows])),
        median_log10_per_active_episode=float(np.median(eps)) if len(eps) else np.nan,
        n_episodes_total=int(len(eps)),
        frac_episodes_negative=float(np.mean(eps < 0)) if len(eps) else np.nan,
        median_thirds=[float(np.median(thirds[:, k])) for k in range(3)],
        mean_thirds=[float(np.mean(thirds[:, k])) for k in range(3)],
        frac_runmax_gt_0p3_and_final_lt_m0p3=float(np.mean(rev)),
        frac_final_negative=float(np.mean(neg_final)),
        frac_of_final_negative_with_reversal=(float(np.mean(rev[neg_final]))
                                              if neg_final.any() else np.nan),
        frac_reversal_pos_then_neg=float(np.mean([x["reversal_pos_then_neg"] for x in rows])),
        max_sum_check_abs=float(np.max(np.abs([x["sum_inc_check"] for x in rows]))),
    )


def main():
    t0 = time.time()
    # ------------------------------------------------ finance data and triples
    r, cond = load_panel("stocks13")          # daily returns, cond = SPY
    vol = target_series(r, "vol")              # |r_t|
    std = {c: predictable_standardise(vol[c].values, warmup=30) for c in vol.columns}
    assets = [c for c in r.columns if c != cond]           # 12 non-conditioning assets
    hyps = []
    for C in assets:
        others = [c for c in assets if c != C]
        for A_, B_ in combinations(others, 2):
            hyps.append((C, A_, B_))
    assert len(hyps) == 660, len(hyps)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(hyps), N_TRIPLES, replace=False).tolist())
    triples = [hyps[i] for i in pick]
    Zc = std[cond]
    print(f"finance: {len(hyps)} hypotheses, {len(triples)} sampled, T={len(r)}", flush=True)

    fin_tasks = []
    for k, (C, A_, B_) in zip(pick, triples):
        fin_tasks.append((k, C, A_, B_, std[C], std[A_], std[B_], Zc))

    sim_tasks = [("null", rep) for rep in range(N_REP)] + [("garch", rep) for rep in range(N_REP)]

    recs = {"null": [], "garch": [], "finance": []}
    done = 0
    total = len(sim_tasks) + len(fin_tasks)
    with Pool(N_PROC) as pool:
        for out in pool.imap(task_null, sim_tasks, chunksize=2):
            recs[out["group"]].append(out)
            done += 1
            if done % 50 == 0:
                print(f"  progress {done}/{total} ({time.time() - t0:.0f}s)", flush=True)
        for out in pool.imap(task_fin, fin_tasks, chunksize=2):
            recs["finance"].append(out)
            done += 1
            if done % 50 == 0:
                print(f"  progress {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------ per-triple CSV
    rows = []
    for o in sorted(recs["finance"], key=lambda z: z["idx"]):
        for f in FORGETS:
            m = o["res"][str(f)]
            rows.append(dict(idx=o["idx"], C=o["C"], A=o["A"], B=o["B"], forget=f,
                             final_log10E=m["final_log10E"], max_log10E=m["max_log10E"],
                             reject=m["reject"], frac_lam_pos=m["frac_lam_pos"],
                             mean_s=m["mean_s"], mean_cw_adj=m["mean_cw_adj"],
                             reversal=m["reversal"]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "r4_finance_triples.csv"), index=False)

    # ------------------------------------------------ aggregates
    part = {}
    for f in FORGETS:
        part[str(f)] = {g: agg(recs[g], f) for g in ("null", "garch", "finance")}
    garch_only = {str(f): agg(recs["garch"], f) for f in FORGETS}
    null_only = {str(f): agg(recs["null"], f) for f in FORGETS}

    settings = dict(
        forgets=FORGETS, n_rep_null=N_REP, n_rep_garch=N_REP, n_triples=N_TRIPLES,
        triple_sampling="default_rng(0).choice(660 hypotheses, 150, replace=False); hypotheses = C in 12 non-SPY stocks13 assets x unordered pair from the other 11 non-SPY assets",
        panel="stocks13", target="vol (|r|, predictable standardisation warm-up 30), sources transformed the same way, conditioning SPY (|r|, standardised)",
        p=P, kind="addjoint", cap=CAP, gamma=GAMMA, warmup=WARMUP, alpha=ALPHA,
        ridge=RIDGE, feat_clip_finance=FIN_CLIP, feat_clip_sim=None,
        sim_dgp="gen_var('null', T=2500, rho=0.5); garch = heavy=True, garch=True",
        seeds=dict(null=SEED_NULL, garch=SEED_GARCH, finance="deterministic (seed 0 sampling)"),
        bettor="bet_process(u, gamma=0.9, warmup=50, start=1); reject if max E >= 20",
        cw_adjustment="mean of s_t + (p_add - p_joint)^2 with s_t the truncated (addjoint) score",
        episode_definition="contiguous runs of steps with lambda_t > 0; log10 product of (1 + lambda u) over the run",
        reversal_definition="running-max log10 E > 0.3 and final log10 E < -0.3; reversal_pos_then_neg additionally requires mean s before the peak > 0 and mean s after the peak < 0",
    )
    out = dict(settings=settings, part_a_b=part,
               null_only_by_forget=null_only, garch_only_by_forget=garch_only,
               finance_triples_csv="results/r4_finance_triples.csv",
               runtime_s=float(time.time() - t0))
    with open(os.path.join(RES, "r4_forgetting_cw.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # ------------------------------------------------ markdown summary
    L = ["# R4 summary: forgetting sensitivity, CW score, null vs finance gap", "",
         f"n_rep (null, garch+heavy) = {N_REP}; finance triples = {N_TRIPLES} (seed 0), "
         f"T = {SIM_T} for simulations, {len(r)} for finance.", "",
         "## (a) Forgetting sensitivity", "",
         "| forget | group | median final log10E | median max log10E | reject rate [95% CI] | "
         "mean frac lambda>0 | mean s | mean CW-adj s | mean s^2 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for f in FORGETS:
        for g, lab in (("null", "null rho=.5"), ("garch", "garch+heavy null"), ("finance", "finance (150)")):
            if g == "garch":
                a = agg(recs["garch"], f)
            else:
                a = part[str(f)][g]
            L.append(f"| {f} | {lab} | {a['median_final_log10E']:.3f} | {a['median_max_log10E']:.3f} | "
                     f"{a['reject_rate']:.3f} [{a['reject_ci'][0]:.3f}, {a['reject_ci'][1]:.3f}] | "
                     f"{a['mean_frac_lam_pos']:.3f} | {a['mean_s']:.4f} | {a['mean_cw_adj']:.4f} | {a['mean_s2']:.4f} |")
    L += ["", "## (b) Null vs finance gap diagnostics", "",
          "| forget | group | median log10 per active episode | frac episodes < 0 | "
          "median thirds (1st / mid / last) | frac runmax>0.3 & final<-0.3 | frac final<0 with reversal | "
          "reversal with pos-then-neg running mean |",
          "|---|---|---|---|---|---|---|---|"]
    for f in FORGETS:
        for g, lab in (("null", "null rho=.5"), ("garch", "garch+heavy null"), ("finance", "finance (150)")):
            a = agg(recs["garch"], f) if g == "garch" else part[str(f)][g]
            th = a["median_thirds"]
            L.append(f"| {f} | {lab} | {a['median_log10_per_active_episode']:.3f} | "
                     f"{a['frac_episodes_negative']:.3f} | {th[0]:.3f} / {th[1]:.3f} / {th[2]:.3f} | "
                     f"{a['frac_runmax_gt_0p3_and_final_lt_m0p3']:.3f} | "
                     f"{a['frac_of_final_negative_with_reversal']:.3f} | "
                     f"{a['frac_reversal_pos_then_neg']:.3f} |")
    L += ["", f"Runtime: {out['runtime_s']:.0f}s"]
    with open(os.path.join(RES, "r4_forgetting_cw_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    # ------------------------------------------------ figure: distributions per forget
    fig, axes = plt.subplots(1, len(FORGETS), figsize=(16, 4.8), sharey=True)
    colors = {"null": "#4c72b0", "garch": "#dd8452", "finance": "#c44e52"}
    for ax, f in zip(axes, FORGETS):
        data = [[x["res"][str(f)]["final_log10E"] for x in recs[g]] for g in ("null", "garch", "finance")]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for i, d in enumerate(data, start=1):
            jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
            ax.scatter(np.full(len(d), i) + jit, d, s=6, color="black", alpha=0.35, zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(-0.3, color="gray", ls=":", lw=0.8)
        ax.axhline(0.3, color="gray", ls=":", lw=0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["null", "garch+\nheavy", "finance"])
        ax.set_title(f"forget = {f}")
    axes[0].set_ylabel("final log10 E")
    fig.suptitle("R4: final log10 E by forgetting factor (addjoint, lag 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "r4_forgetting_cw.png"), dpi=200)
    print(f"done in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
