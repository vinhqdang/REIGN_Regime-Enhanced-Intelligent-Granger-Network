"""R8 -- contemporaneous portfolio-variance control with an explicit target.

Target: RV_t = (0.5 r_A,t + 0.5 r_B,t)^2 for the 14 curated pairs PAIRS in
experiments/portfolio_variance_synergy.py, daily returns from data/returns_panel.csv.
Every series (RV, r_A, r_B) is standardised predictably (30-step warm-up) before the
feature maps; feature winsorisation is feat_clip=5.  The v2 test is run without
conditioning and with p=1, once with contemp=True (a contemporaneous interaction,
not Granger) and once with contemp=False (the lagged counterpart).  The addjoint and
four scores come from one run_errors call per pair and setting.

Per-pair diagnostics (full-sample raw returns):
  corr(r_A, r_B)                       : cross-sectional co-movement
  sign of E[r_A r_B] (covariance term) : sign of the cross term 0.5 r_A r_B in RV
  R^2 of r_A*r_B on [1, r_A, r_B, r_A^2, r_B^2] : near 1 means the cross term is almost
      a linear combination of the own-quadratic features, so the test cannot separate
      the interaction from the own-variance terms (why near-collinear equity pairs are
      not flagged).

Outputs in experiments/revision/results/:
  r8_portfolio_variance_control.json, r8_portfolio_variance_control_summary.md,
  r8_portfolio_variance_control.png  (200 dpi)
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from common import DATA, RES  # noqa: E402
from ante_v2 import (predictable_standardise, run_errors, bounded_score,  # noqa: E402
                     normalise, bet_process, MODELS)
from portfolio_variance_synergy import PAIRS  # noqa: E402

FORGET, FEAT_CLIP, CAP, GAMMA, WARMUP, ALPHA, P = 0.995, 5.0, 3.0, 0.9, 50, 0.05, 1
M = len(PAIRS)                 # 14
THR_RAW = 1.0 / ALPHA          # per-hypothesis rejection: max E >= 20
THR_FW = M / ALPHA             # family-wise (Bonferroni-style e-value): max E >= m/alpha


def summarise(E, lam, s, p):
    rej = np.where(E >= THR_RAW)[0]
    return dict(final_E=float(E[-1]), max_E=float(E.max()),
                final_log10E=float(np.log10(E[-1])), max_log10E=float(np.log10(E.max())),
                argmax_t=int(np.argmax(E)),
                reject_time=int(rej[0]) if len(rej) else None,
                mean_s=float(np.mean(s[p:])), frac_lam_pos=float(np.mean(lam[p:] > 0)))


def v2_both(Y, A, B, contemp, p=P):
    """One run_errors call; both score kinds from the same predictions."""
    pred, _ = run_errors(Y, A, B, None, p, contemp, FORGET, 1e-2, FEAT_CLIP)
    out = {}
    for kind in ("addjoint", "four"):
        s, b = bounded_score(Y, pred, MODELS, kind, CAP)
        u = normalise(s, b)
        E, lam = bet_process(u, GAMMA, WARMUP, start=p)
        out[kind] = summarise(E, lam, s, p)
    return out


def r2_cross(ra, rb):
    """R^2 of r_A*r_B on [1, r_A, r_B, r_A^2, r_B^2] (OLS, raw returns)."""
    y = ra * rb
    X = np.column_stack([np.ones_like(ra), ra, rb, ra ** 2, rb ** 2])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ w
    return float(1.0 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def main():
    t0 = time.time()
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    rows = []
    for a, b, desc in PAIRS:
        ra_raw = rets[a].values.astype(float)
        rb_raw = rets[b].values.astype(float)
        rv_raw = (0.5 * ra_raw + 0.5 * rb_raw) ** 2
        Y = predictable_standardise(rv_raw, 30)
        A = predictable_standardise(ra_raw, 30)
        B = predictable_standardise(rb_raw, 30)
        cont = v2_both(Y, A, B, contemp=True)
        lag = v2_both(Y, A, B, contemp=False)
        row = dict(pair=f"{a}/{b}", desc=desc,
                   corr=float(np.corrcoef(ra_raw, rb_raw)[0, 1]),
                   cov_sign=int(np.sign(np.mean(ra_raw * rb_raw))),
                   cross_R2=r2_cross(ra_raw, rb_raw),
                   contemp=cont, lagged=lag)
        rows.append(row)
        print(f"{row['pair']:8s} corr={row['corr']:+.3f} covsign={row['cov_sign']:+d} "
              f"R2={row['cross_R2']:.3f} | contemp add {cont['addjoint']['max_log10E']:+.2f} "
              f"four {cont['four']['max_log10E']:+.2f} | lag add {lag['addjoint']['max_log10E']:+.2f} "
              f"four {lag['four']['max_log10E']:+.2f}", flush=True)

    counts = {}
    for setting in ("contemp", "lagged"):
        for kind in ("addjoint", "four"):
            mx = np.array([r[setting][kind]["max_E"] for r in rows])
            counts[f"{setting}_{kind}"] = dict(
                n=M, raw_rejections=int(np.sum(mx >= THR_RAW)),
                familywise_rejections=int(np.sum(mx >= THR_FW)),
                median_max_log10E=float(np.median(np.log10(mx))),
                median_final_log10E=float(np.median([r[setting][kind]["final_log10E"] for r in rows])))

    corr = np.array([r["corr"] for r in rows])
    r2 = np.array([r["cross_R2"] for r in rows])
    m_add = np.array([r["contemp"]["addjoint"]["max_log10E"] for r in rows])
    m_four = np.array([r["contemp"]["four"]["max_log10E"] for r in rows])
    explain = dict(
        spearman_R2_vs_contemp_addjoint=float(spearmanr(r2, m_add)[0]),
        spearman_corr_vs_contemp_addjoint=float(spearmanr(corr, m_add)[0]),
        spearman_R2_vs_contemp_four=float(spearmanr(r2, m_four)[0]),
        spearman_corr_vs_contemp_four=float(spearmanr(corr, m_four)[0]),
        n_cross_covariance_positive=int(sum(r["cov_sign"] > 0 for r in rows)),
        R2_min=float(r2.min()), R2_max=float(r2.max()),
        corr_min=float(corr.min()), corr_max=float(corr.max()),
    )

    out = dict(
        description="R8 portfolio-variance control, explicit target RV=(0.5 rA+0.5 rB)^2",
        data="data/returns_panel.csv", n_pairs=M, p=P, forget=FORGET, feat_clip=FEAT_CLIP,
        cap=CAP, gamma=GAMMA, warmup=WARMUP, alpha=ALPHA, standardisation="predictable, warm-up 30",
        thresholds=dict(raw=THR_RAW, familywise=THR_FW, familywise_log10=float(np.log10(THR_FW))),
        counts=counts, explain=explain, pairs=rows, runtime_s=round(time.time() - t0, 1),
        n_rep=None, deterministic=True,
    )
    with open(os.path.join(RES, "r8_portfolio_variance_control.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- markdown summary
    L = ["# R8 summary: portfolio-variance control (explicit RV target)", "",
         f"Target RV_t=(0.5 r_A+0.5 r_B)^2, predictable standardisation, feat_clip=5, p=1, no conditioning. "
         f"Thresholds: raw max E>=20; family-wise max E>=m/alpha={THR_FW:.0f} (m={M}).", "",
         "## Per pair", "",
         "| pair | description | corr(rA,rB) | cov sign | R^2(rA rB on 1,rA,rB,rA^2,rB^2) | "
         "contemp addjoint final / max log10E | contemp four final / max log10E | "
         "lagged addjoint final / max log10E | lagged four final / max log10E |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, l = r["contemp"], r["lagged"]
        L.append(f"| {r['pair']} | {r['desc']} | {r['corr']:+.3f} | {r['cov_sign']:+d} | {r['cross_R2']:.3f} | "
                 f"{c['addjoint']['final_log10E']:+.2f} / {c['addjoint']['max_log10E']:+.2f} | "
                 f"{c['four']['final_log10E']:+.2f} / {c['four']['max_log10E']:+.2f} | "
                 f"{l['addjoint']['final_log10E']:+.2f} / {l['addjoint']['max_log10E']:+.2f} | "
                 f"{l['four']['final_log10E']:+.2f} / {l['four']['max_log10E']:+.2f} |")
    L += ["", "## Rejection counts", "",
          "| setting / score | raw (max E>=20) | family-wise (max E>=280) | median max log10E |",
          "|---|---|---|---|"]
    for k, v in counts.items():
        L.append(f"| {k} | {v['raw_rejections']}/{M} | {v['familywise_rejections']}/{M} | "
                 f"{v['median_max_log10E']:+.2f} |")
    L += ["", "## Explanation numbers", "",
          f"- Spearman(cross R^2, contemp addjoint max log10E) = {explain['spearman_R2_vs_contemp_addjoint']:+.2f}",
          f"- Spearman(corr, contemp addjoint max log10E) = {explain['spearman_corr_vs_contemp_addjoint']:+.2f}",
          f"- Spearman(cross R^2, contemp four max log10E) = {explain['spearman_R2_vs_contemp_four']:+.2f}",
          f"- Spearman(corr, contemp four max log10E) = {explain['spearman_corr_vs_contemp_four']:+.2f}",
          f"- cross-term R^2 range [{explain['R2_min']:.3f}, {explain['R2_max']:.3f}]; "
          f"corr range [{explain['corr_min']:+.3f}, {explain['corr_max']:+.3f}]; "
          f"{explain['n_cross_covariance_positive']}/{M} pairs with positive covariance term.", ""]
    with open(os.path.join(RES, "r8_portfolio_variance_control_summary.md"), "w") as f:
        f.write("\n".join(L))

    # ---------------- figure: max log10E per pair, addjoint vs four, contemp vs lagged
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(M)
    labels = [r["pair"] for r in rows]
    for ax, setting, title in zip(axes, ("contemp", "lagged"),
                                  ("contemporaneous (contemp=True)", "lagged (contemp=False, p=1)")):
        va = [r[setting]["addjoint"]["max_log10E"] for r in rows]
        vf = [r[setting]["four"]["max_log10E"] for r in rows]
        ax.bar(x - 0.2, va, 0.4, label="addjoint", color="#4c72b0", edgecolor="black", lw=0.4)
        ax.bar(x + 0.2, vf, 0.4, label="four", color="#dd8452", edgecolor="black", lw=0.4)
        ax.axhline(np.log10(THR_RAW), color="red", ls="--", lw=1.1, label="raw threshold log10(20)")
        ax.axhline(np.log10(THR_FW), color="purple", ls=":", lw=1.3,
                   label=f"family-wise log10(m/alpha), m={M}")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(r"max $\log_{10} E_t$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("R8: portfolio-variance control, explicit RV target (etf16 daily returns)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "r8_portfolio_variance_control.png"), dpi=200)
    plt.close(fig)

    print(json.dumps(counts, indent=1))
    print(f"runtime {out['runtime_s']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
