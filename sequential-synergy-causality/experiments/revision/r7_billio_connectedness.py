"""R7 -- Billio et al. (2012) pairwise Granger-connectedness layer.

(i)  Full-sample pairwise network on panels etf16 and stocks13 (daily log returns, and
     |r| as a volatility series), for all ordered pairs i->j with pairwise_granger_f
     (p=1 and p=5).  Edges are counted at raw alpha=.05 and Bonferroni alpha/(N(N-1)).
     DGC = fraction of significant ordered pairs.
(ii) Rolling 252-day window DGC (step 21 days, raw .05, p=1, returns).  Windows are
     labelled by their end date; a window is flagged 'covid' if its span overlaps
     2020-02-01..2020-05-31 and 'y2022' if its span overlaps calendar 2022.
(iii) Same hypothesis space as R5 lag1 for stocks13 target 'vol' (cond SPY, 660 triples):
     number of triples where BOTH single-source Granger F-tests (A->C and B->C) are
     significant, versus the number with a v2 addjoint rejection.  The R5 CSV
     results/r5_scan_stocks13_vol_lag1.csv is reused if present; otherwise the v2 scan
     is recomputed here with one process and cached in
     results/r7_scan_stocks13_vol_lag1_recomputed.csv.  Single-source F-tests are
     reported both as pairwise_granger_f (unconditional) and conditional on the SPY lag
     (same OLS F with one extra regressor).  The family for the single-source tests is
     the set of distinct ordered pairs (A->C), 12 x 11 = 132, so Bonferroni alpha is
     0.05/132.  The addjoint family-wise threshold is m/alpha with m=660.

Outputs in experiments/revision/results/:
  r7_billio_connectedness.json, r7_billio_connectedness_summary.md,
  r7_billio_connectedness.png (200 dpi)
"""
import os
import sys
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f as fdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from common import load_panel, target_series, RES  # noqa: E402
from synergistic_granger import pairwise_granger_f  # noqa: E402
from ante_v2 import (predictable_standardise, run_errors, bounded_score,  # noqa: E402
                     normalise, bet_process, MODELS)

ALPHA = 0.05
PANELS = ["etf16", "stocks13"]
R5_CSV = os.path.join(RES, "r5_scan_stocks13_vol_lag1.csv")
RECOMP_CSV = os.path.join(RES, "r7_scan_stocks13_vol_lag1_recomputed.csv")
FORGET, FEAT_CLIP, CAP, GAMMA, WARMUP = 0.995, 5.0, 3.0, 0.9, 50
THR_RAW = 1.0 / ALPHA


# ---------------------------------------------------------------- Granger helpers
def granger_F_cond(y, x, z, p=1):
    """Granger F for x->y conditional on z (lags of y, x and z; intercept)."""
    T = len(y)
    rows_y, Xr, Xf = [], [], []
    for t in range(p, T):
        ylag = [y[t - 1 - j] for j in range(p)]
        zlag = [z[t - 1 - j] for j in range(p)]
        xlag = [x[t - 1 - j] for j in range(p)]
        base = [1.0] + ylag + zlag
        rows_y.append(y[t]); Xr.append(base); Xf.append(base + xlag)
    yy, Xr, Xf = np.array(rows_y), np.array(Xr), np.array(Xf)

    def rss(M):
        w, *_ = np.linalg.lstsq(M, yy, rcond=None)
        return float(np.sum((yy - M @ w) ** 2))

    r0, r1 = rss(Xr), rss(Xf)
    n, k = len(yy), Xf.shape[1]
    F = ((r0 - r1) / p) / (r1 / (n - k))
    return dict(F=float(F), pval=float(fdist.sf(F, p, n - k)))


def pairwise_matrix(X, p):
    """pv[i, j] = p-value of the test i -> j (pairwise_granger_f(Y=col j, X=col i))."""
    N = X.shape[1]
    pv = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            if i != j:
                pv[i, j] = pairwise_granger_f(X[:, j], X[:, i], p=p)["pval"]
    return pv


def dgc_from_pv(pv, cols):
    N = pv.shape[0]
    m = N * (N - 1)
    off = ~np.eye(N, dtype=bool)
    p_off = pv[off]
    raw = p_off < ALPHA
    bon = p_off < ALPHA / m
    edges_raw = [[cols[i], cols[j], float(pv[i, j])]
                 for i in range(N) for j in range(N) if i != j and pv[i, j] < ALPHA]
    edges_raw.sort(key=lambda e: e[2])
    return dict(N=N, m=m, raw_alpha=ALPHA, bonf_alpha=ALPHA / m,
                raw_count=int(raw.sum()), raw_dgc=float(raw.mean()),
                bonf_count=int(bon.sum()), bonf_dgc=float(bon.mean()),
                raw_edges=edges_raw)


# ---------------------------------------------------------------- (i) full sample
def full_sample(panel_returns):
    out = {}
    for kind in ("ret", "vol"):
        series = panel_returns if kind == "ret" else target_series(panel_returns, "vol")
        X = series.values.astype(float)
        cols = list(series.columns)
        for p in (1, 5):
            pv = pairwise_matrix(X, p)
            res = dgc_from_pv(pv, cols)
            res["pvalue_matrix"] = np.round(pv, 6).tolist()
            res["columns"] = cols
            out[f"{kind}_p{p}"] = res
            print(f"  full {kind} p={p}: N={res['N']} raw {res['raw_count']}/{res['m']} "
                  f"(DGC {res['raw_dgc']:.3f}), Bonferroni {res['bonf_count']}/{res['m']}", flush=True)
    return out


# ---------------------------------------------------------------- (ii) rolling
def rolling(panel_returns, win=252, step=21, p=1):
    X = panel_returns.values.astype(float)
    idx = panel_returns.index
    T, N = X.shape
    rows = []
    for s in range(0, T - win + 1, step):
        seg = X[s:s + win]
        pv = pairwise_matrix(seg, p)
        off = ~np.eye(N, dtype=bool)
        dgc = float((pv[off] < ALPHA).mean())
        start, end = idx[s], idx[s + win - 1]
        covid = (start <= pd.Timestamp("2020-05-31")) and (end >= pd.Timestamp("2020-02-01"))
        y2022 = (start <= pd.Timestamp("2022-12-31")) and (end >= pd.Timestamp("2022-01-01"))
        rows.append(dict(start=str(start.date()), end=str(end.date()), dgc=dgc,
                         covid=bool(covid), y2022=bool(y2022)))
    return rows


# ---------------------------------------------------------------- (iii) triples
def v2_addjoint_scan(vol, cond_col):
    """Recompute the v2 addjoint/four scan for the stocks13 'vol' lag1 space (1 process)."""
    cols = list(vol.columns)
    z = {c: predictable_standardise(vol[c].values, warmup=30) for c in cols}
    zc = z[cond_col]
    targets = [c for c in cols if c != cond_col]
    rows = []
    t0 = time.time()
    total = len(targets) * (len(targets) - 1) * (len(targets) - 2) // 2
    k = 0
    for C in targets:
        srcs = [c for c in targets if c != C]
        for A, B in combinations(srcs, 2):
            pred, _ = run_errors(z[C], z[A], z[B], [zc], p=1, contemp=False,
                                 forget=FORGET, feat_clip=FEAT_CLIP)
            rec = dict(A=A, B=B, C=C)
            for kind in ("addjoint", "four"):
                s, b = bounded_score(z[C], pred, MODELS, kind, CAP)
                u = normalise(s, b)
                E, _ = bet_process(u, GAMMA, WARMUP, start=1)
                rej = np.where(E >= THR_RAW)[0]
                rec[f"{kind}_final_log10E"] = float(np.log10(E[-1]))
                rec[f"{kind}_max_log10E"] = float(np.log10(E.max()))
                rec[f"{kind}_max_E"] = float(E.max())
                rec[f"{kind}_reject_time"] = int(rej[0]) if len(rej) else -1
            rows.append(rec)
            k += 1
            if k % 50 == 0 or k == total:
                print(f"  scan {k}/{total} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def single_source_F(vol, cond_col):
    """Unique ordered pairs (A->C), A,C in the 12 non-conditioner assets."""
    cols = [c for c in vol.columns if c != cond_col]
    res = {}
    for C in cols:
        for A in cols:
            if A == C:
                continue
            y = vol[C].values.astype(float)
            x = vol[A].values.astype(float)
            z = vol[cond_col].values.astype(float)
            res[(A, C)] = dict(p_uncond=pairwise_granger_f(y, x, p=1)["pval"],
                               p_cond=granger_F_cond(y, x, z, p=1)["pval"])
    return res


def triples_analysis(vol, cond_col):
    if os.path.exists(RECOMP_CSV):
        df = pd.read_csv(RECOMP_CSV)
        src = RECOMP_CSV + " (cached recompute)"
    else:
        df = v2_addjoint_scan(vol, cond_col)
        df.to_csv(RECOMP_CSV, index=False)
        src = RECOMP_CSV + " (recomputed)"
    F = single_source_F(vol, cond_col)
    n_uniq = len(F)                    # 132
    thr_bonf = ALPHA / n_uniq
    fw_thr = len(df) / ALPHA           # m = 660

    ok_u_raw = np.array([F[(a, c)]["p_uncond"] < ALPHA for a, c in zip(df.A, df.C)])
    ok_u_bon = np.array([F[(a, c)]["p_uncond"] < thr_bonf for a, c in zip(df.A, df.C)])
    ok_c_raw = np.array([F[(a, c)]["p_cond"] < ALPHA for a, c in zip(df.A, df.C)])
    ok_c_bon = np.array([F[(a, c)]["p_cond"] < thr_bonf for a, c in zip(df.A, df.C)])
    # per-triple single-source F p-values (A->C and B->C)
    pAC_u = np.array([F[(a, c)]["p_uncond"] for a, c in zip(df.A, df.C)])
    pBC_u = np.array([F[(b, c)]["p_uncond"] for b, c in zip(df.B, df.C)])
    pAC_c = np.array([F[(a, c)]["p_cond"] for a, c in zip(df.A, df.C)])
    pBC_c = np.array([F[(b, c)]["p_cond"] for b, c in zip(df.B, df.C)])
    both_u_raw = (pAC_u < ALPHA) & (pBC_u < ALPHA)
    both_u_bon = (pAC_u < thr_bonf) & (pBC_u < thr_bonf)
    both_c_raw = (pAC_c < ALPHA) & (pBC_c < ALPHA)
    both_c_bon = (pAC_c < thr_bonf) & (pBC_c < thr_bonf)

    add_raw = df["addjoint_max_E"].values >= THR_RAW
    add_fw = df["addjoint_max_E"].values >= fw_thr
    four_raw = df["four_max_E"].values >= THR_RAW

    def table(x, y):
        return dict(both=int(np.sum(x & y)), x_only=int(np.sum(x & ~y)),
                    y_only=int(np.sum(~x & y)), neither=int(np.sum(~x & ~y)))

    summary = dict(
        source=src, n_triples=int(len(df)), n_unique_single_source_tests=n_uniq,
        bonf_alpha_single_source=thr_bonf, addjoint_familywise_threshold=fw_thr,
        both_Ftests_uncond_raw=int(both_u_raw.sum()), both_Ftests_uncond_bonf=int(both_u_bon.sum()),
        both_Ftests_cond_raw=int(both_c_raw.sum()), both_Ftests_cond_bonf=int(both_c_bon.sum()),
        single_Ftests_uncond_raw_pairs=int(ok_u_raw.sum()), single_Ftests_cond_raw_pairs=int(ok_c_raw.sum()),
        addjoint_reject_raw=int(add_raw.sum()), addjoint_reject_familywise=int(add_fw.sum()),
        four_reject_raw=int(four_raw.sum()),
        table_uncond_bonf_vs_addjoint_raw=table(both_u_bon, add_raw),
        table_cond_bonf_vs_addjoint_raw=table(both_c_bon, add_raw),
        table_uncond_raw_vs_addjoint_raw=table(both_u_raw, add_raw),
        table_cond_raw_vs_addjoint_raw=table(both_c_raw, add_raw),
        addjoint_reject_and_no_both_Ftests_cond_raw=int(np.sum(add_raw & ~both_c_raw)),
        median_addjoint_max_log10E=float(np.median(df["addjoint_max_log10E"])),
        median_four_max_log10E=float(np.median(df["four_max_log10E"])),
        triples_table=df.assign(pAC_uncond=pAC_u, pBC_uncond=pBC_u, pAC_cond=pAC_c,
                                pBC_cond=pBC_c).round(6).to_dict(orient="records"),
    )
    return summary


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if os.path.exists(R5_CSV):
        # The R5 CSV schema is not defined yet, so it cannot be mapped here. Stop before any work.
        raise SystemExit(f"{R5_CSV} exists: map its columns to v2_addjoint_scan's schema first")
    out = dict(description="R7 Billio-style pairwise Granger connectedness", alpha=ALPHA,
               panels=PANELS, settings=dict(window=252, step=21, rolling_p=1, full_p=[1, 5],
                                            kinds=["ret", "vol"], scan_forget=FORGET,
                                            scan_feat_clip=FEAT_CLIP, scan_cap=CAP, scan_gamma=GAMMA,
                                            scan_warmup=WARMUP, scan_standardisation="predictable, 30"),
               deterministic=True, n_rep=None)
    full, roll = {}, {}
    for panel in PANELS:
        r, cond = load_panel(panel)
        print(f"panel {panel}: {r.shape[1]} assets, {r.shape[0]} days, cond {cond}", flush=True)
        full[panel] = full_sample(r)
        roll[panel] = rolling(r)
        vals = np.array([w["dgc"] for w in roll[panel]])
        print(f"  rolling {panel}: {len(vals)} windows, DGC min {vals.min():.3f} max {vals.max():.3f}",
              flush=True)
    out["full_sample"] = {p: {k: {kk: vv for kk, vv in v.items() if kk != "pvalue_matrix"}
                              for k, v in full[p].items()} for p in PANELS}
    out["pvalue_matrices"] = {p: {k: v["pvalue_matrix"] for k, v in full[p].items()} for p in PANELS}
    out["rolling"] = roll

    rolling_stats = {}
    for panel in PANELS:
        w = roll[panel]
        d = np.array([x["dgc"] for x in w])
        cv = np.array([x["dgc"] for x in w if x["covid"]])
        y22 = np.array([x["dgc"] for x in w if x["y2022"]])
        rolling_stats[panel] = dict(
            n_windows=len(w), min=float(d.min()), max=float(d.max()), median=float(np.median(d)),
            argmin_end=w[int(np.argmin(d))]["end"], argmax_end=w[int(np.argmax(d))]["end"],
            n_covid=int(len(cv)), covid_mean=float(cv.mean()) if len(cv) else None,
            covid_min=float(cv.min()) if len(cv) else None, covid_max=float(cv.max()) if len(cv) else None,
            n_2022=int(len(y22)), y2022_mean=float(y22.mean()) if len(y22) else None,
            y2022_min=float(y22.min()) if len(y22) else None, y2022_max=float(y22.max()) if len(y22) else None,
        )
    out["rolling_stats"] = rolling_stats

    # (iii) triples, stocks13 vol lag1
    r13, cond13 = load_panel("stocks13")
    vol13 = target_series(r13, "vol")
    print("triples analysis (stocks13 vol lag1)", flush=True)
    out["triples_stocks13_vol_lag1"] = triples_analysis(vol13, cond13)
    s3 = out["triples_stocks13_vol_lag1"]
    print(f"  triples: n={s3['n_triples']}, addjoint raw {s3['addjoint_reject_raw']}, "
          f"familywise {s3['addjoint_reject_familywise']}, both F bonf (uncond) "
          f"{s3['both_Ftests_uncond_bonf']}, cond {s3['both_Ftests_cond_bonf']}", flush=True)
    out["runtime_s"] = None  # set below
    tri_copy = out["triples_stocks13_vol_lag1"].pop("triples_table")

    # JSON (triples table to its own CSV to keep JSON compact)
    pd.DataFrame(tri_copy).to_csv(os.path.join(RES, "r7_triples_stocks13_vol_lag1.csv"), index=False)
    out["triples_stocks13_vol_lag1"]["triples_csv"] = "results/r7_triples_stocks13_vol_lag1.csv"
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "r7_billio_connectedness.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---------------- markdown summary
    L = ["# R7 summary: Billio-style pairwise Granger connectedness", "",
         "Pairwise tests: pairwise_granger_f, ordered pairs i->j, raw alpha=.05 and Bonferroni alpha/(N(N-1)).", "",
         "## (i) Full-sample network", "",
         "| panel | series | p | N | raw edges | raw DGC | Bonferroni edges | Bonferroni DGC |",
         "|---|---|---|---|---|---|---|---|"]
    for panel in PANELS:
        for kind in ("ret", "vol"):
            for p in (1, 5):
                d = full[panel][f"{kind}_p{p}"]
                L.append(f"| {panel} | {kind} | {p} | {d['N']} | {d['raw_count']}/{d['m']} | {d['raw_dgc']:.3f} | "
                         f"{d['bonf_count']}/{d['m']} | {d['bonf_dgc']:.3f} |")
    L += ["", "## (ii) Rolling 252-day DGC (returns, p=1, raw .05, step 21)", "",
          "| panel | windows | min DGC (end) | max DGC (end) | median | covid-overlap windows: mean [min, max] | 2022-overlap windows: mean [min, max] |",
          "|---|---|---|---|---|---|---|"]
    for panel in PANELS:
        s = rolling_stats[panel]
        cv = f"{s['covid_mean']:.3f} [{s['covid_min']:.3f}, {s['covid_max']:.3f}] (n={s['n_covid']})" if s["n_covid"] else "none"
        y2 = f"{s['y2022_mean']:.3f} [{s['y2022_min']:.3f}, {s['y2022_max']:.3f}] (n={s['n_2022']})" if s["n_2022"] else "none"
        L.append(f"| {panel} | {s['n_windows']} | {s['min']:.3f} ({s['argmin_end']}) | "
                 f"{s['max']:.3f} ({s['argmax_end']}) | {s['median']:.3f} | {cv} | {y2} |")
    L += ["", "## (iii) stocks13 target 'vol', lag1, cond SPY (660 triples)", "",
          f"- scan source: {s3['source']}",
          f"- addjoint rejections raw (max E>=20): {s3['addjoint_reject_raw']}/{s3['n_triples']}; "
          f"family-wise (max E>=m/alpha={s3['addjoint_familywise_threshold']:.0f}): {s3['addjoint_reject_familywise']}/{s3['n_triples']}",
          f"- four rejections raw: {s3['four_reject_raw']}/{s3['n_triples']}",
          f"- single-source F-tests: {s3['n_unique_single_source_tests']} unique (A->C) tests; Bonferroni alpha = {s3['bonf_alpha_single_source']:.2e}",
          f"- triples with BOTH A->C and B->C significant, unconditional F: raw {s3['both_Ftests_uncond_raw']}, Bonferroni {s3['both_Ftests_uncond_bonf']}",
          f"- triples with BOTH significant, conditional on SPY F: raw {s3['both_Ftests_cond_raw']}, Bonferroni {s3['both_Ftests_cond_bonf']}",
          f"- contingency (both F conditional Bonferroni) x (addjoint raw): {s3['table_cond_bonf_vs_addjoint_raw']}",
          f"- contingency (both F unconditional Bonferroni) x (addjoint raw): {s3['table_uncond_bonf_vs_addjoint_raw']}",
          f"- median addjoint max log10E {s3['median_addjoint_max_log10E']:+.2f}; median four max log10E {s3['median_four_max_log10E']:+.2f}",
          f"- per-triple table: results/r7_triples_stocks13_vol_lag1.csv", ""]
    with open(os.path.join(RES, "r7_billio_connectedness_summary.md"), "w") as f:
        f.write("\n".join(L))

    # ---------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    colors = {"etf16": "#4c72b0", "stocks13": "#dd8452"}
    for panel in PANELS:
        w = roll[panel]
        x = pd.to_datetime([x["end"] for x in w])
        y = [x["dgc"] for x in w]
        ax1.plot(x, y, marker="o", ms=3, lw=1.2, color=colors[panel], label=f"{panel} (ret, p=1)")
        hc = [xx for xx, ww in zip(x, w) if ww["covid"]]
        h22 = [xx for xx, ww in zip(x, w) if ww["y2022"]]
        for xx in hc:
            ax1.axvspan(xx - pd.Timedelta(days=10), xx + pd.Timedelta(days=10), color="red", alpha=0.10, lw=0)
        for xx in h22:
            ax1.axvspan(xx - pd.Timedelta(days=10), xx + pd.Timedelta(days=10), color="green", alpha=0.08, lw=0)
    ax1.set_title("Rolling 252-day DGC (raw .05; window end date)\nred: spans 2020-02..2020-05, green: spans 2022", fontsize=10)
    ax1.set_ylabel("DGC (fraction of significant ordered pairs)")
    ax1.set_xlabel("window end date")
    ax1.legend(fontsize=8)

    labels, raw_c, bon_c = [], [], []
    for panel in PANELS:
        for kind in ("ret", "vol"):
            for p in (1, 5):
                d = full[panel][f"{kind}_p{p}"]
                labels.append(f"{panel}\n{kind} p={p}")
                raw_c.append(d["raw_count"]); bon_c.append(d["bonf_count"])
    xx = np.arange(len(labels))
    ax2.bar(xx - 0.2, raw_c, 0.4, color="#8da0cb", edgecolor="black", lw=0.4, label="raw .05")
    ax2.bar(xx + 0.2, bon_c, 0.4, color="#66c2a5", edgecolor="black", lw=0.4, label="Bonferroni")
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("significant ordered pairs (edges)")
    ax2.set_title("Full-sample edge counts", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("R7: pairwise Granger connectedness (Billio-style layer)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "r7_billio_connectedness.png"), dpi=200)
    plt.close(fig)
    print(f"runtime {out['runtime_s']} s", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
