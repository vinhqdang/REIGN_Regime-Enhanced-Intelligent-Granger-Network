#!/usr/bin/env python3
"""R5: unified, transparent financial master scan with the v2 predictive-synergy test.

Run from the repository directory:
    python3 experiments/revision/r5_master_scan.py --time-one   # timing probe, writes r5_timing_probe.json
    python3 experiments/revision/r5_master_scan.py              # full scan (resumable)

Hypothesis = (panel, target kind, setting mode, C, {A,B}) where C is the target asset,
{A,B} an unordered pair of distinct assets, both different from C and from the conditioning
asset of the panel (the conditioner is never a target or a source).  The conditioner enters
as one lagged (or contemporaneous) covariate, transformed with the same target kind as C.

Per hypothesis one forgetting-RLS pass (src/ante_v2.run_errors) gives the predictions; the
'addjoint' and 'four' bounded scores are derived from that single pass.  Exclusion bounds
(reverse-hypothesis e-process) are recorded for the addjoint score only.

Outputs (experiments/revision/results/):
    r5_search_space.json                  formal search space and test settings
    r5_scan_<panel>_<target>_<mode>.csv   one row per hypothesis (written incrementally)
    r5_summary.json                       per-setting and GLOBAL summaries, dedup report, top-20
    r5_master_scan_summary.md             compact markdown tables
"""
import os
import sys
import csv
import json
import time
import argparse
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from common import load_panel, target_series, weekly, rate_ci, RES  # noqa: E402
from ante_v2 import (run_errors, bounded_score, normalise, bet_process,  # noqa: E402
                     exclusion_bounds, predictable_standardise, MODELS)

np.seterr(all="ignore")

# ---------------------------------------------------------------------------
# Formal search space
# ---------------------------------------------------------------------------
PANELS = {
    # name: (data file, conditioning asset, sampling frequency)
    "etf16":    ("returns_panel.csv", "SPY", "daily"),
    "stocks13": ("famous_returns.csv", "SPY", "daily"),
    "world6":   ("world_indices_returns.csv", "US_SP500", "daily"),
    "hourly12": ("famous_hourly_returns.csv", "SPY", "hourly"),
    "crypto10": ("crypto_hourly_returns.csv", "BTC-USD", "hourly"),
}
ALL_PANELS = list(PANELS)
TARGETS = ["ret", "vol", "tail"]

# mode -> settings.  lag5 only on daily panels; weekly only on etf16/stocks13 with ret/vol.
SETTINGS = [
    {"mode": "lag1", "p": 1, "contemp": False, "weekly": False,
     "panels": ALL_PANELS, "targets": TARGETS,
     "label": "lagged, p=1"},
    {"mode": "lag5", "p": 5, "contemp": False, "weekly": False,
     "panels": ["etf16", "stocks13", "world6"], "targets": TARGETS,
     "label": "lagged, p=5 (daily panels only)"},
    {"mode": "contemp", "p": 1, "contemp": True, "weekly": False,
     "panels": ALL_PANELS, "targets": TARGETS,
     "label": "contemporaneous interaction (A_t,B_t with Y_t; NOT Granger)"},
    {"mode": "weekly", "p": 1, "contemp": False, "weekly": True,
     "panels": ["etf16", "stocks13"], "targets": ["ret", "vol"],
     "label": "non-overlapping 5-day sums, then lag1 (daily panels etf16, stocks13)"},
]

TEST = dict(
    estimator="v2 (ante_v2)", kinds=["addjoint", "four"],
    forget=0.995, ridge=1e-2, feat_clip=5.0, std_warmup=30, cap=3.0,
    gamma=0.9, warmup=50, alpha=0.05, reject_threshold=20.0,
    tail_q=0.90, tail_min=100,
    exclusion_deltas=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    exclusion_threshold_per_hypothesis=20.0,
    fdr_levels=[0.1, 0.05],
    seed_subsample=0, subsample_cap=400, time_budget_min=90,
)
DELTAS = np.array(TEST["exclusion_deltas"])
KINDS = TEST["kinds"]
ALPHA = TEST["alpha"]


def hypothesis_def():
    return ("For a panel with N columns and conditioner c: assets = columns minus c (N' = N-1); "
            "hypotheses = { (C, {A,B}) : C in assets, {A,B} an unordered pair of distinct assets "
            "in assets minus {C} }, so the count is N' * C(N'-1, 2).")


def assets_of(cols, cond):
    return [c for c in cols if c != cond]


def hypotheses(cols, cond):
    """Ordered deterministic list of (C, A, B) names."""
    assets = assets_of(cols, cond)
    H = []
    for C in assets:
        others = [a for a in assets if a != C]
        for A, B in combinations(others, 2):
            H.append((C, A, B))
    return H


def build_matrix(panel, target, mode):
    """Target construction -> common index (dropna) -> predictable standardisation."""
    r, cond = load_panel(panel)
    if mode == "weekly":
        r = weekly(r)
    X = target_series(r, target, tail_q=TEST["tail_q"], tail_min=TEST["tail_min"])
    X = X.dropna()  # common index shared by all columns of the panel
    cols = list(X.columns)
    Z = np.column_stack([predictable_standardise(X[c].values, warmup=TEST["std_warmup"])
                         for c in cols])
    return np.ascontiguousarray(Z), cols, cond, X.index


def enumerate_settings():
    out = []
    for st in SETTINGS:
        for panel in st["panels"]:
            for target in st["targets"]:
                out.append((panel, target, st))
    return out


def setting_name(panel, target, mode):
    return f"r5_scan_{panel}_{target}_{mode}"


def search_space_dict():
    """Formal search space with per-setting counts computed from the panel column counts."""
    ncols = {}
    for p in ALL_PANELS:
        r, cond = load_panel(p)
        ncols[p] = (list(r.columns), cond)
    settings = []
    total = 0
    for panel, target, st in enumerate_settings():
        cols, cond = ncols[panel]
        Np = len(assets_of(cols, cond))
        n_hyp = Np * ((Np - 1) * (Np - 2) // 2)
        total += n_hyp
        settings.append(dict(panel=panel, target=target, mode=st["mode"], p=st["p"],
                             contemp=st["contemp"], weekly=st["weekly"],
                             conditioner=cond, n_assets_Nprime=Np, n_hypotheses=n_hyp,
                             file=PANELS[panel][0], frequency=PANELS[panel][2],
                             csv=setting_name(panel, target, st["mode"]) + ".csv",
                             label=st["label"]))
    return dict(
        description="R5 unified financial master scan, v2 test (addjoint and four from one run_errors call).",
        panels={p: dict(file=PANELS[p][0], conditioner=PANELS[p][1], frequency=PANELS[p][2],
                        columns=ncols[p][0]) for p in ALL_PANELS},
        targets=dict(ret="r_t", vol="|r_t|", tail="1{|r_t| > expanding q90 of |r|_{<t}} (target_series)"),
        modes=SETTINGS,
        hypothesis=hypothesis_def(),
        test=TEST,
        settings=settings,
        total_hypotheses=total,
    )


# ---------------------------------------------------------------------------
# Per-hypothesis evaluation
# ---------------------------------------------------------------------------
def run_one(Z, cond_i, p, contemp, ci, ai, bi):
    Y = np.ascontiguousarray(Z[:, ci])
    A = np.ascontiguousarray(Z[:, ai])
    B = np.ascontiguousarray(Z[:, bi])
    Cd = np.ascontiguousarray(Z[:, cond_i])
    pred, _ = run_errors(Y, A, B, cond=[Cd], p=p, contemp=contemp, forget=TEST["forget"],
                         ridge=TEST["ridge"], feat_clip=TEST["feat_clip"])
    row = {"T": int(len(Y))}
    gamma, warm, alpha = TEST["gamma"], TEST["warmup"], TEST["alpha"]
    for kind in KINDS:
        s, b = bounded_score(Y, pred, MODELS, kind, cap=TEST["cap"])
        u = normalise(s, b)
        E, lam = bet_process(u, gamma=gamma, warmup=warm, start=p)
        rej = np.where(E >= 1.0 / alpha)[0]
        row[f"{kind}_final_log10E"] = float(np.log10(E[-1]))
        row[f"{kind}_max_log10E"] = float(np.log10(E.max()))
        row[f"{kind}_argmax_t"] = int(E.argmax())
        row[f"{kind}_reject_t"] = int(rej[0]) if len(rej) else -1
        row[f"{kind}_mean_s"] = float(np.mean(s[p:]))
        row[f"{kind}_frac_lam_pos"] = float(np.mean(lam[p:] > 0))
        if kind == "addjoint":
            maxcap, smallest = exclusion_bounds(
                s, b, DELTAS, thr=TEST["exclusion_threshold_per_hypothesis"],
                gamma=gamma, warmup=warm, start=p)
            for d, mc in zip(DELTAS, maxcap):
                row[f"excl_maxcap_{d}"] = float(mc)
            row["excl_smallest_delta_thr20"] = float(smallest)
    return row


_W = {}


def _init(Z, cond_i, p, contemp):
    _W.update(Z=Z, cond_i=cond_i, p=p, contemp=contemp)


def _task(idx):
    ci, ai, bi = idx
    return run_one(_W["Z"], _W["cond_i"], _W["p"], _W["contemp"], ci, ai, bi)


CSV_FIELDS = (["panel", "setting", "target", "C", "A", "B", "T"]
              + [f"{k}_{m}" for k in KINDS
                 for m in ["final_log10E", "max_log10E", "argmax_t", "reject_t", "mean_s", "frac_lam_pos"]]
              + [f"excl_maxcap_{d}" for d in DELTAS]
              + ["excl_smallest_delta_thr20"])


def _read_partial(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header != CSV_FIELDS:
            return rows
        for rec in rd:
            if len(rec) == len(CSV_FIELDS):
                rows.append(dict(zip(CSV_FIELDS, rec)))
    return rows


def timing_probe(procs):
    """Time one hypothesis per setting (single process) and extrapolate the full scan."""
    probe = []
    total_s = 0.0
    for panel, target, st in enumerate_settings():
        Z, cols, cond, _ = build_matrix(panel, target, st["mode"])
        H = hypotheses(cols, cond)
        C, A, B = H[0]
        t0 = time.time()
        _init(Z, cols.index(cond), st["p"], st["contemp"])
        _task((cols.index(C), cols.index(A), cols.index(B)))
        dt = time.time() - t0
        n = len(H)
        total_s += n * dt
        probe.append(dict(panel=panel, target=target, mode=st["mode"], T=int(Z.shape[0]),
                          n_hypotheses=n, sec_per_hypothesis=dt))
        print(f"probe {panel:9s} {target:4s} {st['mode']:8s} T={Z.shape[0]:6d} "
              f"n={n:5d} {dt:6.2f}s/hyp", flush=True)
    est_min = total_s / procs / 60.0 * 1.10  # 10% overhead for pool / IO
    out = dict(procs=procs, settings=probe, serial_estimate_min=total_s / 60.0,
               parallel_estimate_min=est_min, budget_min=TEST["time_budget_min"],
               subsample_required=bool(est_min > TEST["time_budget_min"]))
    with open(os.path.join(RES, "r5_timing_probe.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"serial estimate {total_s/60:.1f} min; with {procs} processes ~{est_min:.1f} min "
          f"(budget {TEST['time_budget_min']} min)", flush=True)
    return out


def run_scan(procs, subsample_flag):
    ss = search_space_dict()
    probe_path = os.path.join(RES, "r5_timing_probe.json")
    subsample = bool(subsample_flag)
    if os.path.exists(probe_path):
        with open(probe_path) as fh:
            probe = json.load(fh)
        if probe["subsample_required"]:
            subsample = True
    ss["execution"] = dict(procs=procs, subsample_applied=subsample,
                           subsample_rule=("deterministic, seed 0, settings with more than "
                                           f"{TEST['subsample_cap']} hypotheses reduced to "
                                           f"{TEST['subsample_cap']}; applied only if the timing "
                                           f"estimate exceeds {TEST['time_budget_min']} min"))
    with open(os.path.join(RES, "r5_search_space.json"), "w") as fh:
        json.dump(ss, fh, indent=2, default=str)

    t_start = time.time()
    n_done_total = 0
    for panel, target, st in enumerate_settings():
        name = setting_name(panel, target, st["mode"])
        final = os.path.join(RES, name + ".csv")
        part = final + ".partial"
        if os.path.exists(final):
            print(f"skip (complete) {name}", flush=True)
            continue
        Z, cols, cond, _ = build_matrix(panel, target, st["mode"])
        H = hypotheses(cols, cond)
        n_full = len(H)
        if subsample and n_full > TEST["subsample_cap"]:
            rng = np.random.default_rng(TEST["seed_subsample"])
            sel = np.sort(rng.choice(n_full, TEST["subsample_cap"], replace=False))
            H = [H[i] for i in sel]
        done_rows = _read_partial(part)
        k = len(done_rows)
        idx_list = [(cols.index(C), cols.index(A), cols.index(B)) for C, A, B in H]
        mode_p, mode_c = st["p"], st["contemp"]
        print(f"{name}: T={Z.shape[0]} hypotheses={len(H)} (full {n_full}) resume_from={k}",
              flush=True)
        t0 = time.time()
        fh = open(part, "w" if k == 0 else "a", newline="")
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if k == 0:
            wr.writeheader()
        else:
            # rewrite cleanly (drops any truncated trailing line from an interrupted run)
            fh.close()
            with open(part, "w", newline="") as f2:
                w2 = csv.DictWriter(f2, fieldnames=CSV_FIELDS)
                w2.writeheader()
                for rec in done_rows:
                    w2.writerow(rec)
            fh = open(part, "a", newline="")
            wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        todo = idx_list[k:]
        names = H[k:]
        with Pool(procs, initializer=_init,
                  initargs=(Z, cols.index(cond), mode_p, mode_c)) as pool:
            for j, row in enumerate(pool.imap(_task, todo, chunksize=2)):
                C, A, B = names[j]
                rec = {"panel": panel, "setting": st["mode"], "target": target,
                       "C": C, "A": A, "B": B}
                rec.update(row)
                wr.writerow({f: rec.get(f, "") for f in CSV_FIELDS})
                fh.flush()
                n_done_total += 1
                if (j + 1) % 50 == 0 or j + 1 == len(todo):
                    el = time.time() - t0
                    rate = (j + 1) / max(el, 1e-6)
                    eta = (len(todo) - j - 1) / rate
                    print(f"  {name}: {k + j + 1}/{len(H)}  elapsed {el/60:.1f} min  "
                          f"ETA {eta/60:.1f} min", flush=True)
        fh.close()
        os.replace(part, final)
        print(f"done {name} in {(time.time()-t0)/60:.2f} min", flush=True)
    print(f"scan finished in {(time.time()-t_start)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def ebh_count(log10e, m, q):
    lg = np.sort(np.asarray(log10e, float))[::-1]
    n = len(lg)
    if n == 0:
        return 0
    ks = np.arange(1, n + 1)
    ok = lg >= (np.log10(m) - np.log10(q) - np.log10(ks))
    return int(ks[ok].max()) if ok.any() else 0


def group_stats(df, m_full, n_full):
    alpha = ALPHA
    thr_fw = np.log10(m_full / alpha)
    out = dict(n_evaluated=int(len(df)), n_full=int(n_full), m_family=int(m_full),
               sampling_fraction=float(len(df) / n_full) if n_full else np.nan,
               subsampled=bool(len(df) < n_full),
               family_threshold_log10=float(thr_fw))
    for kind in KINDS:
        fl = df[f"{kind}_final_log10E"].astype(float).values
        mx = df[f"{kind}_max_log10E"].astype(float).values
        rj = df[f"{kind}_reject_t"].astype(int).values
        raw = int((rj >= 0).sum())
        fw = int((mx >= thr_fw).sum())
        rr = rate_ci(rj >= 0)
        o = dict(raw_rejections=raw, raw_rate=rr["rate"], raw_rate_ci=[rr["lo"], rr["hi"]],
                 familywise_rejections=fw,
                 ebh_final_fdr0_1=ebh_count(fl, m_full, 0.1),
                 ebh_final_fdr0_05=ebh_count(fl, m_full, 0.05),
                 ebh_max_fdr0_1_NOT_FORMALLY_VALID=ebh_count(mx, m_full, 0.1),
                 ebh_max_fdr0_05_NOT_FORMALLY_VALID=ebh_count(mx, m_full, 0.05),
                 final_log10E=dict(median=float(np.median(fl)), q25=float(np.percentile(fl, 25)),
                                   q75=float(np.percentile(fl, 75)), max=float(fl.max())),
                 max_log10E=dict(median=float(np.median(mx)), q25=float(np.percentile(mx, 25)),
                                 q75=float(np.percentile(mx, 75)), max=float(mx.max())),
                 mean_s=float(df[f"{kind}_mean_s"].astype(float).mean()),
                 median_frac_lam_pos=float(np.median(df[f"{kind}_frac_lam_pos"].astype(float))))
        out[kind] = o
    # exclusion bounds (addjoint only)
    smallest = df["excl_smallest_delta_thr20"].astype(float).values
    fin = np.isfinite(smallest)
    mc = df[[f"excl_maxcap_{d}" for d in DELTAS]].astype(float).values
    fam_thr = m_full / alpha
    fam_idx = np.array([np.argmax(row >= fam_thr) if (row >= fam_thr).any() else -1 for row in mc])
    fam_delta = np.where(fam_idx >= 0, DELTAS[np.clip(fam_idx, 0, None)], np.nan)
    out["exclusion_addjoint"] = dict(
        per_hypothesis_threshold=TEST["exclusion_threshold_per_hypothesis"],
        n_excluded_any_delta=int(fin.sum()),
        median_smallest_excluded_delta=float(np.nanmedian(smallest)) if fin.any() else None,
        n_excluded_familywise=int(np.isfinite(fam_delta).sum()),
        median_smallest_excluded_delta_familywise=(float(np.nanmedian(fam_delta))
                                                  if np.isfinite(fam_delta).any() else None),
        familywise_capital_threshold=float(fam_thr))
    return out


def dedup_report(ss_settings):
    """Hypotheses whose (A,B,C) are all shared across two panels (same conditioner, same mode/target)."""
    cols = {p: list(load_panel(p)[0].columns) for p in ALL_PANELS}
    rep = []
    for i, p in enumerate(ALL_PANELS):
        for q in ALL_PANELS[i + 1:]:
            shared = sorted(set(cols[p]) & set(cols[q]))
            if not shared:
                continue
            per_set = []
            total = 0
            for st in SETTINGS:
                if p not in st["panels"] or q not in st["panels"]:
                    continue
                for target in st["targets"]:
                    Hp = hypotheses(cols[p], PANELS[p][1])
                    n_shared = sum(1 for C, A, B in Hp if {C, A, B} <= set(shared))
                    per_set.append(dict(mode=st["mode"], target=target, n_shared_tuples_in_each_panel=n_shared))
                    total += n_shared
            rep.append(dict(panel_a=p, panel_b=q, shared_columns=shared,
                            frequency_a=PANELS[p][2], frequency_b=PANELS[q][2],
                            conditioner_a=PANELS[p][1], conditioner_b=PANELS[q][1],
                            tuples_per_panel_with_all_shared_assets_summed_over_common_settings=total,
                            per_setting=per_set))
    # empirical check that the shared series coincide on common dates (daily pair)
    data_check = []
    base = {}
    for p in ["etf16", "stocks13"]:
        base[p] = load_panel(p)[0]
    common_cols = sorted(set(base["etf16"].columns) & set(base["stocks13"].columns))
    idx = base["etf16"].index.intersection(base["stocks13"].index)
    for c in common_cols:
        d = np.abs(base["etf16"].loc[idx, c].values - base["stocks13"].loc[idx, c].values)
        data_check.append(dict(column=c, n_common_dates=int(len(idx)), max_abs_diff=float(np.nanmax(d))))
    return dict(pairs=rep, etf16_vs_stocks13_raw_return_identity=data_check)


def top20(all_df):
    recs = []
    for kind in KINDS:
        sub = all_df[["panel", "setting", "target", "C", "A", "B", f"{kind}_max_log10E",
                      f"{kind}_final_log10E", f"{kind}_reject_t"]].copy()
        sub.columns = ["panel", "setting", "target", "C", "A", "B", "max_log10E", "final_log10E", "reject_t"]
        sub["kind"] = kind
        recs.append(sub)
    d = pd.concat(recs).sort_values("max_log10E", ascending=False).head(20)
    return [{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer,)) else v))
             for k, v in r.items()} for r in d.to_dict(orient="records")]


def summarise():
    ss = search_space_dict()
    frames = []
    per_group = {}
    missing = []
    for s in ss["settings"]:
        path = os.path.join(RES, s["csv"])
        if not os.path.exists(path):
            missing.append(s["csv"])
            continue
        df = pd.read_csv(path)
        frames.append(df)
        key = f"{s['panel']}|{s['target']}|{s['mode']}"
        per_group[key] = group_stats(df, m_full=s["n_hypotheses"], n_full=s["n_hypotheses"])
        per_group[key].update(panel=s["panel"], target=s["target"], mode=s["mode"],
                              conditioner=s["conditioner"], n_hypotheses_full=s["n_hypotheses"])
    all_df = pd.concat(frames, ignore_index=True)
    m_total = sum(s["n_hypotheses"] for s in ss["settings"])
    glob = group_stats(all_df, m_full=m_total, n_full=m_total)
    glob.update(scope="GLOBAL", n_settings=len(ss["settings"]), m_total=m_total)
    # subsampled settings: fraction on the row
    summary = dict(
        missing_settings=missing,
        execution=ss.get("execution"),
        global_hypothesis_count=m_total,
        global_row=glob,
        settings=per_group,
        dedup=dedup_report(ss["settings"]),
        top20_max_log10E=top20(all_df),
        notes=dict(
            max_E_ebh_not_valid=("e-BH on max_E is reported as requested but is not formally valid: "
                                 "the maximum of a test martingale is not an e-value (E[max E] can exceed 1). "
                                 "Final E is a valid e-value."),
            subsampled_definition=("n_full = all hypotheses of the setting; family thresholds use n_full "
                                   "(conservative for subsampled settings)."),
        ),
    )
    with open(os.path.join(RES, "r5_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    write_markdown(summary, per_group, glob)
    return summary


def write_markdown(summary, per_group, glob):
    L = []
    L.append("# R5 master scan summary\n")
    ex = summary.get("execution") or {}
    L.append(f"Global hypotheses m = {summary['global_hypothesis_count']}; "
             f"family threshold log10(m/alpha) = {glob['family_threshold_log10']:.3f}; "
             f"subsampling applied: {ex.get('subsample_applied')}; "
             f"evaluated rows = {glob['n_evaluated']}\n")
    L.append("## Per setting (addjoint score; four in the second table)\n")
    L.append("| panel | target | mode | m | n_eval | raw rej | fw rej | e-BH final 0.1/0.05 | median final log10E | median max log10E | max max log10E | median smallest excl. delta (thr 20) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, g in per_group.items():
        a = g["addjoint"]
        ex_a = g["exclusion_addjoint"]
        L.append(f"| {g['panel']} | {g['target']} | {g['mode']} | {g['m_family']} | {g['n_evaluated']} | "
                 f"{a['raw_rejections']} | {a['familywise_rejections']} | "
                 f"{a['ebh_final_fdr0_1']}/{a['ebh_final_fdr0_05']} | "
                 f"{a['final_log10E']['median']:.3f} | {a['max_log10E']['median']:.3f} | "
                 f"{a['max_log10E']['max']:.3f} | "
                 f"{ex_a['median_smallest_excluded_delta'] if ex_a['median_smallest_excluded_delta'] is not None else 'n/a'} |")
    L.append("\n## Per setting (four score)\n")
    L.append("| panel | target | mode | raw rej | fw rej | e-BH final 0.1/0.05 | median final log10E | median max log10E | max max log10E |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, g in per_group.items():
        f = g["four"]
        L.append(f"| {g['panel']} | {g['target']} | {g['mode']} | {f['raw_rejections']} | "
                 f"{f['familywise_rejections']} | {f['ebh_final_fdr0_1']}/{f['ebh_final_fdr0_05']} | "
                 f"{f['final_log10E']['median']:.3f} | {f['max_log10E']['median']:.3f} | {f['max_log10E']['max']:.3f} |")
    L.append("\n## GLOBAL\n")
    for kind in KINDS:
        k = glob[kind]
        L.append(f"- {kind}: raw rejections {k['raw_rejections']} ({k['raw_rate']:.4f}), family-wise "
                 f"{k['familywise_rejections']}, e-BH final FDR 0.1/0.05 = {k['ebh_final_fdr0_1']}/{k['ebh_final_fdr0_05']}, "
                 f"median final log10E {k['final_log10E']['median']:.3f}, median max log10E {k['max_log10E']['median']:.3f}, "
                 f"max max log10E {k['max_log10E']['max']:.3f}")
    L.append("\n## Top 20 by max log10E\n")
    L.append("| panel | setting | target | kind | C | A | B | max log10E | final log10E |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in summary["top20_max_log10E"]:
        L.append(f"| {r['panel']} | {r['setting']} | {r['target']} | {r['kind']} | {r['C']} | {r['A']} | {r['B']} | "
                 f"{r['max_log10E']:.3f} | {r['final_log10E']:.3f} |")
    with open(os.path.join(RES, "r5_master_scan_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true", help="timing probe only")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()
    if args.time_one:
        timing_probe(args.procs)
        return
    if not args.summarise_only:
        run_scan(args.procs, subsample_flag=False)
    summarise()
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
