"""R15: permutation-style null calibration on REAL data.  For a random sample of
hypotheses from a scan setting, the target series C is circularly shifted by a random
amount (>= 252 observations) relative to (A, B, conditioner): this keeps C's marginal law and
autocorrelation but destroys all cross-dependence, so every null holds.  Compare the
distribution of max / final log10 E under the placebo with that of the real series.
Settings: stocks13 vol lag1; etf16 ret lag1; hourly12 vol lag1."""
import os, sys, json
import numpy as np
import pandas as pd
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS, predictable_standardise
from common import RES, rate_ci, load_panel, target_series
from multiprocessing import Pool
np.seterr(all="ignore")
SETTINGS = [("stocks13", "vol"), ("etf16", "ret"), ("hourly12", "vol")]
NHYP = 250
CACHE = {}


def prep(panel, kind):
    r, cond = load_panel(panel)
    X = target_series(r, kind)
    S = {c: predictable_standardise(X[c].values, 30) for c in X.columns}
    return S, cond


def run(S, cond, C, A, B, shift):
    Y = np.roll(S[C], shift) if shift else S[C]
    pred, _ = run_errors(Y, S[A], S[B], [S[cond]], 1, False, 0.995, 1e-2, 5.0)
    s, b = bounded_score(Y, pred, MODELS, "addjoint", 3.0)
    E, _ = bet_process(normalise(s, b), 0.9, 50, start=1)
    return float(E.max()), float(E[-1])


def job(args):
    panel, kind, C, A, B, shift = args
    key = (panel, kind)
    if key not in CACHE:
        CACHE[key] = prep(panel, kind)
    S, cond = CACHE[key]
    return args, run(S, cond, C, A, B, 0), run(S, cond, C, A, B, shift)


if __name__ == "__main__":
    rng = np.random.default_rng(15)
    jobs = []
    for panel, kind in SETTINGS:
        r, cond = load_panel(panel)
        assets = [c for c in r.columns if c != cond]
        hyps = [(C, A, B) for C in assets for A, B in combinations([a for a in assets if a != C], 2)]
        idx = rng.choice(len(hyps), NHYP, replace=False)
        T = len(r)
        for i in idx:
            C, A, B = hyps[i]
            jobs.append((panel, kind, C, A, B, int(rng.integers(252, T - 252))))
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=5)
    out = {"nhyp_per_setting": NHYP, "settings": {}}
    thr_raw = np.log10(20)
    for panel, kind in SETTINGS:
        rows = [(r[1], r[2]) for r in res if r[0][0] == panel and r[0][1] == kind]
        real_max = np.array([x[0][0] for x in rows]); real_fin = np.array([x[0][1] for x in rows])
        pl_max = np.array([x[1][0] for x in rows]); pl_fin = np.array([x[1][1] for x in rows])
        q = lambda a: [float(np.quantile(np.log10(np.maximum(a, 1e-12)), p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)]
        out["settings"][f"{panel}_{kind}"] = dict(
            real_log10max_quantiles=q(real_max), placebo_log10max_quantiles=q(pl_max),
            real_log10final_quantiles=q(real_fin), placebo_log10final_quantiles=q(pl_fin),
            real_reject_raw=rate_ci(real_max >= 20), placebo_reject_raw=rate_ci(pl_max >= 20),
            placebo_reject_mOverAlpha_660=rate_ci(pl_max >= 660 / 0.05))
    json.dump(out, open(os.path.join(RES, "r15_placebo_calibration.json"), "w"), indent=2)
    lines = ["# R15 summary: real vs circular-shift placebo (all nulls true under placebo)", "",
             "quantiles (5,25,50,75,95,100%) of log10 of running-max E; raw rejection = max E >= 20", ""]
    for k, v in out["settings"].items():
        lines += [f"## {k}",
                  f"- real   log10 max E quantiles: {np.round(v['real_log10max_quantiles'], 2).tolist()}; raw rejection {v['real_reject_raw']['k']}/{v['real_reject_raw']['n']}",
                  f"- placebo log10 max E quantiles: {np.round(v['placebo_log10max_quantiles'], 2).tolist()}; raw rejection {v['placebo_reject_raw']['k']}/{v['placebo_reject_raw']['n']} [{v['placebo_reject_raw']['lo']:.3f}, {v['placebo_reject_raw']['hi']:.3f}]",
                  f"- real   log10 final E quantiles: {np.round(v['real_log10final_quantiles'], 2).tolist()}",
                  f"- placebo log10 final E quantiles: {np.round(v['placebo_log10final_quantiles'], 2).tolist()}", ""]
    open(os.path.join(RES, "r15_placebo_calibration_summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
