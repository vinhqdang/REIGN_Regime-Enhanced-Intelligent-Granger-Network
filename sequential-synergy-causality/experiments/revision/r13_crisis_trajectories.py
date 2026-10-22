"""R13: running e-process trajectories across a whole family of hypotheses, to see whether
evidence rises during crises (2018 vol spike, 2020 COVID crash, 2022 rate-hike drawdown) and then decays,
which a final-value summary would hide.  Families: stocks13 vol lag1 (660) and etf16 ret lag1 (1365).
Reports, per calendar year and per crisis window, the maximum over hypotheses of log10 E_t and the
cross-sectional median / 95th percentile; also figure."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_master_scan as R5
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS
from common import RES
from multiprocessing import Pool
np.seterr(all="ignore")
TEST = R5.TEST
WIN = {"2018 vol spike": ("2018-01-25", "2018-04-30"), "2020 COVID": ("2020-02-15", "2020-06-30"),
       "2022 drawdown": ("2022-01-01", "2022-10-31")}


def one(args):
    Z, cond_i, ci, ai, bi = args
    Y, A, B, Cd = Z[:, ci], Z[:, ai], Z[:, bi], Z[:, cond_i]
    pred, _ = run_errors(Y, A, B, cond=[Cd], p=1, contemp=False, forget=TEST["forget"],
                         ridge=TEST["ridge"], feat_clip=TEST["feat_clip"])
    s, b = bounded_score(Y, pred, MODELS, "addjoint", cap=TEST["cap"])
    E, _ = bet_process(normalise(s, b), TEST["gamma"], TEST["warmup"], start=1)
    return np.log10(np.maximum(E, 1e-12)).astype(np.float32)


if __name__ == "__main__":
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (panel, target) in zip(axes, (("stocks13", "vol"), ("etf16", "ret"))):
        Z, cols, cond, idx = R5.build_matrix(panel, target, "lag1")
        cond_i = cols.index(cond)
        H = R5.hypotheses(cols, cond)
        jobs = [(Z, cond_i, cols.index(C), cols.index(A), cols.index(B)) for C, A, B in H]
        with Pool(3) as pool:
            L = np.array(pool.map(one, jobs, chunksize=8))
        dates = pd.DatetimeIndex(idx)
        med, p95, mx = np.median(L, 0), np.quantile(L, 0.95, 0), L.max(0)
        res = {"m": len(H), "final_median": float(med[-1]), "final_max": float(mx[-1]),
               "global_max": float(mx.max()), "global_max_date": str(dates[int(mx.argmax())].date())}
        for k, (a, b) in WIN.items():
            sel = (dates >= a) & (dates <= b)
            res[k] = dict(max_over_hyp_of_max_log10E=float(L[:, sel].max()),
                          median_of_window_max_log10E=float(np.median(L[:, sel].max(1))),
                          share_hyp_with_window_max_gt_0p3=float(np.mean(L[:, sel].max(1) > 0.3)))
        yrs = {}
        for y in sorted(set(dates.year)):
            sel = dates.year == y
            yrs[int(y)] = float(L[:, sel].max())
        res["max_log10E_by_year"] = yrs
        out[f"{panel}_{target}_lag1"] = res
        ax.plot(dates, med, label="median over hypotheses", color="#4c72b0")
        ax.plot(dates, p95, label="95th percentile", color="#dd8452")
        ax.plot(dates, mx, label="maximum", color="#c44e52", lw=0.9)
        ax.axhline(np.log10(20), color="k", ls="--", lw=0.8)
        ax.axhline(np.log10(len(H) / 0.05), color="k", ls=":", lw=0.8)
        for k, (a, b) in WIN.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.18)
        ax.set_title(f"{panel}, {target}, lag 1 ({len(H)} hypotheses)")
        ax.set_ylabel(r"$\log_{10}E_t$")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "r13_crisis_trajectories.png"), dpi=200)
    json.dump(out, open(os.path.join(RES, "r13_crisis_trajectories.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))
