"""R16: time-uniform upper confidence bound on the running AVERAGE conditional predictive
synergy for every hypothesis of the daily return / volatility searches (R5 settings), using the
same data preparation as R5.  For each hypothesis store S = sum_s s_s, V = sum_s b_s^2, n so that the
bound for any multiplicity m can be formed afterwards:
    upper(m) = min_lambda (S + lambda V/2 + log(K m/alpha)/lambda) / n ,  lambda on a fixed grid of K points."""
import os, sys, json, csv, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_master_scan as R5
from ante_v2 import run_errors, bounded_score, MODELS, LAMBDA_GRID
from common import RES
from multiprocessing import Pool
np.seterr(all="ignore")
TEST = R5.TEST
START = 50


def one(args):
    Z, cond_i, p, contemp, ci, ai, bi = args
    Y, A, B, Cd = Z[:, ci], Z[:, ai], Z[:, bi], Z[:, cond_i]
    pred, _ = run_errors(Y, A, B, cond=[Cd], p=p, contemp=contemp, forget=TEST["forget"],
                         ridge=TEST["ridge"], feat_clip=TEST["feat_clip"])
    s, b = bounded_score(Y, pred, MODELS, "addjoint", cap=TEST["cap"])
    st = max(p, START)
    return float(s[st:].sum()), float((b[st:] ** 2).sum()), int(len(s) - st)


def upper(S, V, n, m, alpha=0.05):
    L = np.log(len(LAMBDA_GRID) * m / alpha)
    return np.min([(S + lam * V / 2 + L / lam) / n for lam in LAMBDA_GRID], axis=0)


if __name__ == "__main__":
    settings = []
    for panel in ("etf16", "stocks13", "world6"):
        for target in ("ret", "vol"):
            for mode, p, ct, wk in (("lag1", 1, False, False), ("lag5", 5, False, False),
                                    ("contemp", 1, True, False), ("weekly", 1, False, True)):
                if mode == "weekly" and panel == "world6":
                    continue
                settings.append((panel, target, mode, p, ct, wk))
    out = {}
    for panel, target, mode, p, ct, wk in settings:
        name = f"r16_bound_{panel}_{target}_{mode}"
        f = os.path.join(RES, name + ".csv")
        if os.path.exists(f):
            continue
        t0 = time.time()
        Z, cols, cond, idx = R5.build_matrix(panel, target, mode)
        cond_i = cols.index(cond)
        H = R5.hypotheses(cols, cond)
        jobs = [(Z, cond_i, p, ct, cols.index(C), cols.index(A), cols.index(B)) for C, A, B in H]
        with Pool(3) as pool:
            res = pool.map(one, jobs, chunksize=8)
        df = pd.DataFrame(H, columns=["C", "A", "B"])
        df["S"], df["V"], df["n"] = zip(*res)
        df.to_csv(f, index=False)
        print(name, len(df), f"{time.time() - t0:.0f}s", flush=True)
    # summary
    summ = []
    tot_m = 0
    frames = {}
    for panel, target, mode, p, ct, wk in settings:
        name = f"r16_bound_{panel}_{target}_{mode}"
        df = pd.read_csv(os.path.join(RES, name + ".csv"))
        frames[name] = df
        tot_m += len(df)
    for name, df in frames.items():
        m = len(df)
        mean = df.S / df.n
        row = dict(setting=name.replace("r16_bound_", ""), m=m,
                   median_mean_s=float(mean.median()),
                   median_upper_m1=float(np.median(upper(df.S, df.V, df.n, 1))),
                   median_upper_famm=float(np.median(upper(df.S, df.V, df.n, m))),
                   max_upper_famm=float(np.max(upper(df.S, df.V, df.n, m))),
                   max_upper_globalm=float(np.max(upper(df.S, df.V, df.n, tot_m))))
        summ.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(RES, "r16_average_bounds_summary.csv"), index=False)
    json.dump(dict(total_m=tot_m, rows=summ), open(os.path.join(RES, "r16_average_bounds.json"), "w"), indent=2)
    print(pd.DataFrame(summ).to_string())
