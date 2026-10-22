"""R15c: placebo calibration (circularly shifted target) for further daily/weekly/hourly
returns-and-volatility families of the R5 search, using the R5 data preparation."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_master_scan as R5
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS
from common import RES, rate_ci
from multiprocessing import Pool
np.seterr(all="ignore")
TEST = R5.TEST
SETTINGS = [("etf16", "vol", "lag1"), ("stocks13", "ret", "lag1"), ("etf16", "vol", "lag5"), ("stocks13", "ret", "lag5"),
            ("etf16", "ret", "contemp"), ("stocks13", "vol", "contemp"), ("etf16", "ret", "weekly"),
            ("stocks13", "vol", "weekly"), ("hourly12", "vol", "contemp")]
MODES = {"lag1": (1, False), "lag5": (5, False), "contemp": (1, True), "weekly": (1, False)}
NH = 150


def one(args):
    Z, cond_i, p, ct, ci, ai, bi, shift = args
    out = []
    for sh in (0, shift):
        Y = np.ascontiguousarray(np.roll(Z[:, ci], sh))
        pred, _ = run_errors(Y, Z[:, ai], Z[:, bi], cond=[Z[:, cond_i]], p=p, contemp=ct, forget=TEST["forget"],
                             ridge=TEST["ridge"], feat_clip=TEST["feat_clip"])
        s, b = bounded_score(Y, pred, MODELS, "addjoint", cap=TEST["cap"])
        E, _ = bet_process(normalise(s, b), TEST["gamma"], TEST["warmup"], start=p)
        out.append(float(E.max()))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    res = {}
    for panel, target, mode in SETTINGS:
        p, ct = MODES[mode]
        Z, cols, cond, idx = R5.build_matrix(panel, target, mode)
        cond_i = cols.index(cond)
        H = R5.hypotheses(cols, cond)
        T = Z.shape[0]
        lo = 52 if mode == "weekly" else 252
        sel = rng.choice(len(H), NH, replace=False)
        jobs = [(Z, cond_i, p, ct, cols.index(H[i][0]), cols.index(H[i][1]), cols.index(H[i][2]), int(rng.integers(lo, T - lo))) for i in sel]
        with Pool(3) as pool:
            r = np.array(pool.map(one, jobs, chunksize=5))
        real, pl = r[:, 0], r[:, 1]
        q = lambda a: [float(np.quantile(np.log10(np.maximum(a, 1e-12)), x)) for x in (0.5, 0.95, 1.0)]
        res[f"{panel}_{target}_{mode}"] = dict(real_reject=rate_ci(real >= 20), placebo_reject=rate_ci(pl >= 20),
                                               real_q50_95_100=q(real), placebo_q50_95_100=q(pl))
        print(panel, target, mode, res[f"{panel}_{target}_{mode}"]["real_reject"]["k"], res[f"{panel}_{target}_{mode}"]["placebo_reject"]["k"], flush=True)
    json.dump(res, open(os.path.join(RES, "r15c_placebo_more.json"), "w"), indent=2)
    L = ["# R15c placebo calibration, further families (150 hypotheses each)", "", "| family | real raw rejections | placebo raw rejections [95% CI] |", "|---|---|---|"]
    for k, v in res.items():
        L.append(f"| {k} | {v['real_reject']['k']}/150 | {v['placebo_reject']['k']}/150 [{v['placebo_reject']['lo']:.3f}, {v['placebo_reject']['hi']:.3f}] |")
    open(os.path.join(RES, "r15c_placebo_more_summary.md"), "w").write("\n".join(L) + "\n")
