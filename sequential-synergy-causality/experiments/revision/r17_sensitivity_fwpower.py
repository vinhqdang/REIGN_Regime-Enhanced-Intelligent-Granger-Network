"""R17: (a) sensitivity of the real-data results to the truncation level c and to feature winsorisation
(200 random hypotheses each from stocks13 vol lag1 and etf16 ret lag1);
(b) power of ANTE-SG on the synthetic interaction DGP at the FAMILY-WISE thresholds m/alpha
(m = 660 and 4400) and at 1/alpha, c in {0.15,0.2,0.3,0.45}, 150 replications, T=2500."""
import os, sys, json
import numpy as np
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS, predictable_standardise
from common import RES, rate_ci, load_panel, target_series, gen_var
from multiprocessing import Pool
np.seterr(all="ignore")
VARIANTS = {"base (c=3, winsor 5)": (3.0, 5.0), "c=2": (2.0, 5.0), "c=5": (5.0, 5.0),
            "no winsorisation": (3.0, None), "c=5, no winsorisation": (5.0, None)}
CACHE = {}


def prep(panel, kind):
    r, cond = load_panel(panel)
    X = target_series(r, kind)
    return {c: predictable_standardise(X[c].values, 30) for c in X.columns}, cond


def job_a(args):
    panel, kind, C, A, B = args
    if (panel, kind) not in CACHE:
        CACHE[(panel, kind)] = prep(panel, kind)
    S, cond = CACHE[(panel, kind)]
    out = {}
    for name, (cap, fc) in VARIANTS.items():
        pred, _ = run_errors(S[C], S[A], S[B], [S[cond]], 1, False, 0.995, 1e-2, fc)
        s, b = bounded_score(S[C], pred, MODELS, "addjoint", cap)
        E, _ = bet_process(normalise(s, b), 0.9, 50, start=1)
        out[name] = (float(E.max()), float(E[-1]))
    return args, out


def job_b(args):
    c, i = args
    rng = np.random.default_rng([17, int(c * 100), i])
    Y, A, B, Z = gen_var("syn", 2500, rng, rho=0.2, strength=c)
    pred, _ = run_errors(Y, A, B, [Z], 1, False, 0.995, 1e-2, None)
    s, b = bounded_score(Y, pred, MODELS, "addjoint", 3.0)
    E, _ = bet_process(normalise(s, b), 0.9, 50, start=1)
    return c, float(E.max())


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    jobs = []
    for panel, kind in (("stocks13", "vol"), ("etf16", "ret")):
        r, cond = load_panel(panel)
        assets = [a for a in r.columns if a != cond]
        H = [(C, A, B) for C in assets for A, B in combinations([a for a in assets if a != C], 2)]
        for i in rng.choice(len(H), 200, replace=False):
            jobs.append((panel, kind) + H[i])
    with Pool(2) as pool:
        resa = pool.map(job_a, jobs, chunksize=5)
        cs = [0.15, 0.2, 0.3, 0.45]
        resb = pool.map(job_b, [(c, i) for c in cs for i in range(150)], chunksize=5)
    out = {"a_sensitivity": {}, "b_fw_power": {}}
    for panel, kind in (("stocks13", "vol"), ("etf16", "ret")):
        sub = [r[1] for r in resa if r[0][0] == panel]
        row = {}
        for name in VARIANTS:
            mx = np.array([x[name][0] for x in sub]); fin = np.array([x[name][1] for x in sub])
            row[name] = dict(raw_reject=rate_ci(mx >= 20), median_final_log10E=float(np.median(np.log10(np.maximum(fin, 1e-12)))),
                             max_log10E=float(np.log10(mx.max())), median_max_log10E=float(np.median(np.log10(mx))))
        out["a_sensitivity"][f"{panel}_{kind}"] = row
    for c in cs:
        mx = np.array([r[1] for r in resb if r[0] == c])
        out["b_fw_power"][str(c)] = dict(p_1_over_alpha=rate_ci(mx >= 20), p_m660=rate_ci(mx >= 660 / 0.05), p_m4400=rate_ci(mx >= 4400 / 0.05))
    json.dump(out, open(os.path.join(RES, "r17_sensitivity_fwpower.json"), "w"), indent=2)
    L = ["# R17 summary", "", "## (a) sensitivity to truncation level and winsorisation (200 hypotheses each)"]
    for k, row in out["a_sensitivity"].items():
        L += [f"### {k}", "| variant | raw rejections | median final log10E | max log10E |", "|---|---|---|---|"]
        for n, v in row.items():
            L.append(f"| {n} | {v['raw_reject']['k']}/{v['raw_reject']['n']} | {v['median_final_log10E']:.2f} | {v['max_log10E']:.2f} |")
    L += ["", "## (b) synthetic power at family-wise thresholds (150 reps)", "| c | 1/alpha | m=660 | m=4400 |", "|---|---|---|---|"]
    for c, v in out["b_fw_power"].items():
        L.append(f"| {c} | {v['p_1_over_alpha']['rate']:.2f} [{v['p_1_over_alpha']['lo']:.2f},{v['p_1_over_alpha']['hi']:.2f}] | "
                 f"{v['p_m660']['rate']:.2f} [{v['p_m660']['lo']:.2f},{v['p_m660']['hi']:.2f}] | {v['p_m4400']['rate']:.2f} [{v['p_m4400']['lo']:.2f},{v['p_m4400']['hi']:.2f}] |")
    open(os.path.join(RES, "r17_sensitivity_fwpower_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
