"""R14: spurious-regression stress test.  Y, A, B are mutually independent AR(1)
processes with coefficient phi (no cross dependence at all => every null holds).
Reports Type-I of the v2 test (addjoint, truncated loss) and the legacy clipped test
as persistence grows toward a unit root, T=2500, n_rep=100 per phi.
Also reports sample ACF of real return / volatility series for comparison."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS, predictable_standardise
from synergistic_granger import ante_sg
from common import RES, rate_ci, load_panel, target_series
from multiprocessing import Pool
np.seterr(all="ignore")


def ar(phi, T, rng):
    e = rng.standard_normal(T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def one(args):
    phi, i = args
    rng = np.random.default_rng([14, int(phi * 1000), i])
    T = 2500
    Y, A, B = ar(phi, T, rng), ar(phi, T, rng), ar(phi, T, rng)
    Ys, As, Bs = [predictable_standardise(v, 30) for v in (Y, A, B)]
    pred, _ = run_errors(Ys, As, Bs, None, 1, False, 0.995, 1e-2, 5.0)
    s, b = bounded_score(Ys, pred, MODELS, "addjoint", 3.0)
    E, _ = bet_process(normalise(s, b), 0.9, 50, start=1)
    leg = ante_sg(Y, A, B, p=1, alpha=0.05)
    return phi, float(E.max()), float(E[-1]), bool(leg["rejected"]), float(leg["final_E"])


if __name__ == "__main__":
    phis = [0.5, 0.9, 0.97, 0.99, 0.999]
    jobs = [(p, i) for p in phis for i in range(100)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=4)
    out = {"setting": "independent AR(1) Y,A,B (all nulls true); T=2500; n_rep=100", "rows": []}
    for p in phis:
        r = [x for x in res if x[0] == p]
        out["rows"].append(dict(phi=p,
                                v2=rate_ci([x[1] >= 20 for x in r]),
                                legacy=rate_ci([x[3] for x in r]),
                                v2_median_log10_max=float(np.median(np.log10([x[1] for x in r]))),
                                v2_median_log10_final=float(np.median(np.log10([max(x[2], 1e-12) for x in r])))))
    # real-data persistence
    acf = {}
    for name, panel in (("stocks13", "stocks13"), ("etf16", "etf16")):
        r, cond = load_panel(panel)
        for kind in ("ret", "vol"):
            X = target_series(r, kind)
            a1 = [np.corrcoef(X[c].values[1:], X[c].values[:-1])[0, 1] for c in X.columns]
            acf[f"{name}_{kind}"] = dict(median_acf1=float(np.median(a1)), min=float(np.min(a1)), max=float(np.max(a1)))
    out["real_acf1"] = acf
    json.dump(out, open(os.path.join(RES, "r14_persistence_stress.json"), "w"), indent=2)
    lines = ["# R14 summary: independent AR(1) series (spurious-regression stress)", "",
             "| phi | v2 Type-I (max E>=20) | legacy Type-I |", "|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['phi']} | {r['v2']['k']}/{r['v2']['n']} = {r['v2']['rate']:.2f} [{r['v2']['lo']:.2f}, {r['v2']['hi']:.2f}] "
                     f"| {r['legacy']['k']}/{r['legacy']['n']} = {r['legacy']['rate']:.2f} [{r['legacy']['lo']:.2f}, {r['legacy']['hi']:.2f}] |")
    lines += ["", "Real-data lag-1 autocorrelation:", ""] + [f"- {k}: median {v['median_acf1']:.3f} (range {v['min']:.3f} to {v['max']:.3f})" for k, v in acf.items()]
    open(os.path.join(RES, "r14_persistence_stress_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
