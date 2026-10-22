"""R12: k-way group test (exactly valid v2) on (a) real turbulence energy cascade,
(b) systemic-risk sector groups (daily volatility, contemporaneous), (c) synthetic
pure 3-way mechanism Y = A*B*C and a 3-way null; Type-I of the group test."""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.io as sio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ante_v2 import ante_group_v2, predictable_standardise
from common import RES, DATA, rate_ci, load_panel
np.seterr(all="ignore")
SURD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "SURD", "data", "energy_cascade_signals.mat")
ps = lambda x, w=30: predictable_standardise(x, warmup=w)
out = {}

# (a) turbulence: 3 coarse scales -> finest scale (lagged), full data and subsample sizes
X = sio.loadmat(SURD)["X"]
Zt = np.array([ps(r) for r in X])
r = ante_group_v2(Zt[3], [Zt[0], Zt[1], Zt[2]], p=1, feat_clip=5.0)
out["turbulence_3way_lag1"] = dict(T=int(X.shape[1]), rejected=r["rejected"], reject_time=r["reject_time"],
                                   log10E_final=float(np.log10(max(r["final_E"], 1e-9))),
                                   log10E_max=float(np.log10(max(r["max_E"], 1e-9))), mean_s=r["mean_s"])
# placebo: finest scale replaced by a circularly shifted copy (breaks the dependence)
rng = np.random.default_rng(0)
plac = []
for i in range(20):
    sh = int(rng.integers(500, X.shape[1] - 500))
    rr = ante_group_v2(np.roll(Zt[3], sh), [Zt[0], Zt[1], Zt[2]], p=1, feat_clip=5.0)
    plac.append(rr["max_E"] >= 20)
out["turbulence_circular_shift_placebo"] = rate_ci(plac)

# (b) systemic groups (same groups as the original study)
rets, _ = load_panel("etf16")
vol = rets.abs()
mkt = ps(vol["SPY"].values)
groups = [(["XLF", "HYG"], "banks + credit"), (["XLF", "HYG", "TLT"], "banks + credit + rates"),
          (["XLF", "HYG", "USO"], "banks + credit + oil"), (["XLF", "HYG", "TLT", "USO"], "banks + credit + rates + oil"),
          (["XLF", "XLK", "XLE"], "financials + tech + energy"), (["HYG", "TLT", "UUP"], "credit + rates + dollar")]
sysr = []
for names, lbl in groups:
    for mode, ct in (("lagged", False), ("contemporaneous", True)):
        rr = ante_group_v2(mkt, [ps(vol[n].values) for n in names], p=1, contemp=ct, feat_clip=5.0)
        sysr.append(dict(group=lbl, k=len(names), mode=mode, rejected=rr["rejected"],
                         log10E_final=float(np.log10(max(rr["final_E"], 1e-9))),
                         log10E_max=float(np.log10(max(rr["max_E"], 1e-9)))))
out["systemic_groups"] = sysr

# (c) synthetic pure 3-way: Y = s*A*B*C + noise ; and additive 3-way null; n_rep=100, T=2500
def gen3(kind, T, rng, s=0.9):
    A, B, C = rng.standard_normal((3, T))
    Z = rng.standard_normal(T)
    Y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        a, b, c = A[t - 1], B[t - 1], C[t - 1]
        if kind == "null3":
            Y[t] = 0.4 * a + 0.4 * b + 0.4 * c + 0.3 * Z[t] + e[t]
        else:
            Y[t] = s * a * b * c + 0.3 * Z[t] + e[t]
    return Y, [A, B, C], Z
syn = {}
for kind, strength in (("null3", 0), ("pure3", 0.9), ("pure3", 0.5)):
    flags = []
    for i in range(100):
        rg = np.random.default_rng(1000 + i)
        Y, S, Z = gen3(kind, 2500, rg, s=strength)
        rr = ante_group_v2(Y, S, cond=[Z], p=1)
        flags.append(rr["max_E"] >= 20)
    syn[f"{kind}_{strength}"] = rate_ci(flags)
out["synthetic_3way"] = syn
json.dump(out, open(os.path.join(RES, "r12_groups_turbulence.json"), "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float)[:3000])
