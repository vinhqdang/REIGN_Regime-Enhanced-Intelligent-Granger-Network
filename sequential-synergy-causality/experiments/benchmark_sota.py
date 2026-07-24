"""
Head-to-head benchmark: ANTE vs SOTA synergy methods.

Baselines (2020-2026):
  - PEID   (Yang, Wang & Zhang 2026, arXiv:2605.03267)  -- re-implemented (no code released)
  - SURD   (Martinez-Sanchez et al., Nat. Commun. 2024) -- cloned reference code
  - PID_WB (Williams & Beer I_min)                       -- classical, re-implemented
  - InteractionInfo (Gaussian co-information)            -- classical
  - O-information (Rosas et al., via `hoi`)              -- covers the 2026 finance OIR method

Fair-comparison axes (ANTE is a sequential *test* with anytime-valid error
control; the baselines are batch *estimators/decompositions*):
  0  Canonical discrete ground truth (XOR/AND/redundant/independent).
  A  Detection AUC on continuous DGPs (synergy vs additive/own-nonlinear).
  C  Type-I under continuous monitoring (ANTE vs peeking permutation test).
  D  Real portfolio-variance ranking (cross-asset vs redundant equity pairs).
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from synergistic_granger import ante_sg
import synergy_eprocess as sep_int          # interventional ANTE (binary)
from baselines import (interaction_information, surd_synergy, oinfo,
                       peid_synergy, pid_wb_synergy, permutation_pvalue)
OUT = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "..", "data")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05


def _bin(x, nbins):
    x = np.asarray(x, float)
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    return np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)


def peid_cont(a, b, y, nbins=3):
    return peid_synergy(_bin(a, nbins), _bin(b, nbins), _bin(y, nbins), states=nbins)


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    s = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[lab == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


# ---------------------------------------------------------------- Part 0: canonical discrete
def part0_canonical(n=40000, seed=0, n_stream=3000, reps=200):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, n); B = rng.integers(0, 2, n)
    cases = {"XOR (pure synergy)": A ^ B, "AND": A & B,
             "redundant (copy A)": A.copy(), "independent (null)": rng.integers(0, 2, n)}
    rows = []
    for name, Y in cases.items():
        Y = np.asarray(Y, int)
        rows.append(dict(case=name, PEID_2026=peid_synergy(A, B, Y), PID_WB=pid_wb_synergy(A, B, Y),
                         SURD=surd_synergy(A, B, Y, nbins=2)))
    # interventional ANTE reject-rate over short randomized streams (shows error control + power)
    ante_rej = {}
    for name in cases:
        rej = 0
        for _ in range(reps):
            a = rng.integers(0, 2, n_stream); b = rng.integers(0, 2, n_stream)
            if name == "XOR (pure synergy)": c = a ^ b
            elif name == "AND": c = a & b
            elif name == "redundant (copy A)": c = a.copy()
            else: c = rng.integers(0, 2, n_stream)
            r = sep_int.run_eprocess(a, b, c, alpha=ALPHA)
            rej += r["rejected"]
        ante_rej[name] = rej / reps
    for row in rows:
        row["ANTE_rejrate"] = ante_rej[row["case"]]
    print("Part0 canonical:")
    for row in rows:
        print(f"  {row['case']:22s} PEID={row['PEID_2026']:+.3f} PID_WB={row['PID_WB']:+.3f} "
              f"SURD={row['SURD']:.3f} ANTE_rej={row['ANTE_rejrate']:.2f}")
    return rows


# ---------------------------------------------------------------- Part A: continuous AUC
def gen(kind, T, rng):
    Z = rng.standard_normal(T); A = np.zeros(T); B = np.zeros(T); Y = np.zeros(T)
    for t in range(1, T):
        A[t] = 0.3*A[t-1] + 0.5*Z[t] + rng.standard_normal()
        B[t] = 0.3*B[t-1] + 0.5*Z[t] + rng.standard_normal()
    for t in range(1, T):
        c = 0.3*Z[t]
        if kind == "additive":     Y[t] = 0.5*A[t-1]+0.5*B[t-1]+c+rng.standard_normal()
        elif kind == "ownnonlin":  Y[t] = 0.6*A[t-1]**2+0.4*B[t-1]+c+rng.standard_normal()
        elif kind == "synergistic":Y[t] = 0.9*A[t-1]*B[t-1]+0.2*A[t-1]+c+rng.standard_normal()
    return Y, A, B, Z


def partA_auc(n_rep=100, T=3000, seed=1):
    rng = np.random.default_rng(seed)
    classes = ["additive", "ownnonlin", "synergistic"]
    methods = ["ANTE_logE", "SURD", "PEID_2026", "InteractionInfo"]
    sc = {m: {k: [] for k in classes} for m in methods}
    for _ in range(n_rep):
        for k in classes:
            Y, A, B, Z = gen(k, T, rng)
            r = ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
            sc["ANTE_logE"][k].append(np.log10(max(r["final_E"], 1e-9)))
            a, b, y = A[:-1], B[:-1], Y[1:]
            sc["SURD"][k].append(surd_synergy(a, b, y, nbins=6))
            sc["PEID_2026"][k].append(peid_cont(a, b, y, nbins=4))
            sc["InteractionInfo"][k].append(interaction_information(a, b, y))
    aucs = {m: auc(sc[m]["synergistic"], sc[m]["additive"] + sc[m]["ownnonlin"]) for m in methods}
    # discrimination: does own-nonlinearity get mistaken for synergy? (syn vs ownnonlin only)
    auc_ownnl = {m: auc(sc[m]["synergistic"], sc[m]["ownnonlin"]) for m in methods}
    means = {m: {k: float(np.mean(sc[m][k])) for k in classes} for m in methods}
    print("PartA AUC(syn vs {add,ownnl}):", {m: round(v, 3) for m, v in aucs.items()})
    print("PartA AUC(syn vs ownnl only)  :", {m: round(v, 3) for m, v in auc_ownnl.items()})
    # figure
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(methods)); w = 0.38
    ax.bar(x - w/2, [aucs[m] for m in methods], w, label="syn vs {additive, own-nl}", color=PAL[0])
    ax.bar(x + w/2, [auc_ownnl[m] for m in methods], w, label="syn vs own-nonlinearity", color=PAL[2])
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15); ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection AUC"); ax.set_title("Synergy detection on continuous DGPs")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/bench_auc.png"); plt.close()
    return dict(auc_syn_vs_non=aucs, auc_syn_vs_ownnl=auc_ownnl, means=means, n_rep=n_rep, T=T)


# ---------------------------------------------------------------- Part C: monitoring
def partC_monitoring(n_rep=250, T=3000, seed=2, checks=15):
    rng = np.random.default_rng(seed)
    ante_fp = peek_fp = 0
    horizons = np.linspace(400, T-1, checks).astype(int)
    for _ in range(n_rep):
        Y, A, B, Z = gen("additive", T, rng)         # null: no synergy
        ante_fp += ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA)["rejected"]
        a, b, y = A[:-1], B[:-1], Y[1:]
        hit = False
        for h in horizons:
            _, p = permutation_pvalue(a[:h], b[:h], y[:h], n_perm=50, seed=0)
            if p < ALPHA: hit = True; break
        peek_fp += hit
    res = dict(ante_type1=ante_fp/n_rep, peeking_batch_type1=peek_fp/n_rep, n_rep=n_rep,
               checks=checks, alpha=ALPHA)
    print("PartC monitoring:", res)
    return res


# ---------------------------------------------------------------- Part D: real data
def partD_real():
    from portfolio_variance_synergy import PAIRS
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    rows = []
    for a, b, name in PAIRS:
        Ra, Rb = rets[a].values, rets[b].values; RV = (0.5*Ra + 0.5*Rb)**2
        r = ante_sg(z(RV), z(Ra), z(Rb), p=1, alpha=ALPHA, contemp=True)
        cross = not any(t in name for t in ["both equity", "both cyclical"])
        rows.append(dict(pair=f"{a}/{b}", cross_asset=cross,
                         ANTE_logE=float(np.log10(max(r["final_E"], 1e-9))),
                         SURD=surd_synergy(z(Ra), z(Rb), z(RV), nbins=6),
                         PEID_2026=peid_cont(z(Ra), z(Rb), z(RV), nbins=4),
                         InteractionInfo=interaction_information(z(Ra), z(Rb), z(RV))))
    df = pd.DataFrame(rows)
    sep = {}
    for m in ["ANTE_logE", "SURD", "PEID_2026", "InteractionInfo"]:
        sep[m] = auc(df[df.cross_asset][m].values, df[~df.cross_asset][m].values)
    print("PartD real-data AUC (cross-asset vs redundant equity):",
          {m: round(v, 3) for m, v in sep.items()})
    return dict(separation_auc=sep, table=df.to_dict(orient="records"))


def main():
    np.seterr(all="ignore")
    res = {}
    res["part0_canonical"] = part0_canonical()
    res["partA_detection_auc"] = partA_auc()
    res["partC_monitoring"] = partC_monitoring()
    res["partD_real"] = partD_real()
    json.dump(res, open(f"{OUT}/benchmark_results.json", "w"), indent=2, default=str)
    print("\nbenchmark complete.")


if __name__ == "__main__":
    main()
