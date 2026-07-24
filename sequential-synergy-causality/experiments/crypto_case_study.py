"""
ANTE-SG on hourly crypto (16.8k hours, 10 assets) -- the inefficient, high-sample
regime where genuine synergistic Granger structure is most plausible.

Scans every (target; source-pair) triple for synergy beyond BTC (the dominant
common factor), validates the top hits out-of-sample, and plots the running
e-value of the strongest triple.  Outputs to experiments/results/.
"""
import os, sys, json
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from synergistic_granger import ante_sg, batch_synergy_estimate, pairwise_granger_f, build_feature_fn
OUT = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "..", "data")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05; BTC = "BTC-USD"


def oos_syn(z, tgt, a, b, cond, p=1, split=0.7):
    feats = build_feature_fn(z[tgt].values, {"A": z[a].values, "B": z[b].values},
                             [z[c].values for c in cond], p)
    T = len(z); cut = int(p + split * (T - p))
    def design(idx):
        X = {k: [] for k in ("base", "A", "B", "AB")}; y = []
        for t in idx:
            f = feats(t)
            for k in X: X[k].append(f[k])
            y.append(z[tgt].values[t])
        return {k: np.array(v) for k, v in X.items()}, np.array(y)
    Xtr, ytr = design(range(p, cut)); Xte, yte = design(range(cut, T))
    def L(k, ridge=1e-2):
        X = Xtr[k]; w = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ ytr)
        return np.mean((yte - Xte[k] @ w) ** 2)
    Lb, La, Lbb, Lab = L("base"), L("A"), L("B"), L("AB")
    return float(La + Lbb - Lab - Lb)


def main(mode="return"):
    rets = pd.read_csv(os.path.join(DATA, "crypto_hourly_returns.csv"), index_col=0, parse_dates=True)
    panel = rets if mode == "return" else rets.abs()
    z = (panel - panel.mean()) / panel.std()
    cols = list(z.columns)
    print(f"[{mode}] crypto panel {z.shape} {z.index[0]} .. {z.index[-1]}")

    recs = []
    for C in cols:
        others = [a for a in cols if a != C]
        for A, B in combinations(others, 2):
            cond = [BTC] if (C != BTC and A != BTC and B != BTC) else []
            r = ante_sg(z[C].values, z[A].values, z[B].values,
                        cond=[z[c].values for c in cond], p=1, alpha=ALPHA)
            recs.append(dict(target=C, a=A, b=B, cond=",".join(cond),
                             final_E=r["final_E"], reject_time=r["reject_time"]))
    df = pd.DataFrame(recs).sort_values("final_E", ascending=False).reset_index(drop=True)
    m = len(df)
    df["fw"] = df["final_E"] >= m / ALPHA
    n_raw = int((df["reject_time"].notna()).sum()); n_fw = int(df["fw"].sum())
    print(f"{m} triples | raw rejections {n_raw} | family-wise {n_fw} (thr {m/ALPHA:.0f})")

    def lab(r): return f"{r['a'].replace('-USD','')}+{r['b'].replace('-USD','')}→{r['target'].replace('-USD','')}"
    top = df.head(15).copy()
    print("Top synergistic triples (beyond BTC):")
    for _, r in top.head(10).iterrows():
        print(f"  {lab(r):20s} log10E={np.log10(max(r['final_E'],1e-9)):+.2f} t*={r['reject_time']} fw={r['fw']}")

    # ranked chart
    fig, ax = plt.subplots(figsize=(9, 6))
    labs = [lab(r) for _, r in top[::-1].iterrows()]
    vals = [np.log10(max(v, 1e-9)) for v in top["final_E"][::-1]]
    cols_b = [PAL[2] if s else PAL[0] for s in top["fw"][::-1]]
    ax.barh(labs, vals, color=cols_b, edgecolor="black", lw=0.5)
    ax.axvline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject $\log_{10}(1/\alpha)$")
    ax.axvline(np.log10(m / ALPHA), color="purple", ls=":", lw=1.3, label=r"family-wise")
    ax.set_xlabel(r"final log$_{10}$ e-value (synergy beyond BTC)")
    ax.set_title(f"Hourly crypto: top synergistic Granger triples ({mode})")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/crypto_synergy_network_{mode}.png"); plt.close()

    # OOS validation for top 8
    oos = []
    for _, r in top.head(8).iterrows():
        cond = [c for c in [BTC] if c and r["cond"]]
        so = oos_syn(z, r["target"], r["a"], r["b"], cond)
        oos.append(dict(triple=lab(r), syn_oos=so,
                        reject_time=(None if pd.isna(r["reject_time"]) else int(r["reject_time"]))))
        print(f"  OOS {lab(r):20s} syn_oos={so:+.4f}")

    # event trajectory of strongest triple
    r0 = top.iloc[0]; cond0 = [BTC] if r0["cond"] else []
    tr = ante_sg(z[r0["target"]].values, z[r0["a"]].values, z[r0["b"]].values,
                 cond=[z[c].values for c in cond0], p=1, alpha=ALPHA)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(z.index, np.log10(np.maximum(tr["E"], 1e-3)), color=PAL[3], lw=1.1)
    ax.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject threshold")
    ax.set_xlabel("time (hourly)"); ax.set_ylabel(r"$\log_{10} E_t$")
    ax.set_title(f"Running synergy evidence: {lab(r0)} (hourly, {mode})")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/crypto_event_trajectory_{mode}.png"); plt.close()

    out = dict(mode=mode, n_obs=int(z.shape[0]), n_assets=int(z.shape[1]),
               start=str(z.index[0]), end=str(z.index[-1]),
               n_tests=m, n_reject_raw=n_raw, n_reject_familywise=n_fw,
               top=[dict(triple=lab(r), target=r["target"], a=r["a"], b=r["b"],
                         final_E=float(r["final_E"]),
                         reject_time=(None if pd.isna(r["reject_time"]) else int(r["reject_time"])),
                         fw=bool(r["fw"])) for _, r in top.iterrows()],
               oos=oos)
    json.dump(out, open(f"{OUT}/crypto_results_{mode}.json", "w"), indent=2)
    print(f"[{mode}] done.")
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    for mode in ["return", "volatility"]:
        main(mode)
