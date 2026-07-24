"""
Real-data case study: synergistic Granger causality in a 16-asset financial panel
(sector ETFs + macro/vol/commodity drivers, 10y daily), conditioned on the market
(SPY) so we isolate synergy BEYOND common-market co-movement.

Outputs (experiments/results/):
  fin_synergy_network.png    top synergistic (A,B -> C) triples as a ranked chart
  fin_event_trajectory.png   running e-value of a flagged triple across 2016-2026,
                             annotated with COVID (Mar 2020) and the 2022 shock
  fin_oos_validation.png     out-of-sample predictive-gain check for flagged triples
  financial_results.json     ranked discoveries, Bonferroni-e control, OOS gains
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
from synergistic_granger import ante_sg, batch_synergy_estimate, pairwise_granger_f

OUT = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05
MARKET = "SPY"


def load(mode="volatility"):
    """Return (raw_returns, analysis_panel, manifest).

    Financial synergy in the conditional *mean* of daily returns is weak (markets
    are near-efficient at the mean).  It is strong and economically documented in
    *volatility* (volatility spillovers, Diebold-Yilmaz).  So the headline
    analysis uses a daily volatility proxy = |return|, standardized; mode='return'
    reproduces the (weaker) return-mean analysis.
    """
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    if mode == "volatility":
        panel = rets.abs()
    else:
        panel = rets
    z = (panel - panel.mean()) / panel.std()
    return rets, z, manifest


def scan(z, roles, p=1):
    """ANTE-SG over all (target C; unordered source pair {A,B}) beyond the market."""
    assets = [a for a in z.columns if a != MARKET]
    Zmkt = z[MARKET].values
    records = []
    for C in assets:
        others = [a for a in assets if a != C]
        for A, B in combinations(others, 2):
            r = ante_sg(z[C].values, z[A].values, z[B].values, cond=[Zmkt],
                        p=p, alpha=ALPHA)
            records.append(dict(target=C, a=A, b=B,
                                final_E=r["final_E"], reject_time=r["reject_time"],
                                mean_score=r["mean_score"]))
    df = pd.DataFrame(records).sort_values("final_E", ascending=False).reset_index(drop=True)
    m = len(df)
    # Bonferroni-on-e-values: family-wise valid discovery if final_E >= m/alpha
    df["fw_significant"] = df["final_E"] >= m / ALPHA
    return df, m


def oos_gain(z, target, a, b, p=1, split=0.7):
    """Out-of-sample check: does the JOINT (with cross-term) model predict the
    target better than the best ADDITIVE (sum-of-parts) model on held-out data?
    Positive => the discovered synergy is real predictive structure, not in-sample
    overfit.  Uses a simple train/test split with ridge OLS.
    """
    from synergistic_granger import build_feature_fn
    Zmkt = z[MARKET].values
    feats = build_feature_fn(z[target].values,
                             {"A": z[a].values, "B": z[b].values}, [Zmkt], p)
    T = len(z); idx = list(range(p, T)); cut = int(p + split * (T - p))
    def design(keys, rng_idx):
        X = {k: [] for k in keys}; y = []
        for t in rng_idx:
            f = feats(t)
            for k in keys: X[k].append(f[k])
            y.append(z[target].values[t])
        return {k: np.array(v) for k, v in X.items()}, np.array(y)
    tr = list(range(p, cut)); te = list(range(cut, T))
    Xtr, ytr = design(["base", "A", "B", "AB"], tr)
    Xte, yte = design(["base", "A", "B", "AB"], te)
    def fit_pred(k, ridge=1e-2):
        X = Xtr[k]; d = X.shape[1]
        w = np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ ytr)
        return np.mean((yte - Xte[k] @ w) ** 2)
    L = {k: fit_pred(k) for k in ["base", "A", "B", "AB"]}
    # synergy on OOS loss scale: (gain of joint) - (sum of individual gains)
    syn_oos = (L["A"] + L["B"] - L["AB"] - L["base"])
    return dict(L_base=L["base"], L_A=L["A"], L_B=L["B"], L_AB=L["AB"], syn_oos=syn_oos)


def event_trajectory(z, target, a, b, dates, p=1):
    Zmkt = z[MARKET].values
    r = ante_sg(z[target].values, z[a].values, z[b].values, cond=[Zmkt], p=p, alpha=ALPHA)
    E = r["E"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(dates, np.log10(np.maximum(E, 1e-3)), color=PAL[3], lw=1.4)
    ax.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.4,
               label=r"reject threshold $\log_{10}(1/\alpha)$")
    for lbl, d0 in [("COVID crash", "2020-03-01"), ("2022 rate shock", "2022-06-01")]:
        try:
            x = pd.Timestamp(d0)
            ax.axvline(x, color="gray", ls=":", lw=1.2)
            ax.annotate(lbl, xy=(x, ax.get_ylim()[1] * 0.9), fontsize=8, color="gray",
                        rotation=90, va="top")
        except Exception:
            pass
    ax.set_xlabel("date"); ax.set_ylabel(r"$\log_{10} E_t$")
    ax.set_title(f"Running synergy evidence: {{{a}, {b}}} $\\to$ {target} (beyond market)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fin_event_trajectory.png"); plt.close()
    return r


def main(mode="volatility"):
    rets, z, manifest = load(mode=mode)
    dates = rets.index
    print(f"[{mode}] panel: {z.shape[0]} days x {z.shape[1]} assets, "
          f"{manifest['start']}..{manifest['end']}")

    df, m = scan(z, {t: manifest['tickers'][t]['role'] for t in manifest['tickers']})
    top = df.head(15).copy()
    names = manifest["tickers"]
    def lab(row): return f"{row['a']}+{row['b']}→{row['target']}"
    print("\nTop synergistic triples (beyond market):")
    for _, r in top.head(10).iterrows():
        print(f"  {lab(r):18s} log10E={np.log10(max(r['final_E'],1e-9)):6.2f} "
              f"t*={r['reject_time']} fw_sig={r['fw_significant']}")

    # ranked bar chart of top triples
    fig, ax = plt.subplots(figsize=(9, 6))
    labs = [lab(r) for _, r in top[::-1].iterrows()]
    vals = [np.log10(max(v, 1e-9)) for v in top["final_E"][::-1]]
    cols = [PAL[2] if s else PAL[0] for s in top["fw_significant"][::-1]]
    ax.barh(labs, vals, color=cols, edgecolor="black", lw=0.5)
    ax.axvline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"$\log_{10}(1/\alpha)$")
    ax.axvline(np.log10(m / ALPHA), color="purple", ls=":", lw=1.3,
               label=r"family-wise ($\log_{10}(m/\alpha)$)")
    ax.set_xlabel(r"final log$_{10}$ e-value (evidence for synergy beyond market)")
    ax.set_title("Top synergistic Granger triples in the financial panel")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fin_synergy_network.png"); plt.close()

    # OOS validation for the top few flagged triples + pairwise-Granger contrast
    oos_rows = []
    for _, r in top.head(8).iterrows():
        o = oos_gain(z, r["target"], r["a"], r["b"])
        gA = pairwise_granger_f(z[r["target"]].values, z[r["a"]].values)["pval"]
        gB = pairwise_granger_f(z[r["target"]].values, z[r["b"]].values)["pval"]
        oos_rows.append(dict(triple=lab(r), pairwise_p_a=gA, pairwise_p_b=gB, **o))
    oosdf = pd.DataFrame(oos_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(oosdf))
    cols = [PAL[2] if v > 0 else PAL[1] for v in oosdf["syn_oos"]]
    ax.barh(yy, oosdf["syn_oos"], color=cols, edgecolor="black", lw=0.5)
    ax.set_yticks(yy); ax.set_yticklabels(oosdf["triple"])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("out-of-sample synergy (joint predictive gain − sum of parts)")
    ax.set_title("Held-out validation: positive ⇒ genuine super-additive predictability")
    plt.tight_layout(); plt.savefig(f"{OUT}/fin_oos_validation.png"); plt.close()

    # event trajectory for the single strongest triple
    r0 = top.iloc[0]
    event_trajectory(z, r0["target"], r0["a"], r0["b"], dates)

    out = dict(
        panel=dict(mode=mode, variable=("|return| volatility proxy" if mode == "volatility"
                                        else "return"),
                   n_days=int(z.shape[0]), n_assets=int(z.shape[1]),
                   start=manifest["start"], end=manifest["end"], market=MARKET),
        n_tests=int(m), n_reject_raw=int((df["reject_time"].notna()).sum()),
        n_reject_familywise=int(df["fw_significant"].sum()),
        top_triples=[dict(triple=lab(r), target=r["target"], a=r["a"], b=r["b"],
                          final_E=float(r["final_E"]),
                          reject_time=(None if pd.isna(r["reject_time"]) else int(r["reject_time"])),
                          fw_significant=bool(r["fw_significant"]))
                     for _, r in top.iterrows()],
        oos_validation=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in row.items()} for row in oos_rows],
    )
    with open(f"{OUT}/financial_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTests={m}  raw rejections={out['n_reject_raw']}  "
          f"family-wise significant={out['n_reject_familywise']}")
    print("financial case study complete.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
