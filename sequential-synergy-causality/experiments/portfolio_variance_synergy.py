"""
POSITIVE real-data case study: contemporaneous synergy in portfolio realized
variance.

Textbook finance: a portfolio's variance is NOT the sum of its constituents'
variances -- the covariance (interaction) term is first-order.  For a 50/50
portfolio of two real assets,

    RV_port = ( (R_A + R_B)/2 )^2 = 1/4 R_A^2 + 1/4 R_B^2 + 1/2 R_A R_B ,

the cross term 1/2 R_A R_B is genuine synergy: neither asset's own (even squared)
return captures it, only the joint state does.  This is the risk-decomposition
that underlies diversification.  ANTE (contemporaneous variant) should recover it
from REAL returns -- and should do so only when the covariance term is a distinct
component (cross-asset-class pairs), not when the two assets are so similar that
the cross term is redundant with the individual variances.

Contrast: the LAGGED (Granger) synergy of the same pairs is near-zero (predictive
cross-asset structure is additive+redundant), reproducing the daily-panel finding.

Outputs to experiments/results/: pv_synergy_bars.png, pv_event_trajectories.png,
portfolio_variance_results.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from synergistic_granger import ante_sg, batch_synergy_estimate
OUT = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "..", "data")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05

# curated pairs spanning asset classes (distinct-covariance) and within-equity
# (redundant) to test discrimination
PAIRS = [
    ("GLD", "USO", "gold x oil"),
    ("XLF", "TLT", "financials x bonds"),
    ("SPY", "TLT", "equity x bonds"),
    ("GLD", "TLT", "gold x bonds"),
    ("USO", "UUP", "oil x dollar"),
    ("TLT", "UUP", "bonds x dollar"),
    ("SPY", "GLD", "equity x gold"),
    ("XLE", "TLT", "energy x bonds"),
    ("HYG", "TLT", "credit x bonds"),
    ("XLK", "GLD", "tech x gold"),
    ("XLE", "XLF", "energy x financials (both cyclical)"),
    ("XLK", "XLI", "tech x industrials (both equity)"),
    ("XLF", "XLI", "financials x industrials (both equity)"),
    ("XLE", "XLB", "energy x materials (both cyclical)"),
]


def zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def rv_synergy(Ra, Rb, contemp=True, p=1):
    port = 0.5 * Ra + 0.5 * Rb
    RV = port ** 2
    r = ante_sg(zscore(RV), zscore(Ra), zscore(Rb), p=p, alpha=ALPHA, contemp=contemp)
    syn = batch_synergy_estimate(zscore(RV), zscore(Ra), zscore(Rb), p=p, contemp=contemp)
    return r, syn


def oos_rv(Ra, Rb, contemp=True, p=1, split=0.7):
    from synergistic_granger import build_feature_fn
    RV = zscore((0.5 * Ra + 0.5 * Rb) ** 2)
    feats = build_feature_fn(RV, {"A": zscore(Ra), "B": zscore(Rb)}, [], p, contemp=contemp)
    T = len(RV); cut = int(p + split * (T - p))
    def design(idx):
        X = {k: [] for k in ("base", "A", "B", "AB")}; y = []
        for t in idx:
            f = feats(t)
            for k in X: X[k].append(f[k])
            y.append(RV[t])
        return {k: np.array(v) for k, v in X.items()}, np.array(y)
    Xtr, ytr = design(range(p, cut)); Xte, yte = design(range(cut, T))
    def L(k, ridge=1e-2):
        X = Xtr[k]; w = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ ytr)
        return np.mean((yte - Xte[k] @ w) ** 2)
    return float(L("A") + L("B") - L("AB") - L("base"))


def main():
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    dates = rets.index
    m = len(PAIRS)
    fw_thr = m / ALPHA
    rows = []
    for a, b, name in PAIRS:
        Ra, Rb = rets[a].values, rets[b].values
        rc, sc = rv_synergy(Ra, Rb, contemp=True)
        rl, sl = rv_synergy(Ra, Rb, contemp=False)
        oc = oos_rv(Ra, Rb, contemp=True)
        rows.append(dict(pair=f"{a}/{b}", desc=name,
                         contemp_log10E=float(np.log10(max(rc["final_E"], 1e-9))),
                         contemp_reject_time=rc["reject_time"], contemp_batchSyn=float(sc),
                         contemp_oos_syn=oc, contemp_fw=bool(rc["final_E"] >= fw_thr),
                         lagged_log10E=float(np.log10(max(rl["final_E"], 1e-9))),
                         lagged_batchSyn=float(sl)))
        print(f"  {a}/{b:4s} {name:34s} contempLog10E={rows[-1]['contemp_log10E']:+.2f} "
              f"t*={rc['reject_time']} fw={rows[-1]['contemp_fw']} | laggedLog10E={rows[-1]['lagged_log10E']:+.2f}")
    df = pd.DataFrame(rows).sort_values("contemp_log10E", ascending=False).reset_index(drop=True)

    # bar chart: contemporaneous synergy e-value by pair, coloured by significance
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    labs = [f"{r.pair} ({r.desc})" for r in df[::-1].itertuples()]
    vals = list(df["contemp_log10E"][::-1])
    cols = [PAL[2] if s else PAL[3] for s in df["contemp_fw"][::-1]]
    ax.barh(labs, vals, color=cols, edgecolor="black", lw=0.5)
    ax.axvline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject $\log_{10}(1/\alpha)$")
    ax.axvline(np.log10(fw_thr), color="purple", ls=":", lw=1.3, label=r"family-wise")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"contemporaneous synergy: final $\log_{10}$ e-value")
    ax.set_title("Portfolio realized-variance synergy in real data:\n"
                 "cross-asset-class pairs are super-additive (green); redundant equity pairs are not")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/pv_synergy_bars.png"); plt.close()

    # event trajectories of the 3 strongest
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, r in df.head(3).iterrows():
        a, b = r["pair"].split("/")
        Ra, Rb = rets[a].values, rets[b].values
        tr, _ = rv_synergy(Ra, Rb, contemp=True)
        ax.plot(dates, np.log10(np.maximum(tr["E"], 1e-3)), lw=1.2,
                color=PAL[i % len(PAL)], label=f"{r['pair']} ({r['desc']})")
    ax.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject threshold")
    ax.set_xlabel("date"); ax.set_ylabel(r"$\log_{10} E_t$")
    ax.set_title("Running synergy evidence for portfolio realized variance (contemporaneous)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/pv_event_trajectories.png"); plt.close()

    out = dict(alpha=ALPHA, n_pairs=m, familywise_threshold_log10=float(np.log10(fw_thr)),
               n_contemp_reject_raw=int(df["contemp_reject_time"].notna().sum()),
               n_contemp_familywise=int(df["contemp_fw"].sum()),
               n_lagged_reject_raw=int((df["lagged_log10E"] >= np.log10(1 / ALPHA)).sum()),
               pairs=df.to_dict(orient="records"))
    json.dump(out, open(f"{OUT}/portfolio_variance_results.json", "w"), indent=2, default=str)
    print(f"\nContemporaneous: raw rejections {out['n_contemp_reject_raw']}/{m}, "
          f"family-wise {out['n_contemp_familywise']}/{m}.  "
          f"Lagged rejections {out['n_lagged_reject_raw']}/{m}.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
