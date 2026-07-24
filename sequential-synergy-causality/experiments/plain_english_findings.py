"""
Plain-English real-data findings on famous stocks + assets (daily, 10y).

(1) Specific volatility lead-lag: does X's turbulence today predict Y's turbulence
    tomorrow, BEYOND the overall market (SPY)?  -- conditional Granger F-test.
(2) "1+1>2" combined-risk synergy: is a pair's realized-variance super-additive
    in its constituents?  -- ANTE contemporaneous synergy.

Produces results/plain_english_findings.png (+ prints the ranked findings).
Honest scope: these are statistical lead-lag / association (Granger predictability
and covariance structure), NOT proof of real-world causation; the reliable daily
signals live in volatility (risk), not next-day price direction.
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f as fdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from synergistic_granger import ante_sg
OUT = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "..", "data")
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})

NAMES = {"TSLA": "Tesla", "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft",
         "AMZN": "Amazon", "META": "Meta", "JPM": "JPMorgan", "XOM": "ExxonMobil",
         "SPY": "S&P500", "GLD": "Gold", "USO": "Oil", "TLT": "Treasuries", "^VIX": "VIX"}


def load():
    fp = os.path.join(DATA, "famous_returns.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    from download_financial import fetch
    px = {}
    for t in NAMES:
        px[t] = fetch(t, "10y"); time.sleep(0.25)
    df = pd.DataFrame(px).sort_index().dropna(how="any")
    rets = np.log(df).diff().dropna()
    rets.to_csv(fp)
    return rets


def cond_granger(Y, X, Z, p=2):
    T = len(Y); yy = Y[p:]
    def design(cols):
        M = [np.ones(T - p)]
        for c in cols:
            for j in range(p): M.append(c[p - 1 - j:T - 1 - j])
        return np.column_stack(M)
    def rss(M): w, *_ = np.linalg.lstsq(M, yy, rcond=None); return float(np.sum((yy - M @ w) ** 2))
    r0 = rss(design([Y, Z])); Xf = design([Y, Z, X]); r1 = rss(Xf)
    n = len(yy); q = p; k = Xf.shape[1]
    return 1 - fdist.cdf(((r0 - r1) / q) / (r1 / (n - k)), q, n - k)


def main():
    rets = load()
    vol = rets.abs(); V = (vol - vol.mean()) / vol.std()
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    cols = [c for c in V.columns if c != "SPY"]

    lead = []
    for X in cols:
        for Y in cols:
            if X != Y:
                lead.append((X, Y, cond_granger(V[Y].values, V[X].values, V["SPY"].values)))
    lead = sorted(lead, key=lambda r: r[2])[:8]
    print("Volatility lead-lag (beyond market):")
    for X, Y, p in lead:
        print(f"  {NAMES[X]} -> {NAMES[Y]}  (p={p:.1e})")

    pairs = [("JPM", "TLT", "JPMorgan + Treasuries"), ("TSLA", "USO", "Tesla + Oil"),
             ("GLD", "USO", "Gold + Oil"), ("XOM", "TLT", "Exxon + Treasuries"),
             ("SPY", "TLT", "Stocks + Treasuries"), ("TSLA", "NVDA", "Tesla + Nvidia"),
             ("NVDA", "GLD", "Nvidia + Gold"), ("AAPL", "MSFT", "Apple + Microsoft")]
    syn = []
    for a, b, l in pairs:
        Ra, Rb = rets[a].values, rets[b].values; RV = (0.5 * Ra + 0.5 * Rb) ** 2
        le = np.log10(max(ante_sg(z(RV), z(Ra), z(Rb), p=1, contemp=True)["final_E"], 1e-9))
        syn.append((l, le))
    syn = sorted(syn, key=lambda x: x[1])
    print("Combined-risk synergy:")
    for l, v in sorted(syn, key=lambda x: -x[1]):
        print(f"  {l}: log10E={v:+.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    ax1.axis("off")
    ax1.set_title("When one gets turbulent today,\nthe other gets turbulent tomorrow", fontsize=13, fontweight="bold")
    for i, (X, Y, p) in enumerate(lead[::-1]):
        yq = i / (len(lead) - 1)
        ax1.annotate("", xy=(0.62, yq), xytext=(0.30, yq), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#c44e52", lw=2.4))
        ax1.text(0.29, yq, NAMES[X], ha="right", va="center", fontsize=11, fontweight="bold", transform=ax1.transAxes)
        ax1.text(0.63, yq, NAMES[Y], ha="left", va="center", fontsize=11, fontweight="bold", transform=ax1.transAxes)
        stars = "***" if p < 1e-6 else ("**" if p < 1e-3 else "*")
        ax1.text(0.46, yq + 0.03, stars, ha="center", va="bottom", fontsize=10, color="#555", transform=ax1.transAxes)
    ax1.text(0.5, -0.04, "(next-day volatility, beyond the overall market · *** p<1e-6)",
             ha="center", fontsize=9, color="#666", transform=ax1.transAxes)
    labs = [l for l, _ in syn]; vals = [v for _, v in syn]
    colb = ["#55a868" if v > 2.5 else ("#dd8452" if v > 1.3 else "#bbbbbb") for v in vals]
    ax2.barh(labs, vals, color=colb, edgecolor="black", lw=0.5)
    ax2.axvline(1.3, color="gray", ls=":", lw=1)
    ax2.set_xlabel("evidence that the PAIR's risk exceeds the sum of its parts  (log scale)", fontsize=10)
    ax2.set_title("'1 + 1 > 2' risk: pairs whose combined\nrisk is more than each alone", fontsize=13, fontweight="bold")
    for i, v in enumerate(vals):
        ax2.text(v + 0.1, i, ("strong" if v > 2.5 else ("yes" if v > 1.3 else "no")), va="center", fontsize=9, color="#333")
    fig.suptitle("What the data actually says (S&P 500 stocks & assets, 2016-2026)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(f"{OUT}/plain_english_findings.png"); plt.close()
    print("saved plain_english_findings.png")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
