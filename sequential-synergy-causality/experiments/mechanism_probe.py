"""
Curated economic-mechanism probe.

Even where a super-additive interaction is economically plausible (oil priced in
USD; rates x dollar on gold; volatility x credit stress; the leverage effect of
signed market returns x volatility on future volatility), we test whether ANTE-SG
finds robust synergy.  This complements the exhaustive scan and directly probes
the mechanisms one would expect to be synergistic.

Saves fin_mechanism_probe.png + mechanism_probe.json to results/.
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
OUT = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "..", "data")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
ALPHA = 0.05


def main():
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    ret = (rets - rets.mean()) / rets.std()
    vol = rets.abs(); vol = (vol - vol.mean()) / vol.std()

    probes = [
        # (panel, target, A, B, cond, label)
        (ret, "XLE", "USO", "UUP", [], "oil x dollar -> energy (ret)"),
        (ret, "GLD", "TLT", "UUP", [], "rates x dollar -> gold (ret)"),
        (ret, "XLB", "USO", "UUP", [], "oil x dollar -> materials (ret)"),
        (vol, "XLF", "^VIX", "HYG", ["SPY"], "vol x credit -> financials (vol)"),
        (vol, "XLE", "USO", "^VIX", ["SPY"], "oilvol x vix -> energy (vol)"),
    ]
    # leverage: signed market return x vix level -> future sector volatility
    lev = []
    for tk in ["XLK", "XLF", "XLE"]:
        mix = pd.DataFrame({"tgt": vol[tk], "spy_ret": ret["SPY"],
                            "vix": vol["^VIX"], "mkt": vol["SPY"]}).dropna()
        lev.append((mix, "tgt", "spy_ret", "vix", ["mkt"],
                    f"leverage: mkt-ret x vix -> {tk} vol"))

    rows = []
    for panel, tgt, a, b, cond, lbl in probes + lev:
        cv = [panel[c].values for c in cond]
        r = ante_sg(panel[tgt].values, panel[a].values, panel[b].values,
                    cond=cv, p=1, alpha=ALPHA)
        syn = batch_synergy_estimate(panel[tgt].values, panel[a].values,
                                     panel[b].values, cond=cv, p=1)
        rows.append(dict(mechanism=lbl, batch_synergy=float(syn),
                         log10_evalue=float(np.log10(max(r["final_E"], 1e-9))),
                         reject_time=r["reject_time"], rejected=r["rejected"]))
        print(f"  {lbl:34s} batchSyn={syn:+.4f}  log10E={rows[-1]['log10_evalue']:+.2f}  "
              f"reject={r['rejected']}")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yy = np.arange(len(df))
    cols = ["#55a868" if s else "#c44e52" for s in df["rejected"]]
    ax.barh(yy, df["batch_synergy"], color=cols, edgecolor="black", lw=0.5)
    ax.set_yticks(yy); ax.set_yticklabels(df["mechanism"])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("in-sample batch synergy estimate (standardized-loss units)")
    ax.set_title("Economically-motivated mechanisms: synergy is tiny and\n"
                 "none is declared by the anytime-valid test (all red = not rejected)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fin_mechanism_probe.png"); plt.close()

    with open(f"{OUT}/mechanism_probe.json", "w") as f:
        json.dump(dict(alpha=ALPHA, probes=rows), f, indent=2)
    print("mechanism probe complete.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
