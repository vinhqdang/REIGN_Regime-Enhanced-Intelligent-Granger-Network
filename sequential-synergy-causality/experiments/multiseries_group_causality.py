"""
Multi-series GROUP causality with ANTE's k-source test (ante_group):
does a GROUP {X1..Xk} jointly cause Y beyond the sum of parts?

B  PHYSICS (real turbulence, SURD's own energy-cascade data): do the 3 coarser
   scales JOINTLY transfer energy to the finest scale?  -> YES, strongly.
C  FINANCE (systemic risk): do groups of sectors JOINTLY drive market stress
   beyond the sum?  -> NO, even 3-/4-way.

Shows group causality is real & detectable in physics but genuinely absent in
daily finance -- consistent with the whole study.  Saves JSON + figure.
"""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from synergistic_granger import ante_group
OUT = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "..", "data")
SURD = os.path.join(os.path.dirname(HERE), "vendor", "SURD", "data", "energy_cascade_signals.mat")
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
ALPHA = 0.05
z = lambda x: (np.asarray(x, float) - np.nanmean(x)) / np.nanstd(x)


def turbulence():
    if not os.path.exists(SURD):
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Computational-Turbulence-Group/SURD.git",
                        os.path.join(os.path.dirname(HERE), "vendor", "SURD")], check=True)
    X = sio.loadmat(SURD)["X"]
    Z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    r = ante_group(Z[3], [Z[0], Z[1], Z[2]], p=1, alpha=ALPHA, contemp=False)
    print(f"B turbulence {{scale-1,2,3}} -> scale-4: reject={r['rejected']} t*={r['reject_time']} "
          f"log10E={np.log10(max(r['final_E'],1e-9)):.2f}")
    return r


def systemic():
    rets = pd.read_csv(os.path.join(DATA, "returns_panel.csv"), index_col=0, parse_dates=True)
    vol = rets.abs(); mkt = z(vol["SPY"].values)
    groups = [(["XLF", "HYG"], "banks + credit"),
              (["XLF", "HYG", "TLT"], "banks + credit + rates"),
              (["XLF", "HYG", "USO"], "banks + credit + oil"),
              (["XLF", "HYG", "TLT", "USO"], "banks + credit + rates + oil"),
              (["XLF", "XLK", "XLE"], "financials + tech + energy"),
              (["HYG", "TLT", "UUP"], "credit + rates + dollar")]
    out = []
    for names, lbl in groups:
        r = ante_group(mkt, [z(vol[n].values) for n in names], p=1, alpha=ALPHA, contemp=True)
        out.append(dict(group=lbl, k=len(names), reject=r["rejected"],
                        log10E=float(np.log10(max(r["final_E"], 1e-9)))))
        print(f"C systemic {lbl} ({len(names)}-way): reject={r['rejected']} log10E={out[-1]['log10E']:.2f}")
    return out


def main():
    np.seterr(all="ignore")
    tb = turbulence(); sy = systemic()
    json.dump({"turbulence_3way": {"reject": tb["rejected"], "reject_time": tb["reject_time"],
                                   "log10E": float(np.log10(max(tb["final_E"], 1e-9)))},
               "systemic_groups": sy}, open(f"{OUT}/multiseries_group.json", "w"), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(np.log10(np.maximum(tb["E"], 1e-3)), color=PAL[2], lw=1.6)
    ax1.axhline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject threshold $\log_{10}(1/\alpha)$")
    if tb["reject_time"]:
        ax1.axvline(tb["reject_time"], color="gray", ls=":", lw=1.2)
        ax1.text(tb["reject_time"], 0.5, f"  detected\n  at t={tb['reject_time']}", fontsize=9, color="#444")
    ax1.set_xlabel("samples seen"); ax1.set_ylabel(r"$\log_{10} E_t$")
    ax1.set_title("B · PHYSICS (real turbulence):\n3 coarse scales JOINTLY cascade into the finest scale", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    labs = [g["group"] for g in sy][::-1]; vals = [g["log10E"] for g in sy][::-1]
    ax2.barh(labs, vals, color=["#55a868" if v > 1.3 else "#c44e52" for v in vals], edgecolor="black", lw=0.5)
    ax2.axvline(np.log10(1 / ALPHA), color="red", ls="--", lw=1.3, label=r"reject threshold")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel(r"group-synergy evidence $\log_{10} E$")
    ax2.set_title("C · FINANCE (systemic risk): sector groups\ndo NOT jointly drive market stress beyond the sum", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    fig.suptitle("Multi-series group causality: real in physics, absent in daily finance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(f"{OUT}/multiseries_group_causality.png"); plt.close()
    print("saved multiseries_group_causality.png")


if __name__ == "__main__":
    main()
