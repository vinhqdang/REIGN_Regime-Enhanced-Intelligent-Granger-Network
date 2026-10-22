"""Paper figures for the revision, drawn from the saved JSON/CSV results.
Output: ../../manuscript/figures/*.png (200 dpi)."""
import os, json, glob, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})
COL = {"addjoint": "#1f77b4", "four": "#ff7f0e", "legacy": "#7f7f7f"}
LAB = {"addjoint": "add-vs-joint (ANTE-SG)", "four": "four-model score", "legacy": "original clipped test"}
J = lambda n: json.load(open(os.path.join(RES, n + ".json")))

# ---- stress tests (R1) -------------------------------------------------
d = J("r1_stress_typeI")
nulls = d["nulls"]
fig, ax = plt.subplots(figsize=(9, 3.3))
for j, m in enumerate(["addjoint", "four", "legacy"]):
    x = np.arange(len(nulls)) + (j - 1) * 0.22
    y = [n["methods"][m]["rate"] for n in nulls]
    lo = [n["methods"][m]["rate"] - n["methods"][m]["lo"] for n in nulls]
    hi = [n["methods"][m]["hi"] - n["methods"][m]["rate"] for n in nulls]
    ax.errorbar(x, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", ms=4, color=COL[m], label=LAB[m], capsize=2)
ax.axhline(0.05, color="red", ls="--", lw=0.9)
ax.set_xticks(range(len(nulls)))
ax.set_xticklabels([n["id"] for n in nulls])
ax.set_ylabel("false-rejection rate\n(max $E_t\\geq 20$)")
ax.set_xlabel("null scenario (see Table 3)")
ax.legend(fontsize=8, loc="upper left")
plt.savefig(os.path.join(FIG, "fig_stress.png")); plt.close()

# ---- correlated sources (R2) -------------------------------------------
d = J("r2_correlated_sources")
cells = d["cells"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for k, dgp in enumerate(["null", "syn"]):
    cs = [c for c in cells if c["dgp"] == dgp]
    rho = [c["methods"]["realised_corr_mean"] for c in cs]
    for m in ["addjoint", "four", "legacy"]:
        r = np.array([c["methods"][m]["rate"] for c in cs])
        lo = np.array([c["methods"][m]["lo"] for c in cs]); hi = np.array([c["methods"][m]["hi"] for c in cs])
        axs[k].plot(rho, r, "o-", ms=3.5, color=COL[m], label=LAB[m])
        axs[k].fill_between(rho, lo, hi, color=COL[m], alpha=0.15)
    axs[k].set_xlabel("correlation of the two sources")
    axs[k].set_ylabel("false-rejection rate" if dgp == "null" else "power")
    axs[k].set_title("additive mechanism (no interaction)" if dgp == "null" else "interaction present (c = 0.9)")
    if dgp == "null": axs[k].axhline(0.05, color="red", ls="--", lw=0.9)
axs[0].legend(fontsize=7)
for dgp, ls in (("null", "--"), ("syn", "-")):
    cs = [c for c in cells if c["dgp"] == dgp]
    rho = [c["methods"]["realised_corr_mean"] for c in cs]
    for m in ["addjoint", "four"]:
        axs[2].plot(rho, [c["methods"][m]["mean_raw_score"] for c in cs], "o" + ls, ms=3.5, color=COL[m],
                    label=f"{LAB[m]}, {'additive' if dgp=='null' else 'interaction'}")
axs[2].axhline(0, color="k", lw=0.7)
axs[2].set_xlabel("correlation of the two sources"); axs[2].set_ylabel("mean raw score (squared-error gap)")
axs[2].set_title("population-like contrast"); axs[2].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_corr.png")); plt.close()

# ---- financial scan (R5) -------------------------------------------------
rows = []
for f in sorted(glob.glob(os.path.join(RES, "r5_scan_*_lag1.csv"))):
    df = pd.read_csv(f)
    panel, target = df.panel[0], df.target[0]
    rows.append((panel, target, df["addjoint_max_log10E"].clip(upper=15).values, len(df)))
order = {"etf16": 0, "stocks13": 1, "world6": 2, "hourly12": 3, "crypto10": 4}
tord = {"ret": 0, "vol": 1, "tail": 2}
rows.sort(key=lambda r: (order[r[0]], tord[r[1]]))
fig, ax = plt.subplots(figsize=(10, 3.8))
for i, (p, t, v, m) in enumerate(rows):
    ax.boxplot([v], positions=[i], widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cfe2f3", lw=0.7), medianprops=dict(color="k"))
    ax.plot(i, v.max(), "v", color="#c44e52", ms=4)
    ax.plot([i - 0.33, i + 0.33], [np.log10(m / 0.05)] * 2, color="k", ls=":", lw=1)
ax.axhline(np.log10(20), color="k", ls="--", lw=0.8)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([f"{p}\n{t}" for p, t, _, _ in rows], fontsize=7)
ax.set_ylabel("running-max $\\log_{10}E$ per hypothesis\n(capped at 15)")
ax.set_ylim(-0.3, 15.5)
ax.text(0.01, 0.95, "dashed: $1/\\alpha$; dotted: family-wise $m/\\alpha$; triangle: maximum", transform=ax.transAxes, fontsize=7, va="top")
plt.savefig(os.path.join(FIG, "fig_scan.png")); plt.close()

# ---- placebo calibration (R15, R15b) -----------------------------------
sets = {}
for fn in ("r15_placebo_calibration", "r15b_placebo_calibration"):
    if os.path.exists(os.path.join(RES, fn + ".json")):
        sets.update(J(fn)["settings"])
if sets:
    n = len(sets)
    cols = min(n, 5); rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(2.6 * cols, 2.6 * rws), squeeze=False)
    qs = [5, 25, 50, 75, 95, 100]
    for ax, (k, v) in zip(axs.ravel(), sets.items()):
        ax.plot(qs, v["real_log10max_quantiles"], "o-", color="#c44e52", ms=3, label="real")
        ax.plot(qs, v["placebo_log10max_quantiles"], "s--", color="#4c72b0", ms=3, label="placebo")
        ax.set_title(k.replace("_", " "), fontsize=8)
        ax.set_xlabel("quantile (%)", fontsize=7)
    axs[0, 0].set_ylabel("$\\log_{10}$ max $E$"); axs[0, 0].legend(fontsize=7)
    for ax in axs.ravel()[n:]: ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_placebo.png")); plt.close()

# ---- semi-synthetic power (R6) ------------------------------------------
d = J("r6_semisynthetic_power")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (k, v) in zip(axs, d["datasets"].items()):
    tab = v["table"]
    x = [t["mean_share"] for t in tab]
    ax.set_title(v["label"], fontsize=9)
    for key, lab, c in (("power_per_hyp_20", "single hypothesis ($E\\geq20$)", "#1f77b4"),
                        ("power_fw_m660", "family-wise, $m=660$", "#ff7f0e"),
                        ("power_fw_m4400", "family-wise, $m=4400$", "#2ca02c")):
        y = np.array([t[key]["rate"] for t in tab]); lo = np.array([t[key]["lo"] for t in tab]); hi = np.array([t[key]["hi"] for t in tab])
        ax.plot(x, y, "o-", ms=3.5, label=lab, color=c)
        ax.fill_between(x, lo, hi, color=c, alpha=0.12)
    ax.axhline(0.8, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("variance share of the injected lagged interaction")
axs[0].set_ylabel("power (40 real triples)"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_semisyn.png")); plt.close()

# ---- direct copies ------------------------------------------------------
for src, dst in (("r11_interventional.png", "fig_interv_typeI.png"), ("r10_power_sweep.png", "fig_power.png"),
                 ("r13_crisis_trajectories.png", "fig_traj.png"), ("r8_portfolio_variance_control.png", "fig_pv.png"),
                 ("r7_billio_connectedness.png", "fig_billio.png")):
    s = os.path.join(RES, src)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(FIG, dst))
print(sorted(os.listdir(FIG)))
