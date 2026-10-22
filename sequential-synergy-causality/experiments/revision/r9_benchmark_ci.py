"""R9: benchmark with replication-level scores and confidence intervals.

Part A  detection AUC (synergy vs additive / own-nonlinear) on the continuous DGPs of
        experiments/benchmark_sota.py (same generator `gen`), 100 replications, T=3000.
        Per-replication scores are stored; AUC CIs are percentile bootstrap (auc_ci).
Part C  monitoring Type-I under 15 looks (additive null, 250 replications, T=3000):
        ANTE-legacy, ANTE-v2 addjoint (max E >= 20), peeking batch permutation test,
        and one-sided z-tests on the mean of the raw addjoint score evaluated at the same
        15 looks with Pocock and O'Brien-Fleming Lan-DeMets boundaries (Monte Carlo, 200k
        paths).  The same look-based tests are also run for power on the 'syn' DGP (100 reps).

Runs with 2 worker processes.  Output: results/r9_benchmark_ci.json, .png, _summary.md
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from common import wilson, rate_ci, auc_ci, RES  # noqa: E402
from gs_shared import v2_addjoint, legacy_run, gs_boundaries  # noqa: E402
from ante_v2 import raw_score  # noqa: E402
from baselines import (surd_synergy, peid_synergy_continuous, interaction_information,  # noqa: E402
                       permutation_pvalue)
from benchmark_sota import gen  # noqa: E402

T = 3000
ALPHA = 0.05
N_A = 100
N_C = 250
N_POW = 100
CLASSES = ["additive", "ownnonlin", "synergistic"]
HORIZONS = np.linspace(400, T - 1, 15).astype(int)   # 15 looks, as in benchmark_sota Part C
SEED_A, SEED_C, SEED_POW = 9001, 9003, 9004
PERM_N = 50
PAL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]


# ----------------------------------------------------------------- workers
def partA_rep(rep):
    rng = np.random.default_rng([SEED_A, rep])
    out = {}
    for k in CLASSES:
        Y, A, B, Z = gen(k, T, rng)
        leg = legacy_run(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
        v2 = v2_addjoint(Y, A, B, cond=[Z], p=1)
        a, b, y = A[:-1], B[:-1], Y[1:]
        out[k] = dict(
            ANTE_legacy=leg["final_log10"],
            ANTE_v2_addjoint=v2["final_log10"],
            SURD=float(surd_synergy(a, b, y, nbins=6)),
            PEID=float(peid_synergy_continuous(a, b, y)),
            InteractionInfo=float(interaction_information(a, b, y)),
        )
    return out


def look_test_rep(args):
    """One replication of the monitoring experiment for DGP `kind`."""
    kind, rep, seed = args
    rng = np.random.default_rng([seed, rep])
    Y, A, B, Z = gen(kind, T, rng)
    leg = legacy_run(Y, A, B, cond=[Z], p=1, alpha=ALPHA)
    v2 = v2_addjoint(Y, A, B, cond=[Z], p=1)
    a, b, y = A[:-1], B[:-1], Y[1:]
    perm_hit = False
    for h in HORIZONS:
        _, pv = permutation_pvalue(a[:h], b[:h], y[:h], n_perm=PERM_N, seed=0)
        if pv < ALPHA:
            perm_hit = True
            break
    raw = raw_score(Y, v2["pred"], kind="addjoint")
    Zs = np.empty(len(HORIZONS))
    for k, h in enumerate(HORIZONS):
        seg = raw[1:h + 1]                      # observations t = 1..h
        Zs[k] = seg.mean() / (seg.std(ddof=1) / np.sqrt(len(seg)))
    return dict(legacy=leg["rejected"], v2=v2["rejected"], v2_max_log10=v2["max_log10"],
                perm=perm_hit, Z=Zs.tolist())


# ----------------------------------------------------------------- helpers
def auc_block(sc):
    out = {}
    for m in ["ANTE_legacy", "ANTE_v2_addjoint", "SURD", "PEID", "InteractionInfo"]:
        syn = np.array(sc[m]["synergistic"])
        add = np.array(sc[m]["additive"]); own = np.array(sc[m]["ownnonlin"])
        out[m] = dict(
            syn_vs_add_plus_own=auc_ci(syn, np.concatenate([add, own]), n_boot=1000, seed=0),
            syn_vs_ownnonlin=auc_ci(syn, own, n_boot=1000, seed=1),
            syn_vs_additive=auc_ci(syn, add, n_boot=1000, seed=2),
            mean_by_class={k: float(np.mean(sc[m][k])) for k in CLASSES},
            median_by_class={k: float(np.median(sc[m][k])) for k in CLASSES},
        )
    return out


def gs_table(rows, c_p, c_o, key_Z="Z"):
    Zs = np.array([r[key_Z] for r in rows])
    hitP = (Zs >= c_p[None, :]).any(axis=1)
    hitO = (Zs >= c_o[None, :]).any(axis=1)
    return hitP, hitO


def make_figure(auc_res, c_res, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    methods = ["ANTE_legacy", "ANTE_v2_addjoint", "SURD", "PEID", "InteractionInfo"]
    x = np.arange(len(methods))
    est = [auc_res[m]["syn_vs_add_plus_own"]["auc"] for m in methods]
    lo = [auc_res[m]["syn_vs_add_plus_own"]["lo"] for m in methods]
    hi = [auc_res[m]["syn_vs_add_plus_own"]["hi"] for m in methods]
    ax1.bar(x, est, color=PAL[0], yerr=[np.array(est) - lo, np.array(hi) - np.array(est)],
            capsize=4, label="syn vs {additive, own-nl}")
    est2 = [auc_res[m]["syn_vs_ownnonlin"]["auc"] for m in methods]
    lo2 = [auc_res[m]["syn_vs_ownnonlin"]["lo"] for m in methods]
    hi2 = [auc_res[m]["syn_vs_ownnonlin"]["hi"] for m in methods]
    ax1.bar(x + 0.38, est2, 0.38, color=PAL[2], yerr=[np.array(est2) - lo2, np.array(hi2) - np.array(est2)],
            capsize=4, label="syn vs own-nonlinearity")
    ax1.axhline(0.5, color="k", ls=":", lw=1)
    ax1.set_xticks(x + 0.19); ax1.set_xticklabels(methods, rotation=15, fontsize=8)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("AUC (95% bootstrap CI)")
    ax1.set_title("Detection AUC, 100 replications"); ax1.legend(fontsize=8, loc="lower right")

    labels = ["ANTE-legacy", "ANTE-v2 (max E>=20)", "perm. batch (peeking)", "GS Pocock (raw score)", "GS OBF (raw score)"]
    keys = ["legacy", "v2", "perm", "gsP", "gsO"]
    tI = [c_res["partC"][k]["rate"] for k in keys]
    l3 = [c_res["partC"][k]["lo"] for k in keys]
    h3 = [c_res["partC"][k]["hi"] for k in keys]
    xx = np.arange(len(keys))
    ax2.bar(xx, tI, color=PAL[:5], yerr=[np.array(tI) - l3, np.array(h3) - np.array(tI)], capsize=4)
    ax2.axhline(ALPHA, color="k", ls="--", lw=1, label="alpha = 0.05")
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, rotation=15, fontsize=8)
    ax2.set_ylabel("Type-I rate (Wilson 95% CI)"); ax2.set_ylim(0, max(0.12, max(h3) * 1.2))
    ax2.set_title("Type-I under 15 looks, additive null (250 reps)"); ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def write_summary(res, path):
    L = []
    L.append("# R9 summary: benchmark with CIs\n")
    L.append(f"Part A: {N_A} reps, T={T}. Part C: {N_C} reps, 15 looks, T={T}, alpha={ALPHA}.\n")
    L.append("## Detection AUC (syn vs additive+own-nonlinear), bootstrap 95% CI\n")
    L.append("| method | AUC | CI | AUC syn vs own-nl | CI |")
    L.append("|---|---|---|---|---|")
    for m, v in res["partA"]["auc"].items():
        a = v["syn_vs_add_plus_own"]; b = v["syn_vs_ownnonlin"]
        L.append(f"| {m} | {a['auc']:.3f} | [{a['lo']:.3f}, {a['hi']:.3f}] | {b['auc']:.3f} | [{b['lo']:.3f}, {b['hi']:.3f}] |")
    L.append("\n## Type-I under 15 looks (additive null), Wilson 95% CI\n")
    L.append("| procedure | rejections | rate | CI |")
    L.append("|---|---|---|---|")
    names = dict(legacy="ANTE-legacy", v2="ANTE-v2 addjoint (max E>=20)", perm="peeking batch permutation",
                 gsP="GS Pocock on raw addjoint score (one-sided)", gsO="GS OBF on raw addjoint score (one-sided)")
    for k, nm in names.items():
        r = res["partC"][k]
        L.append(f"| {nm} | {r['k']}/{r['n']} | {r['rate']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |")
    L.append("\n## Group-sequential boundaries (Monte Carlo, 200k paths)\n")
    L.append("| look | horizon | c Pocock | c OBF |")
    L.append("|---|---|---|---|")
    for k, h in enumerate(HORIZONS):
        L.append(f"| {k+1} | {h} | {res['gs_boundaries']['pocock'][k]:.3f} | {res['gs_boundaries']['obf'][k]:.3f} |")
    L.append(f"\nMC total crossing probability: Pocock {res['gs_boundaries']['mc_total_alpha_pocock']:.4f}, "
             f"OBF {res['gs_boundaries']['mc_total_alpha_obf']:.4f}\n")
    L.append("## Power on syn DGP (100 reps), Wilson 95% CI\n")
    L.append("| procedure | rejections | rate | CI |")
    L.append("|---|---|---|---|")
    for k, nm in names.items():
        r = res["power_syn"][k]
        L.append(f"| {nm} | {r['k']}/{r['n']} | {r['rate']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    np.seterr(all="ignore")
    t0 = time.time()
    res = {"script": "r9_benchmark_ci.py", "T": T, "alpha": ALPHA}
    # --- import check
    import benchmark_sota as bs  # noqa: F401
    try:
        probe = surd_synergy(np.random.default_rng(0).random(300), np.random.default_rng(1).random(300),
                             np.random.default_rng(2).random(300), nbins=6)
        surd_status = f"ok (probe value {probe:.4f}; SURD code at vendor/SURD, serial stub for pymp)"
    except Exception as e:  # report exactly what failed
        surd_status = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
    res["imports"] = dict(benchmark_sota="ok", baselines_surd_synergy=surd_status,
                          peid="peid_synergy_continuous", interaction="interaction_information (Gaussian)")
    print("imports:", res["imports"], flush=True)

    with Pool(2) as pool:
        # ---------------- Part A
        rowsA = []
        for i, r in enumerate(pool.imap(partA_rep, range(N_A), chunksize=4)):
            rowsA.append(r)
            if (i + 1) % 20 == 0:
                print(f"PartA {i+1}/{N_A}  t={time.time()-t0:.0f}s", flush=True)
        sc = {m: {k: [r[k][m] for r in rowsA] for k in CLASSES}
              for m in ["ANTE_legacy", "ANTE_v2_addjoint", "SURD", "PEID", "InteractionInfo"]}
        res["partA"] = dict(n_rep=N_A, T=T, seed=SEED_A, auc=auc_block(sc), per_rep=sc)
        print("PartA done", flush=True)

        # ---------------- boundaries
        info = HORIZONS / HORIZONS[-1]
        c_p, mc_p = gs_boundaries(info, "pocock", ALPHA, sided="one", n_paths=200000, seed=12345)
        c_o, mc_o = gs_boundaries(info, "obf", ALPHA, sided="one", n_paths=200000, seed=12345)
        res["gs_boundaries"] = dict(method="Monte Carlo of Brownian look statistics, 200k paths, seed 12345, "
                                    "Lan-DeMets spending, one-sided, info fraction = horizon/T_last",
                                    horizons=HORIZONS.tolist(), info=info.tolist(),
                                    pocock=c_p.tolist(), obf=c_o.tolist(),
                                    mc_total_alpha_pocock=mc_p, mc_total_alpha_obf=mc_o)

        # ---------------- Part C null
        rowsC = list(pool.imap(look_test_rep, [("additive", r, SEED_C) for r in range(N_C)], chunksize=4))
        print(f"PartC null done t={time.time()-t0:.0f}s", flush=True)
        # ---------------- power on syn
        rowsP = list(pool.imap(look_test_rep, [("synergistic", r, SEED_POW) for r in range(N_POW)], chunksize=4))
        print(f"Power syn done t={time.time()-t0:.0f}s", flush=True)

    def pack(rows, c_pp, c_oo):
        hP, hO = gs_table(rows, c_pp, c_oo)
        d = dict(legacy=rate_ci([r["legacy"] for r in rows]),
                 v2=rate_ci([r["v2"] for r in rows]),
                 perm=rate_ci([r["perm"] for r in rows]),
                 gsP=rate_ci(hP), gsO=rate_ci(hO))
        d["v2_median_max_log10"] = float(np.median([r["v2_max_log10"] for r in rows]))
        return d

    res["partC"] = pack(rowsC, c_p, c_o)
    res["partC"].update(n_rep=N_C, seed=SEED_C, looks=len(HORIZONS), horizons=HORIZONS.tolist())
    res["power_syn"] = pack(rowsP, c_p, c_o)
    res["power_syn"].update(n_rep=N_POW, seed=SEED_POW, dgp="synergistic (gen: 0.9*A_{t-1}B_{t-1} + 0.2*A_{t-1} + 0.3 Z + e)")
    res["settings"] = dict(
        ANTE_legacy="synergistic_granger.ante_sg(p=1, alpha=0.05, forget=0.995, ridge=1e-2, warmup=60, clip_b=10, cap_frac=0.9), cond=[Z]",
        ANTE_v2_addjoint="ante_v2: forget=0.995, ridge=1e-2, cap=3.0, gamma=0.9, warmup=50, p=1, feat_clip=None, cond=[Z], reject if max E>=20",
        SURD="baselines.surd_synergy(a=A_{t-1}, b=B_{t-1}, y=Y_t, nbins=6): histogramdd 6x6x6 then SURD serial; total synergistic info",
        PEID="baselines.peid_synergy_continuous(intervene=True, n_mc=n, seed=0): polynomial-lifted affine-Gaussian MI",
        InteractionInfo="baselines.interaction_information: Gaussian co-information on (A_{t-1},B_{t-1},Y_t)",
        peeking_permutation=f"baselines.permutation_pvalue(stat=interaction_information, n_perm={PERM_N}, seed=0) at 15 horizons",
        group_sequential="z-test of mean of raw_score(addjoint) over observations 1..h (estimated sd), one-sided, 15 looks",
        tuning="No baseline tuned per method; all use the defaults in the code (SURD nbins=6, PEID intervene=True).",
        seeds=dict(partA_rep=f"default_rng([{SEED_A}, rep])", partC=f"default_rng([{SEED_C}, rep])",
                   power_syn=f"default_rng([{SEED_POW}, rep])"),
        sample_sizes=dict(partA=f"{N_A} reps x 3 DGPs, T={T}", partC=f"{N_C} reps, T={T}", power_syn=f"{N_POW} reps, T={T}"),
        bootstrap="auc_ci percentile bootstrap, 1000 resamples; Wilson 95% intervals for rates",
        note="Per-rep seeds differ from the earlier benchmark_sota.py single-stream RNG, so point estimates differ slightly.",
    )
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(RES, "r9_benchmark_ci.json"), "w"), indent=2, default=str)
    make_figure(res["partA"]["auc"], res, os.path.join(RES, "r9_benchmark_ci.png"))
    write_summary(res, os.path.join(RES, "r9_benchmark_ci_summary.md"))
    print("done", round(res["runtime_s"]), "s", flush=True)


if __name__ == "__main__":
    main()
