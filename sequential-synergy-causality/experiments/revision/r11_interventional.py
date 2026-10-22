"""R11: interventional (randomised 2x2) Type-I, power and nonstationary null, with CIs.

Uses src/data_generators.py (gen_null, gen_and) and src/synergy_eprocess.py
(interaction_scores, score_bound, betting_eprocess, wald_peeking).

(a) n=4000, 2000 streams per scenario.
    null  : theta = 0 (gen_null).  Procedures: two-sided e-process (E = (E+ + E-)/2, reject
            if max E >= 20); peeking Wald z-test (|z| >= 1.96 at every t >= 30); Pocock and
            O'Brien-Fleming Lan-DeMets group-sequential Wald tests with 20 equally spaced looks
            (boundaries by Monte Carlo over 400k Brownian paths).
    and   : theta = 0.35 (gen_and), same procedures: power and median detection time.
(b) nonstationary nulls (2000 streams each, n=4000), always within H0: theta_t <= 0:
    switch : theta_t piecewise over 4 blocks of 1000, values (-0.2, 0, -0.2, 0)
    drift  : theta_t = linspace(-0.3, 0, 4000)
    One-sided e-process (E+ arm only, valid under theta_t <= 0) and two-sided average.
    The two-sided process is NOT under its null for these streams (theta_t != 0 on
    part of the path), so rejections are expected there and are reported as such.

Outputs: results/r11_interventional.json, results/r11_interventional.png,
results/r11_interventional_summary.md.  2 worker processes.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from common import rate_ci, wilson, RES  # noqa: E402
from gs_shared import gs_boundaries  # noqa: E402
from data_generators import gen_null, gen_and  # noqa: E402
from synergy_eprocess import interaction_scores, score_bound, betting_eprocess, wald_peeking  # noqa: E402

N = 4000
N_STREAMS = 2000
ALPHA = 0.05
LOOKS = 20
LOOK_N = np.arange(1, LOOKS + 1) * (N // LOOKS)        # 200, 400, ..., 4000
CUM_T = [50, 100, 200, 400, 800, 1600, 3200, 4000]
SEED = 11011
PAL = {"e2": "#4c72b0", "ep": "#55a868", "wald": "#c44e52", "gsP": "#8172b2", "gsO": "#dd8452"}
THETA_SWITCH = np.repeat([-0.2, 0.0, -0.2, 0.0], N // 4)
THETA_DRIFT = np.linspace(-0.3, 0.0, N)


def gen_theta(theta, rng, base=0.15, ea=0.25, eb=0.20):
    """Randomised 2x2 design with time-varying interaction theta_t:
    p = base + ea*A + eb*B + theta_t*A*B  ->  mu11-mu10-mu01+mu00 = theta_t exactly
    (all cell probabilities stay in [0,1] for theta in [-0.3, 0])."""
    n = len(theta)
    A = (rng.random(n) < 0.5).astype(int)
    B = (rng.random(n) < 0.5).astype(int)
    p = base + ea * A + eb * B + theta * A * B
    C = (rng.random(n) < p).astype(int)
    return A, B, C


def scenario_data(name, rng):
    if name == "null":
        d = gen_null(N, rng); return d["A"], d["B"], d["C"]
    if name == "and":
        d = gen_and(N, rng); return d["A"], d["B"], d["C"]
    if name == "switch":
        return gen_theta(THETA_SWITCH, rng)
    if name == "drift":
        return gen_theta(THETA_DRIFT, rng)
    raise ValueError(name)


SCEN_ID = {"null": 1, "and": 2, "switch": 3, "drift": 4}


def stream(args):
    name, rep = args
    rng = np.random.default_rng([SEED, SCEN_ID[name], rep])
    A, B, C = scenario_data(name, rng)
    psi = interaction_scores(A, B, C)
    E2, Ep, Em = betting_eprocess(psi, score_bound())
    thr = 1.0 / ALPHA
    first = lambda m: (int(np.argmax(m)) + 1) if m.any() else -1     # 1-based sample size, -1 = never
    wr = wald_peeking(A, B, C, alpha=ALPHA)
    z = wr["z"]
    z_look = z[LOOK_N - 1]
    return dict(e2_first=first(E2 >= thr), ep_first=first(Ep >= thr), em_first=first(Em >= thr),
                e2_final_log10=float(np.log10(max(E2[-1], 1e-300))),
                e2_max_log10=float(np.log10(max(E2.max(), 1e-300))),
                ep_max_log10=float(np.log10(max(Ep.max(), 1e-300))),
                wald_first=first(z >= wr["zcrit"]), z_look=z_look.tolist(),
                ep_final_log10=float(np.log10(max(Ep[-1], 1e-300))))


def cum_fp_curve(first_times, n_tot=N):
    """Fraction of streams rejected by time t (t=1..n_tot), with Wilson bands."""
    f = np.asarray(first_times)
    N_s = len(f)
    cnt = np.array([np.sum((f > 0) & (f <= t)) for t in range(1, n_tot + 1)], float)
    p = cnt / N_s
    z = 1.96
    den = 1 + z * z / N_s
    c = (p + z * z / (2 * N_s)) / den
    h = z * np.sqrt(p * (1 - p) / N_s + z * z / (4 * N_s * N_s)) / den
    return p, np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def summarize_rejects(first, key_name):
    first = np.asarray(first)
    rej = first > 0
    r = rate_ci(rej)
    det = first[rej]
    r["median_detect_n"] = float(np.median(det)) if len(det) else None
    return r


def main():
    np.seterr(all="ignore")
    t0 = time.time()
    scen = ["null", "and", "switch", "drift"]
    tasks = [(s, r) for s in scen for r in range(N_STREAMS)]
    res_by = {s: [] for s in scen}
    with Pool(2) as pool:
        for i, (s, out) in enumerate(zip([t[0] for t in tasks],
                                         pool.imap(stream, tasks, chunksize=20))):
            res_by[s].append(out)
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(tasks)} streams  t={time.time()-t0:.0f}s", flush=True)

    # group-sequential boundaries (Monte Carlo, 400k paths, two-sided)
    info = LOOK_N / N
    cP, mcP = gs_boundaries(info, "pocock", ALPHA, sided="two", n_paths=400000, seed=777)
    cO, mcO = gs_boundaries(info, "obf", ALPHA, sided="two", n_paths=400000, seed=777)

    def gs_rejects(rows, c):
        Z = np.array([r["z_look"] for r in rows])
        hit = Z >= c[None, :]
        first = np.where(hit.any(axis=1), LOOK_N[np.argmax(hit, axis=1)], -1)
        return first

    out = dict(script="r11_interventional.py", N=N, n_streams=N_STREAMS, alpha=ALPHA, seed=SEED,
               settings=dict(dgp_null="gen_null (theta=0, base .15, ea .25, eb .20)",
                             dgp_and="gen_and (syn=0.35, theta=0.35)",
                             dgp_switch="theta_t in blocks of 1000: (-0.2, 0, -0.2, 0), same cell structure",
                             dgp_drift="theta_t = linspace(-0.3, 0, 4000)",
                             e_process="data_generators-compatible synergy_eprocess.interaction_scores (piA=piB=.5), "
                                       "score_bound()=4, betting_eprocess(warmup=20, cap_frac=0.9); two-sided E=(E+ + E-)/2; reject if max E>=20",
                             one_sided="E+ arm only, reject if max E+>=20",
                             wald_peeking="synergy_eprocess.wald_peeking (IPW estimate, running sd, |z|>=1.96 at every t>=30)",
                             group_sequential=f"Wald |z| at {LOOKS} equally spaced looks (n=200..4000); boundaries by Lan-DeMets spending "
                                              f"(Pocock, OBF) via Monte Carlo over 400k Brownian paths (seed 777); MC total alpha reported",
                             seeds=f"default_rng([{SEED}, scenario_id, stream])"),
               sample=dict(streams_per_scenario=N_STREAMS, n=N),
               gs_boundaries=dict(looks_n=LOOK_N.tolist(), pocock=[float(x) for x in cP], obf=[float(x) for x in cO],
                                  mc_total_alpha_pocock=mcP, mc_total_alpha_obf=mcO,
                                  note="inf = no rejection possible at that look (zero spent alpha)"))

    # ---- (a) null and AND
    part_a = {}
    for s in ["null", "and"]:
        rows = res_by[s]
        e2 = np.array([r["e2_first"] for r in rows]); ep = np.array([r["ep_first"] for r in rows])
        wa = np.array([r["wald_first"] for r in rows])
        gP = gs_rejects(rows, cP); gO = gs_rejects(rows, cO)
        cum = {}
        pw, pwl, pwh = cum_fp_curve(wa)
        pe, pel, peh = cum_fp_curve(e2)
        for t in CUM_T:
            i = t - 1
            cum[str(t)] = dict(wald_peeking=dict(rate=float(pw[i]), lo=float(pwl[i]), hi=float(pwh[i])),
                               e_process_two_sided=dict(rate=float(pe[i]), lo=float(pel[i]), hi=float(peh[i])))
        part_a[s] = dict(
            e_two_sided=summarize_rejects(e2, "e2"),
            e_one_sided=summarize_rejects(ep, "ep"),
            e_two_sided_median_final_log10=float(np.median([r["e2_final_log10"] for r in rows])),
            e_two_sided_median_max_log10=float(np.median([r["e2_max_log10"] for r in rows])),
            wald_peeking=summarize_rejects(wa, "wald"),
            gs_pocock=summarize_rejects(gP, "gsP"),
            gs_obf=summarize_rejects(gO, "gsO"),
            cumulative_false_positive=cum,
        )
    out["partA"] = part_a

    # ---- (b) nonstationary nulls
    part_b = {}
    for s in ["switch", "drift"]:
        rows = res_by[s]
        e2 = np.array([r["e2_first"] for r in rows]); ep = np.array([r["ep_first"] for r in rows])
        part_b[s] = dict(
            e_one_sided_E_plus=summarize_rejects(ep, "ep"),
            e_two_sided=summarize_rejects(e2, "e2"),
            two_sided_note="theta_t != 0 on part of the path: two-sided null is false by construction; rejections expected",
            median_final_log10_E_plus=float(np.median([r["ep_final_log10"] for r in rows])),
            median_final_log10_E_two_sided=float(np.median([r["e2_final_log10"] for r in rows])),
        )
    out["partB"] = part_b
    out["runtime_s"] = time.time() - t0

    json.dump(out, open(os.path.join(RES, "r11_interventional.json"), "w"), indent=2, default=str)

    # ---- figure: cumulative false-positive curves (a) and nonstationary (b)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    t = np.arange(1, N + 1)
    ax = axes[0]
    for key, col, lab, arr in [("wald", PAL["wald"], "peeking Wald |z|>=1.96", np.array([r["wald_first"] for r in res_by["null"]])),
                               ("e2", PAL["e2"], "two-sided e-process (max E>=20)", np.array([r["e2_first"] for r in res_by["null"]]))]:
        p, lo, hi = cum_fp_curve(arr)
        ax.plot(t, p, color=col, label=lab)
        ax.fill_between(t, lo, hi, color=col, alpha=0.2)
    ax.axhline(ALPHA, color="k", ls=":", lw=1, label="alpha = 0.05")
    ax.set_xscale("log"); ax.set_xlabel("time t (observations)"); ax.set_ylabel("P(rejected by t) under null")
    ax.set_title("Null (theta=0): cumulative false positives, 2000 streams"); ax.legend(fontsize=8)
    ax = axes[1]
    for s, ls in [("switch", "-"), ("drift", "--")]:
        for key, col, lab in [("ep", PAL["ep"], "one-sided E+"), ("e2", PAL["e2"], "two-sided")]:
            arr = np.array([r[f"{key}_first"] for r in res_by[s]])
            p, lo, hi = cum_fp_curve(arr)
            ax.plot(t, p, ls, color=col, label=f"{s}: {lab}")
            ax.fill_between(t, lo, hi, color=col, alpha=0.12)
    ax.axhline(ALPHA, color="k", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("time t (observations)"); ax.set_ylabel("P(rejected by t)")
    ax.set_title("Nonstationary nulls (theta_t <= 0)"); ax.legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "r11_interventional.png"), dpi=200); plt.close()

    # ---- summary
    def fmt(r):
        return f"{r['k']}/{r['n']} = {r['rate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
    L = ["# R11 summary: interventional Type-I and power (n=4000, 2000 streams)\n",
         "## (a) null (theta=0) and AND (theta=0.35)\n",
         "| procedure | null rejections | AND power | AND median detection n |",
         "|---|---|---|---|"]
    names = dict(e_two_sided="two-sided e-process", wald_peeking="peeking Wald (every t)",
                 gs_pocock="GS Pocock (20 looks)", gs_obf="GS OBF (20 looks)", e_one_sided="one-sided E+")
    for k, nm in names.items():
        nr = part_a["null"][k]; ar = part_a["and"][k] if k in part_a["and"] else None
        L.append(f"| {nm} | {fmt(nr)} | {fmt(ar) if ar else 'n/a'} | "
                 f"{ar['median_detect_n'] if ar else ''} |")
    L.append(f"\nGS boundaries (two-sided, look n -> Pocock, OBF):")
    for i, nn in enumerate(LOOK_N):
        L.append(f"- n={nn}: {cP[i]:.3f}, {cO[i]:.3f}")
    L.append(f"\nMC total alpha: Pocock {mcP:.4f}, OBF {mcO:.4f}\n")
    L.append("Cumulative false positives (null), Wilson CI:\n")
    L.append("| t | peeking Wald | two-sided e-process |")
    L.append("|---|---|---|")
    for t_ in CUM_T:
        c = part_a["null"]["cumulative_false_positive"][str(t_)]
        w = c["wald_peeking"]; e = c["e_process_two_sided"]
        L.append(f"| {t_} | {w['rate']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}] | {e['rate']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}] |")
    L.append("\n## (b) nonstationary nulls (theta_t <= 0)\n")
    L.append("| scenario | one-sided E+ rejections | two-sided rejections (null false by construction) |")
    L.append("|---|---|---|")
    for s in ["switch", "drift"]:
        L.append(f"| {s} | {fmt(part_b[s]['e_one_sided_E_plus'])} | {fmt(part_b[s]['e_two_sided'])} |")
    open(os.path.join(RES, "r11_interventional_summary.md"), "w").write("\n".join(L) + "\n")
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
