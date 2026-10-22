"""R1: Type-I stress tests of the exactly-valid score.

(a) Toy counterexample: i.i.d. score s_t = -20 w.p. 0.1, +1.5 w.p. 0.9 (E s = -0.65 <= 0),
    legacy-style clipped bettor, T=2000, 2000 replications.  The unclipped bounded
    alternative (v2) is not applicable here because the raw score is not a bounded
    functional of a squared-error loss pair; it is therefore not run on this toy.
(b) VAR nulls (additive or no-synergy), n_rep=200 each, scored by v2 addjoint, v2 four
    and the legacy ante_sg test.
(c) Table JSON, markdown summary and a bar chart of rejection rates with Wilson CIs.

Run from the repository directory:  python3 experiments/revision/r1_stress_typeI.py
"""
import os
import sys
import json
import time

import numpy as np

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from multiprocessing import Pool  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import rate_ci, gen_var, RES  # noqa: E402
from ante_v2 import run_errors, bounded_score, normalise, bet_process, MODELS  # noqa: E402
from synergistic_granger import ante_sg  # noqa: E402

SCRIPT = "r1_stress_typeI"
ALPHA = 0.05
THR = 1.0 / ALPHA            # max_E >= 20 <=> rejection at alpha = .05
T = 2500
N_REP = 200
CAP = 3.0
GAMMA = 0.9
WARM = 50
TOY_T = 2000
TOY_REP = 2000
METHODS = ("addjoint", "four", "legacy")

NULLS = [
    dict(id="N1", label="gaussian additive rho=.2", kind="null", rho=0.2),
    dict(id="N2", label="heavy t3 + skew additive rho=.2", kind="null", rho=0.2, heavy=True, skew=True),
    dict(id="N3", label="heavy + garch additive rho=.2", kind="null", rho=0.2, heavy=True, garch=True),
    dict(id="N4", label="additive rho=.95", kind="null", rho=0.95),
    dict(id="N5", label="additive rho=.99", kind="null", rho=0.99),
    dict(id="N6", label="cubic additive rho=.5", kind="cubic", rho=0.5),
    dict(id="N7", label="cubic rho=.9", kind="cubic", rho=0.9),
    dict(id="N8", label="break (sign flip at T/2) rho=.2", kind="break", rho=0.2),
    dict(id="N9", label="own nonlinearity rho=.5", kind="own", rho=0.5),
    dict(id="N10", label="additive ar=0.9 sources rho=.5", kind="null", rho=0.5, ar=0.9),
    dict(id="N11", label="additive forget=0.98", kind="null", rho=0.2, forget=0.98),
    dict(id="N12", label="additive forget=0.999", kind="null", rho=0.2, forget=0.999),
]


def clean(o):
    """Recursively convert numpy types to JSON-safe Python types (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else None
    return o


# ---------------------------------------------------------------- part (a)
def toy_counterexample(n_rep=TOY_REP, T_toy=TOY_T, seed=12345, sigma_mode="adaptive"):
    """Vectorised over replications.  sigma_mode='adaptive': sigma_hat_t = expanding rms
    of past s (sigma_hat_0 = 1).  sigma_mode='fixed': sigma_hat = 1 for all t (reviewer's
    example): u = clip(s, -5, 5) has E[u] = 0.1*(-5) + 0.9*1.5 = +0.85 > 0 although E[s] < 0."""
    assert sigma_mode in ("adaptive", "fixed")
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((n_rep, T_toy)) < 0.1, -20.0, 1.5)
    E = np.ones(n_rep)
    Emax = np.ones(n_rep)
    s1 = np.zeros(n_rep)
    s2 = np.zeros(n_rep)
    sumsq = np.zeros(n_rep)          # sum of past s^2 for expanding rms
    lam_pos = np.zeros(n_rep)
    cnt = 0
    for t in range(T_toy):
        if sigma_mode == "fixed" or t == 0:
            sigma = np.ones(n_rep)                                 # sigma_hat = 1 (fixed)
        else:
            sigma = np.sqrt(sumsq / t)                             # expanding rms of past s
        u = np.clip(s[:, t] / sigma, -5.0, 5.0)
        if cnt >= WARM:
            mu = s1 / cnt
            v = s2 / cnt + 1e-12
            lam = np.clip(np.maximum(mu / v, 0.0), 0.0, 0.9 / 5.0)
        else:
            lam = np.zeros(n_rep)
        E = E * (1.0 + lam * u)
        Emax = np.maximum(Emax, E)
        lam_pos += (lam > 0)
        s1 += u
        s2 += u * u
        sumsq += s[:, t] ** 2
        cnt += 1
    flags = Emax >= THR
    d = rate_ci(flags)
    d.update(median_log10_final=float(np.median(np.log10(E))),
             median_log10_max=float(np.median(np.log10(Emax))),
             mean_lambda_pos_frac=float(np.mean(lam_pos / T_toy)),
             mean_raw_s=float(np.mean(s)), population_E_s=float(-20 * 0.1 + 1.5 * 0.9),
             population_E_u=(0.85 if sigma_mode == "fixed" else None),
             sigma_mode=sigma_mode, n_rep=n_rep, T=T_toy, seed=seed)
    return d


# ---------------------------------------------------------------- part (b)
def rep_task(args):
    cfg, seed = args
    rng = np.random.default_rng(seed)
    kw = {k: cfg[k] for k in ("rho", "heavy", "skew", "garch", "ar") if k in cfg}
    Y, A, B, Z = gen_var(cfg["kind"], T, rng, **kw)
    forget = cfg.get("forget", 0.995)
    pred, _ = run_errors(Y, A, B, [Z], p=1, forget=forget)
    out = {}
    for kind in ("addjoint", "four"):
        s, b = bounded_score(Y, pred, MODELS, kind, cap=CAP)
        u = normalise(s, b)
        E, lam = bet_process(u, GAMMA, WARM, start=1)
        out[kind] = dict(final=float(E[-1]), maxE=float(E.max()),
                         lam_frac=float(np.mean(lam[1:] > 0)), mean_s=float(np.mean(s[1:])))
    leg = ante_sg(Y, A, B, cond=[Z], p=1, alpha=ALPHA, forget=forget)
    out["legacy"] = dict(final=float(leg["final_E"]), maxE=float(np.max(leg["E"])),
                         lam_frac=float("nan"), mean_s=float(leg["mean_score"]))
    return out


def aggregate(rows):
    res = {}
    for m in METHODS:
        fin = np.array([r[m]["final"] for r in rows], float)
        mx = np.array([r[m]["maxE"] for r in rows], float)
        lf = np.array([r[m]["lam_frac"] for r in rows], float)
        ms = np.array([r[m]["mean_s"] for r in rows], float)
        d = rate_ci(mx >= THR)
        d.update(median_log10_final=float(np.median(np.log10(fin))),
                 median_log10_max=float(np.median(np.log10(mx))),
                 mean_lambda_pos_frac=(float(np.mean(lf)) if np.isfinite(lf).any() else None),
                 mean_raw_s=float(np.mean(ms)))
        res[m] = d
    return res


def main():
    t0 = time.time()
    print("R1 part (a): toy counterexample", flush=True)
    toy = toy_counterexample(sigma_mode="adaptive")
    toy_fixed = toy_counterexample(sigma_mode="fixed")
    for name, tv in (("adaptive", toy), ("fixed", toy_fixed)):
        print(f"  toy[{name}] P(max E>=20)={tv['rate']:.4f} CI=[{tv['lo']:.4f},{tv['hi']:.4f}] "
              f"median log10 final E={tv['median_log10_final']:.3f}", flush=True)

    print(f"R1 part (b): {len(NULLS)} VAR nulls x {N_REP} reps, 2 workers", flush=True)
    tasks = []
    for ci, cfg in enumerate(NULLS):
        for r in range(N_REP):
            tasks.append((cfg, 100000 * (ci + 1) + r))
    results = {cfg["id"]: [] for cfg in NULLS}
    done = 0
    with Pool(2) as pool:
        for (cfg, _), out in zip(tasks, pool.imap(rep_task, tasks, chunksize=4)):
            results[cfg["id"]].append(out)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  progress {done}/{len(tasks)}  elapsed {time.time() - t0:.0f}s", flush=True)

    null_tables = []
    for cfg in NULLS:
        agg = aggregate(results[cfg["id"]])
        null_tables.append(dict(id=cfg["id"], label=cfg["label"],
                                settings={k: v for k, v in cfg.items() if k not in ("id", "label")},
                                n_rep=N_REP, methods=agg))
        print(f"  {cfg['id']} {cfg['label']}: " + " | ".join(
            f"{m} {agg[m]['rate']:.3f} [{agg[m]['lo']:.3f},{agg[m]['hi']:.3f}]" for m in METHODS), flush=True)

    runtime = time.time() - t0
    out = dict(script=SCRIPT, n_rep=N_REP, T=T, alpha=ALPHA, rejection_rule="max_E >= 1/alpha = 20",
               settings=dict(gamma=GAMMA, warmup=WARM, cap=CAP, forget_default=0.995, p=1, cond="[Z]",
                             feat_clip=None, seeds="100000*(config index+1)+rep",
                             legacy="synergistic_granger.ante_sg defaults (warmup=60, clip_b=10, cap_frac=0.9)",
                             notes="N2, N3 use default rho=.2 (not specified in spec). "
                                   "mean_lambda_pos_frac is not exposed by ante_sg for legacy (None)."),
               toy_counterexample={"adaptive_sigma": toy, "fixed_sigma_1": toy_fixed},
               nulls=null_tables, runtime_sec=runtime)
    with open(os.path.join(RES, f"{SCRIPT}.json"), "w") as fh:
        json.dump(clean(out), fh, indent=2)

    # ------------------------------------------------ summary markdown
    L = [f"# R1 Type-I stress tests (auto-generated)", "",
         f"n_rep = {N_REP} per null (T={T}), alpha = {ALPHA}, rejection = max E >= 20.", "",
         "## (a) Toy counterexample (legacy-style clipped bettor)", "",
         "| sigma_hat | n_rep | T | P(max E>=20) | 95% Wilson CI | median log10 final E | median log10 max E | E[s] | E[u] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, tv in (("adaptive (expanding rms)", toy), ("fixed = 1 (reviewer example)", toy_fixed)):
        eu = "n/a" if tv["population_E_u"] is None else f"{tv['population_E_u']:.2f}"
        L.append(f"| {name} | {tv['n']} | {TOY_T} | {tv['rate']:.4f} | [{tv['lo']:.4f}, {tv['hi']:.4f}] | "
                 f"{tv['median_log10_final']:.3f} | {tv['median_log10_max']:.3f} | "
                 f"{tv['population_E_s']:.2f} | {eu} |")
    L += ["", "Clipping at -5 with sigma=1 makes E[u] > 0 although E[s] < 0, so the fixed-sigma bettor is a genuine "
          "Type-I failure of the clipped score; the adaptive version normalises the -20 outliers and does not grow. "
          "The unclipped bounded alternative is not applicable to this toy score.", "",
         "## (b) VAR nulls", "",
         "| id | DGP | test | rejection | 95% Wilson CI | median log10 final E | median log10 max E | mean lambda>0 | mean raw s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nt in null_tables:
        for m in METHODS:
            d = nt["methods"][m]
            lf = "n/a" if d["mean_lambda_pos_frac"] is None else f"{d['mean_lambda_pos_frac']:.3f}"
            L.append(f"| {nt['id']} | {nt['label']} | {m} | {d['rate']:.3f} ({d['k']}/{d['n']}) | "
                     f"[{d['lo']:.3f}, {d['hi']:.3f}] | {d['median_log10_final']:.3f} | "
                     f"{d['median_log10_max']:.3f} | {lf} | {d['mean_raw_s']:.4f} |")
    L += ["", f"Runtime: {runtime / 60:.1f} min. Figure: `{os.path.join(RES, SCRIPT + '.png')}`", ""]
    with open(os.path.join(RES, f"{SCRIPT}_summary.md"), "w") as fh:
        fh.write("\n".join(L))

    # ------------------------------------------------ figure
    x = np.arange(len(NULLS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = {"addjoint": "#1f77b4", "four": "#ff7f0e", "legacy": "#7f7f7f"}
    for j, m in enumerate(METHODS):
        rates = np.array([nt["methods"][m]["rate"] for nt in null_tables])
        lo = np.array([nt["methods"][m]["lo"] for nt in null_tables])
        hi = np.array([nt["methods"][m]["hi"] for nt in null_tables])
        ax.bar(x + (j - 1) * w, rates, w, color=colors[m], label=m,
               yerr=[np.clip(rates - lo, 0, None), np.clip(hi - rates, 0, None)],
               capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(ALPHA, color="red", ls="--", lw=1, label="alpha = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([nt["id"] for nt in null_tables])
    ax.set_ylabel("P(max E >= 20)  (Type-I rate)")
    ax.set_ylim(0, max(0.12, min(1.0, max(nt["methods"][m]["hi"] for nt in null_tables for m in METHODS) + 0.02)))
    ax.set_title(f"R1: Type-I rate under VAR nulls (n_rep={N_REP}, Wilson 95% CI)")
    ax.legend(ncol=4, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"{SCRIPT}.png"), dpi=200)
    print(f"Done in {runtime / 60:.1f} min. JSON: {os.path.join(RES, SCRIPT + '.json')}", flush=True)


if __name__ == "__main__":
    main()
