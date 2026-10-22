#!/usr/bin/env python3
"""R3: nonlinear dictionary extension via random Fourier features (RFF).

Compares the default polynomial dictionary ("poly": base, own linear+squared
lags, cross-products in the joint model) with "poly+RFF", where the `extra`
hook of ``ante_v2.make_features`` adds

  * add-only features : cos(w_a*A_{t-1}+b_a), cos(w_b*B_{t-1}+b_b)   (2q features)
  * joint-only features: cos(W_ab[0]*A_{t-1}+W_ab[1]*B_{t-1}+b_ab)  (q features)

with W_a, W_b ~ N(0,1)^q, W_ab ~ N(0,1)^{2 x q}, biases ~ U(0,2pi), q=15,
drawn fresh for every replication from the replication's seed.

Test: v2 addjoint score, forget=0.995, p=1, cond=[Z], cap=3, gamma=0.9,
warm-up 50, reject when max_E >= 1/alpha = 20.
DGPs: null (additive), syn (bilinear, strength 0.9), a2b (0.9), tanh (1.5),
sinab (1.5); rho=0.2; n_rep=100; T=2500 for all, and T=5000 for a2b, tanh, sinab.

Outputs (experiments/revision/results/):
  r3_nonlinear.json, r3_nonlinear_summary.md, r3_nonlinear.png
"""
import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
np.seterr(all="ignore")

from common import gen_var, rate_ci, RES  # noqa: E402
from ante_v2 import (run_errors, bounded_score, normalise, bet_process,  # noqa: E402
                     make_features, MODELS)

N_REP = 100
N_PROC = 2
RHO = 0.2
FORGET = 0.995
RIDGE = 1e-2
CAP = 3.0
GAMMA = 0.9
WARMUP = 50
ALPHA = 0.05
Q = 15
THR = 1.0 / ALPHA

# (dgp, strength, T, seed)
CONFIGS = [
    ("null", 0.0, 2500, 301),
    ("syn", 0.9, 2500, 302),
    ("a2b", 0.9, 2500, 303),
    ("tanh", 1.5, 2500, 304),
    ("sinab", 1.5, 2500, 305),
    ("a2b", 0.9, 5000, 306),
    ("tanh", 1.5, 5000, 307),
    ("sinab", 1.5, 5000, 308),
]


# ---------------------------------------------------------------- RFF hook
def make_rff(rng, q=Q):
    """Return an `extra(t, la, la_B)` hook for ante_v2.make_features."""
    Wa = rng.standard_normal(q)
    Wb = rng.standard_normal(q)
    Wab = rng.standard_normal((2, q))
    ba = rng.uniform(0, 2 * np.pi, q)
    bb = rng.uniform(0, 2 * np.pi, q)
    bab = rng.uniform(0, 2 * np.pi, q)

    def extra(t, la, lb):
        a = la[0]  # first lag of A (p=1, no contemporaneous term)
        b = lb[0]  # first lag of B
        add_extra = np.concatenate([np.cos(Wa * a + ba), np.cos(Wb * b + bb)])
        joint_extra = np.cos(Wab[0] * a + Wab[1] * b + bab)
        return add_extra.tolist(), joint_extra.tolist()

    return extra


def hook_selftest():
    """Check that the RFF hook is wired into make_features as intended."""
    rng = np.random.default_rng(12345)
    T = 50
    Y, A, B, Z = gen_var("null", T, rng)
    extra = make_rff(np.random.default_rng(7))
    f_plain = make_features(Y, A, B, [Z], 1, extra=None)(5)
    f_rff = make_features(Y, A, B, [Z], 1, extra=extra)(5)
    n_add_plain = len(f_plain["add"])
    n_joint_plain = len(f_plain["joint"])
    assert len(f_rff["add"]) == n_add_plain + 2 * Q, "add dimension mismatch"
    # joint = add + cross + joint_extra, so it carries the 2q add-RFF plus q joint-RFF
    assert len(f_rff["joint"]) == n_joint_plain + 3 * Q, "joint dimension mismatch"
    assert len(f_rff["A"]) == len(f_plain["A"]) and len(f_rff["B"]) == len(f_plain["B"])
    # the RFF block must use la[0], lb[0] = A_{t-1}, B_{t-1}
    a1, b1 = A[4], B[4]
    chk = make_rff(np.random.default_rng(7))(5, [a1], [b1])
    assert np.allclose(f_rff["add"][n_add_plain:], np.asarray(chk[0]))
    assert np.allclose(f_rff["joint"][n_add_plain:n_add_plain + 2 * Q], np.asarray(chk[0]))
    assert np.allclose(f_rff["joint"][-Q:], np.asarray(chk[1]))
    return dict(add_dim_plain=n_add_plain, add_dim_rff=len(f_rff["add"]),
                joint_dim_expected_extra=3 * Q,
                joint_dim_plain=n_joint_plain, joint_dim_rff=len(f_rff["joint"]),
                status="ok")


def signal_share(dgp, strength, T=20000, seed=99):
    """Share of Var(Y) carried by the non-linear interaction term of each DGP
    (the term that the linear dictionary cannot represent). Mirrors gen_var."""
    rng = np.random.default_rng(seed)
    Y, A, B, Z = gen_var(dgp, T, rng, rho=RHO, strength=strength)
    a, b = A[:-1], B[:-1]
    if dgp == "syn":
        nl = strength * a * b
    elif dgp == "a2b":
        nl = strength * a * a * b
    elif dgp == "tanh":
        nl = strength * np.tanh(2 * a) * np.tanh(2 * b)
    elif dgp == "sinab":
        nl = strength * np.sin(a * b)
    else:
        return 0.0
    return float(np.var(nl) / np.var(Y[1:]))


# ---------------------------------------------------------------- one replication
def one_rep(task):
    dgp, strength, T, seed, rep = task
    rng = np.random.default_rng([seed, rep])
    Y, A, B, Z = gen_var(dgp, T, rng, rho=RHO, strength=strength)
    extra = make_rff(rng)  # fixed per replication
    out = dict(rep=rep)
    for name, ex in (("poly", None), ("polyrff", extra)):
        pred, _ = run_errors(Y, A, B, cond=[Z], p=1, forget=FORGET, ridge=RIDGE,
                             feat_clip=None, extra=ex, models=MODELS)
        s, b = bounded_score(Y, pred, MODELS, "addjoint", CAP)
        u = normalise(s, b)
        E, lam = bet_process(u, GAMMA, WARMUP, start=1)
        out[name] = dict(final_log10E=float(np.log10(max(E[-1], 1e-300))),
                         max_log10E=float(np.log10(max(E.max(), 1e-300))),
                         reject=bool(E.max() >= THR),
                         mean_s=float(np.mean(s[1:])))
    return out


def summarise(rows, key):
    rej = [r[key]["reject"] for r in rows]
    ci = rate_ci(rej)
    fin = np.array([r[key]["final_log10E"] for r in rows])
    mx = np.array([r[key]["max_log10E"] for r in rows])
    ms = np.array([r[key]["mean_s"] for r in rows])
    return dict(power=ci["rate"], k=ci["k"], n=ci["n"], ci_lo=ci["lo"], ci_hi=ci["hi"],
                median_final_log10E=float(np.median(fin)),
                median_max_log10E=float(np.median(mx)),
                mean_s=float(np.mean(ms)))


def main():
    t0 = time.time()
    print("self-test:", flush=True)
    st = hook_selftest()
    print(" ", st, flush=True)

    tasks = []
    for dgp, strength, T, seed in CONFIGS:
        for rep in range(N_REP):
            tasks.append((dgp, strength, T, seed, rep))

    results = {}
    done = 0
    with Pool(N_PROC) as pool:
        for task, out in zip(tasks, pool.imap(one_rep, tasks, chunksize=2)):
            key = (task[0], task[2])
            results.setdefault(key, []).append(out)
            done += 1
            if done % 50 == 0:
                print(f"  progress {done}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)

    table = []
    for dgp, strength, T, seed in CONFIGS:
        rows = results[(dgp, T)]
        entry = dict(dgp=dgp, strength=strength, T=T, seed=seed, n_rep=len(rows),
                     signal_share_of_var=signal_share(dgp, strength),
                     poly=summarise(rows, "poly"), polyrff=summarise(rows, "polyrff"))
        table.append(entry)
        print(f"{dgp:6s} T={T}: poly power={entry['poly']['power']:.2f} "
              f"polyRFF power={entry['polyrff']['power']:.2f}", flush=True)

    null_rff = [e for e in table if e["dgp"] == "null"][0]["polyrff"]
    null_typeI_ok = null_rff["power"] <= ALPHA
    settings = dict(n_rep=N_REP, rho=RHO, forget=FORGET, ridge=RIDGE, cap=CAP, gamma=GAMMA,
                    warmup=WARMUP, alpha=ALPHA, threshold_max_E=THR, kind="addjoint",
                    p=1, cond="[Z]", feat_clip=None, rff_q=Q,
                    rff_construction="add: cos(w_a*A_{t-1}+b_a), cos(w_b*B_{t-1}+b_b); "
                                     "joint: cos(W_ab @ (A_{t-1},B_{t-1})+b_ab); "
                                     "w ~ N(0,1), b ~ U(0,2pi), fresh per replication",
                    strengths_mapping="syn 0.9, a2b 0.9, tanh 1.5, sinab 1.5 (null has no strength)",
                    seeds="config seed, replication r uses default_rng([seed, r]) for data and RFF",
                    paired_design="poly and poly+RFF share the same simulated data and the same replication",
                    interpretation_note="Type-I check uses null rows; reject = max_E >= 20")
    out = dict(settings=settings, hook_selftest=st, table=table,
               null_polyrff_typeI_ok=bool(null_typeI_ok),
               runtime_s=float(time.time() - t0))
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "r3_nonlinear.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- markdown summary
    lines = ["# R3 summary: poly vs poly+RFF (addjoint, forget=0.995, n_rep=%d)" % N_REP, "",
             "| DGP | T | strength | signal share of Var(Y) | poly power [95% CI] | poly+RFF power [95% CI] | "
             "poly median log10 max E | poly+RFF median log10 max E | poly median final log10 E | poly+RFF median final log10 E |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in table:
        p, r = e["poly"], e["polyrff"]
        lines.append(f"| {e['dgp']} | {e['T']} | {e['strength']} | {e['signal_share_of_var']:.3f} | "
                     f"{p['power']:.2f} [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}] | "
                     f"{r['power']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
                     f"{p['median_max_log10E']:.2f} | {r['median_max_log10E']:.2f} | "
                     f"{p['median_final_log10E']:.2f} | {r['median_final_log10E']:.2f} |")
    lines += ["", f"Type-I of poly+RFF on the null (T=2500): {null_rff['power']:.3f} "
              f"[{null_rff['ci_lo']:.3f}, {null_rff['ci_hi']:.3f}]; "
              f"must be <= 0.05: {'yes' if null_rff['power'] <= ALPHA else 'NO'}",
              f"Runtime: {out['runtime_s']:.0f}s"]
    with open(os.path.join(RES, "r3_nonlinear_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure: grouped bars with Wilson CIs
    import matplotlib.pyplot as plt  # noqa: F811
    panels = [[e for e in table if e["T"] == 2500], [e for e in table if e["T"] == 5000]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[5, 3]))
    cols = {"poly": "#4c72b0", "polyrff": "#dd8452"}
    for ax, group, title in zip(axes, panels, ["T = 2500", "T = 5000"]):
        x = np.arange(len(group))
        w = 0.38
        for j, key in enumerate(["poly", "polyrff"]):
            pw = np.array([e[key]["power"] for e in group])
            lo = pw - np.array([e[key]["ci_lo"] for e in group])
            hi = np.array([e[key]["ci_hi"] for e in group]) - pw
            ax.bar(x + (j - 0.5) * w, pw, w, yerr=[lo, hi], capsize=3,
                   color=cols[key], edgecolor="black", lw=0.5,
                   label="poly" if key == "poly" else "poly + RFF")
        ax.axhline(ALPHA, color="red", ls="--", lw=1, label=r"$\alpha = 0.05$")
        ax.set_xticks(x)
        ax.set_xticklabels([e["dgp"] for e in group])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("rejection rate (max E >= 20)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("R3: polynomial vs polynomial + random Fourier features (addjoint, n_rep=%d)" % N_REP)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "r3_nonlinear.png"), dpi=200)
    print(f"done in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
