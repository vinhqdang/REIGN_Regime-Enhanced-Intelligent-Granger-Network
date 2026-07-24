# ANTE: Anytime-valid Nonparametric Test of synErgy

*Sequential, anytime-valid testing for group (synergistic) causality.*

A betting-martingale (e-process) test for **group causality** — the hypothesis
that a set of causes acts on an effect *jointly*, beyond the sum of their
individual contributions — monitored **anytime-valid** as data streams in.

The three existing literatures on group causality — sufficient-cause interaction
(RERI/PRISM/generalized synergy index), joint-effect estimation (causal
aggregation, joint potential outcomes), and information-theoretic decomposition
(PID/SURD/PEID) — are **all fixed-sample**. None offers a sequential test you can
monitor continuously and stop as soon as evidence is decisive. This repo
prototypes that missing piece.

## The idea in one paragraph

For two binary causes $A,B$ and effect $C$, the additive interaction contrast
$\theta = \mu_{11}-\mu_{10}-\mu_{01}+\mu_{00}$ is zero exactly when the joint
effect equals the sum of parts (no synergy). Under a known/randomized design the
IPW interaction score $\psi_t = c(A_t,B_t)\,C_t/\pi(A_t,B_t)$ is bounded and has
mean $\theta$, so under $H_0$ it is mean-zero. Betting on the stream,
$E_t = \prod_{s\le t}(1+\lambda_s\psi_s)$ with predictable $\lambda_s$, gives a
nonnegative martingale with $E[E_t]=1$ under $H_0$; **Ville's inequality** then
guarantees $P_{H_0}(\exists t: E_t\ge 1/\alpha)\le\alpha$. Reject the first time
capital crosses $1/\alpha$ — valid under continuous monitoring and optional
stopping. See [`docs/methodology.md`](docs/methodology.md).

## Results (reproducible via `experiments/run_experiments.py`)

| Experiment | Finding |
|---|---|
| **E1 — type-I control** | e-process false-positive rate **0.031** ≤ α=0.05, uniformly over the stream; a fixed-$n$ Wald test monitored continuously inflates to **0.505** (10×). |
| **E2 — sample paths** | Under the null, capital stays flat (0/12 crossings); under AND and XOR synergy every path crosses $1/\alpha$. |
| **E3 — power / latency** | Power rises monotonically with synergy strength (0.02 → 1.0); detection time shrinks as synergy grows. |
| **E4 — subset refinement** | Among three candidate causes with a synergistic (A,B) pair, the search fires on **A-B at 1.00** and on the innocent pairs at ≤ α (A-D 0.00, B-D 0.005). |

Figures: `experiments/results/e{1,2,3,4}_*.png`. Raw numbers: `experiments/results/results.json`.

## Layout

```
src/
  data_generators.py     # null / AND / XOR / 3-var synergistic-triple streams
  synergy_eprocess.py    # IPW interaction score, betting e-process, Wald baseline, subset refinement
experiments/
  run_experiments.py     # E1–E4, saves JSON + PNG
  results/               # outputs
docs/
  methodology.md         # test statistic + martingale construction + extensions
  literature_review.md   # verified citations + gap analysis
```

## Run

```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

## Status / next steps

Prototype supporting a methods paper. Natural extensions (sketched in
`docs/methodology.md`): the doubly-robust/AIPW observational version with
cross-fitted nuisances; the information-theoretic (PEID-scale) e-process via
permutation-recentred effective-information increments; and the graph/GNN
instantiation testing joint multi-node causation of an anomaly.

> **Note on citations.** `docs/literature_review.md` marks each reference as
> verified **[V]** or preprint **[P]**. A few 2025–2026 preprints should be
> re-checked for a peer-reviewed version before submission.
