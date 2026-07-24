# ANTE: Anytime-valid Nonparametric Test of synErgy

*Sequential, anytime-valid testing for group (synergistic) causality.*

A betting-martingale (e-process) test for **group causality** — the hypothesis
that a set of causes acts on an effect *jointly*, beyond the sum of their
individual contributions — monitored **anytime-valid** as data streams in.

The existing literatures on group causality — sufficient-cause interaction
(RERI/PRISM/generalized synergy index), joint-effect estimation, information-
theoretic decomposition (PID/SURD/PEID), and synergy in financial markets
(Scagliarini 2020) — are **all fixed-sample**. None offers a sequential test you
can monitor continuously and stop as soon as evidence is decisive. This repo
builds that missing piece, in two flavours:

* **ANTE** — the interventional/randomized test (additive interaction contrast,
  IPW score). See `docs/methodology.md`.
* **ANTE-SG** — the flagship **financial time-series** algorithm: anytime-valid,
  regime-aware detection of **synergistic Granger causality** (a group of assets
  jointly Granger-causes a target beyond the sum of parts), defined as
  *super-additive out-of-sample predictive gain* and tested by betting. See
  `docs/algorithm_ante_sg.md`, validated on 10 years of real market data.

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

## ANTE-SG on real financial data

`docs/algorithm_ante_sg.md` defines synergy as **super-additive out-of-sample
predictive gain** — the joint history of {A,B} predicts target Y beyond the sum
of what each adds alone — tested sequentially by betting on the per-step score
$s_t=\ell^A_t+\ell^B_t-\ell^{AB}_t-\ell^{\text{base}}_t$, with forgetting-RLS
predictors for regime-adaptivity. No stationarity or correct-specification is
needed (prequential/game-theoretic null).

- **Synthetic (`synthetic_synergy.py`)**: correctly ignores additive effects
  (rej 0.00) *and* within-variable nonlinearity $Y{=}A^2$ (0.01), fires on genuine
  $Y{=}A{\cdot}B$ (0.71); power ↑ and detection latency ↓ with synergy strength.
- **Real data (`financial_case_study.py`, `mechanism_probe.py`)**: 16-asset,
  10-year daily panel (+ 6 world indices) via `data/download_financial.py`. Across
  a 1,365-triple volatility scan, cross-region lead-lag, and economic mechanisms
  (oil×dollar, leverage, vol×credit), daily dependence is **additive + redundant,
  not synergistic** — apparent synergies fail out-of-sample and none survive
  anytime-valid control. ANTE-SG acts as a rigorous filter against false synergy.
  See `docs/RESULTS.md`.

## Benchmark vs SOTA (`docs/benchmark.md`)

Head-to-head against 2024–2026 synergy methods — **PEID** (2026, re-implemented),
**SURD** (2024, cloned code), **Williams–Beer PID**, **interaction information**,
and **O-information** (`hoi`):

- **Detection accuracy** — ANTE **matches** the best batch estimators (synthetic
  AUC: SURD/PEID 1.00, ANTE 0.98, Gaussian interaction-info 0.87; real-data
  cross-asset-vs-redundant AUC: ANTE 1.00, PEID 1.00, SURD 0.85). It does not
  claim to beat them as a point estimator.
- **Anytime-valid error control (decisive)** — under continuous monitoring ANTE
  holds type-I at **0.016 ≤ α**, while a peeking batch permutation test inflates
  to **0.14**. None of the batch SOTA methods have any sequential guarantee.

**Verdict: complementarity, not blanket superiority** — ANTE matches SOTA
detection while adding the sequential, anytime-valid monitoring guarantee they
lack (the gap identified in `docs/literature_review_finance.md`).

## Layout

```
src/
  data_generators.py       # interventional streams (ANTE)
  synergy_eprocess.py      # ANTE: IPW score, betting e-process, subset refinement
  synergistic_granger.py   # ANTE-SG: forgetting-RLS predictors, synergy score, detector, baselines
experiments/
  run_experiments.py       # ANTE E1–E4 (interventional)
  synthetic_synergy.py     # ANTE-SG synthetic S1–S4
  financial_case_study.py  # ANTE-SG real-data scan + OOS + event trajectory
  mechanism_probe.py       # curated economic-mechanism probe
  results/                 # JSON + 300dpi PNGs
data/
  download_financial.py    # Yahoo chart-API downloader (no key)
  returns_panel.csv, world_indices_returns.csv, manifest.json
docs/
  methodology.md           # ANTE (interventional) construction + extensions
  algorithm_ante_sg.md     # ANTE-SG algorithm, positioning, experiment design
  RESULTS.md               # all results
  literature_review.md, literature_review_finance.md   # verified citations + gaps
```

## Run

```bash
pip install -r requirements.txt
python data/download_financial.py            # fetch real market data
python experiments/synthetic_synergy.py      # ANTE-SG synthetic validation
python experiments/financial_case_study.py   # ANTE-SG real-data case study
python experiments/mechanism_probe.py        # curated mechanisms
python experiments/run_experiments.py        # interventional ANTE (E1–E4)
```

## Status / next steps

Working prototype with sound core, synthetic validation, and a real-financial-data
case study. Natural extensions: higher-frequency/intraday data and longer lags
(where synergy may be larger); nonlinear (kernel/neural) predictors in the same
betting scaffold; the doubly-robust/AIPW and graph/GNN instantiations sketched in
`docs/methodology.md`; a full manuscript draft.

> **Note on citations.** The `literature_review*.md` files tag each reference
> verified **[V]** / preprint **[P]** / unopened **[U]**; re-check preprints for a
> peer-reviewed version before submission.
