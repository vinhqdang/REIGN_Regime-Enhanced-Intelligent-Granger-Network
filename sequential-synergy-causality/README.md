# ANTE: Anytime-valid Nonparametric Test of synErgy

*Sequential, anytime-valid testing for group (synergistic) causality.*

A betting-martingale (e-process) test for **group causality** — the hypothesis
that a set of causes acts on an effect *jointly*, beyond the sum of their
individual contributions — monitored **anytime-valid** as data streams in.

## Headline finding: group causality does NOT exist in daily finance

Using ANTE as a demonstrably-powered instrument, we **establish** (evidence of
absence, not absence of evidence) that genuine super-additive group causality is
**absent from daily markets**, while the same test detects it wherever it truly
exists. Full argument: `docs/finding_group_causality.md`.

- **The test is powered:** it detects synthetic synergy (74–100%) and the real
  turbulence energy-cascade synergy (SURD's Nature-Comms data; 3 coarse scales
  jointly → finest scale, log₁₀E = 4.0).
- **Finance matches a TRUE null:** across **1,320** real financial group triples,
  median evidence log₁₀E = **−0.93** (a bet on synergy systematically *loses*
  money) vs 0.00 under the calibrated null and +2.79 under real synergy; **0**
  detections out of ~3,700+ triples spanning returns/volatility/tail-stress,
  daily/hourly, equities/ETFs/macro/crypto, and 2-/3-/4-way groups.

Interpretation: daily markets are common-factor + pairwise; irreducible
higher-order structure is negligible — a quantified caution to in-sample
synergy-in-finance claims. Synergistic group causality is a physics/biology
phenomenon (confirmed on turbulence), not a daily-markets one.

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

Baselines validated against their own papers (PEID: XOR=1.000, AND=0.190 vs paper
0.189; DGC: size 0.035, power 1.0 per Table 1; SURD via cloned code).

- **Detection accuracy** — ANTE **matches** the best batch estimators (synthetic
  AUC: SURD/PEID 1.00, ANTE 0.98; real-data cross-asset-vs-redundant AUC: ANTE
  1.00, PEID 0.90, SURD 0.85). It does not claim to beat them as a point estimator.
- **Fixed-N power** — a calibrated batch SURD test is *more* powerful at a
  pre-committed N; ANTE trades this for anytime-validity (honest tradeoff).
- **Error control under monitoring (decisive)** — ANTE holds type-I at
  **0.016 ≤ α** under continuous monitoring vs **0.14** for a peeking batch test.
- **Group synergy vs the only other sequential method** — on Y=A⊕B, Jha's DGC
  (2026, single-source) detects at 0.01; **ANTE at 1.00**.
- **Super-additive vs joint-dependence** — ANTE gives ~0 on additive Y=A+B;
  PEID/SURD/PID report positive "synergy" (they conflate needing-both with
  interaction).

**Verdict: complementarity, not blanket superiority** — ANTE is the first
anytime-valid, sequential, *group*-synergy test; every prior method is batch
(SURD/PEID/PID/O-info) or single-source (DGC). Full detail: `docs/benchmark.md`.

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
