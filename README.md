# Regime-Aware Causal Discovery — Research Monorepo

This repository hosts **two related but distinct research papers**. They share a
common theme (causal discovery under real-world complications) but are separate
projects with separate manuscripts, code, and experiments.

---

## Paper 1 — REIGN (accepted)

**REIGN: Regime-Enhanced Intelligent Granger Network for Nonstationary Causal
Discovery.** A causal-discovery framework for nonstationary multivariate time
series that combines PELT regime segmentation, zero-shot LLM prior injection, and
a coarse-to-fine message-passing GNN with a DAG-constrained Augmented Lagrangian
objective, aggregated by a confidence-weighted ensemble.

*Status: accepted, in production for publication (Journal of Data Science and
Intelligent Systems).*

| Path | Contents |
|------|----------|
| `manuscripts/` | LaTeX source, figures, and compiled PDF of the REIGN paper |
| `src/` | REIGN pipeline: `regime_detection.py`, `llm_prior.py`, `cuts_plus.py`, `ensemble.py`, `preprocessing.py`, `reign.py` |
| `evaluation/` | Figure-generation scripts |
| `experiments/` | Result artifacts |
| `data/` | Synthetic nonstationary VAR benchmark |
| `tests/` | Pipeline tests |
| `plan.md` | REIGN project plan |

---

## Paper 2 — ANTE (in progress)

**ANTE: Anytime-valid Nonparametric Test of synErgy — Sequential Testing for
Group (Synergistic) Causality.** A
betting-martingale (e-process) test for the hypothesis that a set of causes acts
on an effect *jointly*, beyond the sum of individual contributions — monitored
anytime-valid as data streams in.

The flagship algorithm is **ANTE-SG**, for **financial time series**: anytime-
valid, regime-aware detection of *synergistic Granger causality* (super-additive
out-of-sample predictive gain, tested by betting; generalizes to k-way groups).

**Headline finding — group causality does NOT exist in daily finance.** Used as a
demonstrably-powered instrument, ANTE *establishes* (evidence of absence, not
absence of evidence) that genuine super-additive group causality is absent from
daily markets: across ~3,700+ group triples (returns/volatility/tail-stress,
daily/hourly, equities/ETFs/macro/crypto, 2-/3-/4-way), the financial evidence sits
on a true-null distribution (median log₁₀E = −0.93, i.e. a bet on synergy *loses*)
with 0 detections — while the same test detects synthetic synergy (74–100%) and
real turbulence energy-cascade synergy (SURD's data). So synergistic group
causality is a physics/biology phenomenon, not a daily-markets one — a quantified
caution to in-sample synergy-in-finance claims. See
`sequential-synergy-causality/docs/finding_group_causality.md`.

*Status: algorithm design, implementation, synthetic validation,
real-financial-data case study, SOTA benchmark, and methodology complete. Full
manuscript drafted for* **Statistics and Computing** *(Springer, Q1, free to
publish), in `sequential-synergy-causality/manuscript/` (Springer `sn-jnl`
template, math-heavy with theorems/proofs; compiled PDF included).*

| Path | Contents |
|------|----------|
| `sequential-synergy-causality/src/` | e-process, IPW interaction score, subset-refinement search |
| `sequential-synergy-causality/experiments/` | E1–E4 experiments + figures |
| `sequential-synergy-causality/docs/` | methodology and literature review |
| `sequential-synergy-causality/manuscript/` | Statistics and Computing manuscript (LaTeX + PDF) |

See [`sequential-synergy-causality/README.md`](sequential-synergy-causality/README.md)
for details.

---

## Relationship between the two papers

Paper 1 (REIGN) is a **method** for causal discovery under nonstationarity.
Paper 2 is a **test** for a different, complementary question — whether a *group*
of causes acts synergistically — and is intended to plug into graph/GNN causal
pipelines (including a REIGN-style backbone) rather than replace them. They are
developed and versioned together here for convenience but are independent
submissions.
