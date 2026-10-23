# Regime-Aware Causal Discovery — Research Monorepo

This repository hosts **two related but distinct research papers**. They share a
common theme (causal discovery under real-world complications) but are separate
projects with separate manuscripts, code, and experiments.

---

## Paper 1 — REIGN (published)

**REIGN: Regime-Enhanced Intelligent Granger Network for Nonstationary Causal
Discovery.** A causal-discovery framework for nonstationary multivariate time
series that combines PELT regime segmentation, zero-shot LLM prior injection, and
a coarse-to-fine message-passing GNN with a DAG-constrained Augmented Lagrangian
objective, aggregated by a confidence-weighted ensemble.

*Status: **published** (Online First, 18 September 2026) in the* **Journal of Data
Science and Intelligent Systems** *(Bon View Publishing, open access, CC BY 4.0).
Received 18 May 2026, revised 29 June 2026, accepted 20 July 2026.*

- **Authors:** Quang-Vinh Dang, Minh Ngoc Dinh, Dat Le, Dinh-Minh Nguyen
- **DOI:** [10.47852/bonviewJDSIS620210441](https://doi.org/10.47852/bonviewJDSIS620210441)
- **Article page:** <https://ojs.bonviewpress.com/index.php/jdsis/article/view/10441>
- **Headline result:** on a nonstationary VAR benchmark REIGN reaches Optimal F1 = 0.529
  (PCMCI+ 0.415, PC 0.214), AUROC 0.513 vs 0.406 and AUPR 0.400 vs 0.338.

**How to cite**

> Dang, Q.-V., Dinh, M. N., Le, D., & Nguyen, D.-M. (2026). REIGN: Regime-Enhanced
> Intelligent Granger Network for Nonstationary Causal Discovery. *Journal of Data
> Science and Intelligent Systems*. https://doi.org/10.47852/bonviewJDSIS620210441

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

## Paper 2 — ANTE (revision submitted, 23 October 2026)

*Status: revised manuscript and response letter submitted on 23 October 2026 to* **Discover Applied Mathematics** *(submission ID 7d12c900-4ad8-499d-aff6-602fc39f5a1e, major revision; awaiting the editor's decision).*

**Anytime-Valid Sequential Testing of Group Synergy: Interventional and Predictive
Formulations, with an Application to Financial Markets** (submitted to *Discover
Applied Mathematics*; the first version was titled "Anytime-Valid Nonparametric Testing
of Synergistic (Group) Causality, with an Application Establishing Its Absence in Daily
Financial Markets"). ANTE is a betting-martingale (e-process) test of whether a set of
variables acts on an outcome jointly, beyond the sum of its members, monitored
anytime-valid as data stream in. It has an *interventional* part (known randomised
design, sequential null $\theta_t\le0$) and a *predictive* part, **ANTE-SG**, for time
series, whose null compares an additive and a joint online forecaster.

**What the revision changed.** The group-test validity proof by clipping was wrong; it is
replaced by a truncated-loss score normalised by a bound known before each observation
(`sequential-synergy-causality/src/ante_v2.py`), which gives an exact validity proof. The
isolation proposition is corrected (the four-model contrast of version 1 is contaminated by
redundancy; the add-versus-joint contrast is used), "absence" claims are withdrawn, and the
financial search is fully specified (27,027 hypotheses, 43 families) with placebo
(circularly shifted target) calibration.

**Current empirical findings (see the manuscript for the exact wording).** In daily and
weekly returns and volatility there is no family-wise detection and the real evidence matches
a placebo null; the test has low power on real noise and the bounds on average synergy are weak,
so this is not proof of absence. For binary stress indicators and hourly crypto the placebo shows
that the test rejects a true null in 10-40% of hypotheses, so rejections there are not interpreted.
The turbulence example of version 1 is withdrawn (both tests "detect" on phase-randomised surrogates).
Old documents under `sequential-synergy-causality/docs/` describe version 1 and are superseded.

| Path | Contents |
|------|----------|
| `sequential-synergy-causality/src/` | e-process, original `ante_sg`, revised `ante_v2.py` (exactly valid score) |
| `sequential-synergy-causality/experiments/revision/` | all revision experiments (R1-R18), results in `results/` |
| `sequential-synergy-causality/manuscript/` | revised manuscript (`main.tex/.pdf`), `response_to_reviewers.tex/.pdf` |

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
