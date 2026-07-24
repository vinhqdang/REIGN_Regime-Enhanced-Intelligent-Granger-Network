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
anytime-valid as data streams in. Unifies the additive causal-pie (RERI-type) and
information-theoretic (PID/PEID-type) synergy nulls under one test, with an
anytime-valid subset-refinement search.

*Status: early-stage prototype (code, experiments, methodology, and literature
review complete; manuscript not yet drafted).*

| Path | Contents |
|------|----------|
| `sequential-synergy-causality/src/` | e-process, IPW interaction score, subset-refinement search |
| `sequential-synergy-causality/experiments/` | E1–E4 experiments + figures |
| `sequential-synergy-causality/docs/` | methodology and literature review |

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
