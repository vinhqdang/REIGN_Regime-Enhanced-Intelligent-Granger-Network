# REIGN: Regime-Enhanced Intelligent Granger Network
## Implementation & Evaluation Plan

---

## 1. Overview

REIGN is a hybrid causal discovery framework for nonstationary multivariate financial time series. It combines:

- **LLM-derived soft adjacency priors** (domain knowledge injection)
- **CUTS+ neural Granger engine** (high-dimensional, irregular-TS backbone)
- **BOCPD/PELT regime segmentation** (nonstationarity handling)
- **NOTEARS acyclicity constraint** (valid DAG enforcement)
- **Confidence-weighted ensemble** (regime-local graph fusion)

**Target venue:** Journal of Data Science and Intelligent Systems (JDSIS), Scopus CiteScore 10.5

---

## 2. Algorithm Specification

### 2.1 Notation

| Symbol | Description |
|--------|-------------|
| `X ∈ R^{T×N}` | Input multivariate time series, T timesteps, N variables |
| `t_1,...,t_T` | Observation timestamps (possibly irregular) |
| `R = {r_1,...,r_K}` | Set of K detected regime windows |
| `A_LLM ∈ [0,1]^{N×N}` | LLM-generated soft adjacency prior matrix |
| `W^(k) ∈ R^{N×N}` | Learned causal weight matrix for regime k |
| `G^(k)` | Regime-local causal DAG |
| `G*` | Final ensemble causal DAG |
| `λ` | LLM-prior regularization strength |
| `α` | NOTEARS acyclicity penalty weight |

---

### 2.2 Stage 1 — Preprocessing & Imputation

**Goal:** Normalize the raw time series and handle irregular sampling before causal learning.

```
Input:  X ∈ R^{T×N}, timestamps t_1,...,t_T
Output: X̃ ∈ R^{T'×N} (uniformly resampled, imputed)

Steps:
  1. Z-score normalize each variable independently
  2. Detect and flag outliers (IQR ×3 threshold)
  3. Resample to uniform grid via linear interpolation
     (target frequency: median observed inter-sample gap)
  4. Impute remaining NaN via forward-fill + MICE for
     gaps longer than 3 consecutive steps
```

**Implementation:** `pandas`, `scikit-learn` (MICE = `IterativeImputer`)

---

### 2.3 Stage 2a — Regime Segmentation (BOCPD / PELT)

**Goal:** Partition the time series into K locally stationary windows.

```
Input:  X̃ ∈ R^{T'×N}
Output: Regime boundaries B = {b_0, b_1, ..., b_K}
        Regime windows R = {(b_{k-1}, b_k) : k=1..K}

Algorithm (offline mode — PELT):
  1. Compute per-variable rolling mean + variance features
  2. Run PELT with RBF cost function, penalty β = log(T') × N
  3. Merge regimes shorter than min_regime_length = 30 timesteps
     into adjacent longer regimes

Algorithm (online mode — BOCPD):
  1. Initialize run-length distribution P(r_t | x_{1:t})
  2. At each step t, compute hazard h = 1/λ_0 (λ_0 = expected regime length)
  3. Update belief using Gaussian conjugate model per variable
  4. Declare changepoint when MAP run-length resets to 0
```

**Hyperparameters:**
- PELT penalty β: tuned by BIC on held-out 20% of each dataset
- BOCPD hazard λ_0: set to `T' / expected_num_regimes` (default: T'/5)
- `min_regime_length`: 30 timesteps (prevents degenerate micro-regimes)

**Implementation:** `ruptures` (PELT), `bayesian_changepoint_detection` (BOCPD)

---

### 2.4 Stage 2b — LLM Prior Generation

**Goal:** Generate a soft adjacency matrix A_LLM from variable metadata using an LLM.

```
Input:  Variable names V = {v_1,...,v_N}, domain description D
Output: A_LLM ∈ [0,1]^{N×N}

Prompt template (per variable pair (v_i, v_j)):
  "Given the following business/financial variables:
   Variable i: {v_i} — {description_i}
   Variable j: {v_j} — {description_j}

   On a scale from 0 to 1, how likely is it that
   changes in {v_i} DIRECTLY CAUSE changes in {v_j}
   in a financial/business context?
   
   Respond with a single float between 0 and 1 only.
   Do not explain."

Batching strategy:
  - Query all N×(N-1) ordered pairs in batches of 50
  - Repeat each query 3 times, take mean (reduces hallucination variance)
  - Apply causal-order correction (Vashishtha et al. 2025):
      * Extract topological ordering from LLM via separate prompt
      * Zero out A_LLM[i,j] if j precedes i in topological order
  - Normalize: A_LLM = A_LLM / max(A_LLM)  [0,1 scaling]
```

**Confidence calibration:**
- Compute per-entry variance across 3 repetitions
- Entries with variance > 0.1 are flagged as low-confidence
- Low-confidence entries: weight halved before fusion with data likelihood

**Implementation:** `anthropic` Python SDK (claude-sonnet-4-20250514), async batch queries

---

### 2.5 Stage 3 — CUTS+ Neural Granger per Regime

**Goal:** Learn a causal weight matrix W^(k) for each regime window r_k.

```
Input:  X̃^(k) ∈ R^{T_k×N} (regime-k slice), A_LLM
Output: W^(k) ∈ R^{N×N} (regime-k causal weights)

Sub-stage 3a — Coarse-to-Fine Discovery (C2FD):
  1. Fit a sparse VAR(p) model on X̃^(k) with LASSO penalty
     → Candidate edge set E_coarse = {(i,j) : |VAR_coef[i,j]| > ε}
  2. Prune: only estimate neural Granger weights for (i,j) ∈ E_coarse
     (reduces O(N²) to O(|E_coarse|) neural fits)

Sub-stage 3b — Message-Passing GNN Granger:
  Architecture:
    - Input layer: sliding window of lag-p history for all N variables
    - MPGNN: 2-layer GNN with edge-gated attention
      * Node features: h_i^(l) = ReLU(W_node × [x_i^(t-p:t)])
      * Edge messages: m_{ij}^(l) = σ(W_edge × [h_i^(l) || h_j^(l)])
      * Aggregation: h_i^(l+1) = h_i^(l) + Σ_j m_{ij}^(l) × e_{ij}
      * Edge weights e_{ij} ∈ [0,1] are the learned causal scores → W^(k)
    - Output layer: predict x^(t) from aggregated node representations

  Loss function:
    L(W^(k)) = L_pred(W^(k))              [prediction MSE]
             + λ × ||W^(k) − A_LLM||_F²  [LLM prior regularization]
             + α × h(W^(k))               [NOTEARS acyclicity]
             + γ × ||W^(k)||_1            [sparsity]

  where:
    h(W) = tr(exp(W ⊙ W)) − N            [NOTEARS constraint, = 0 iff DAG]
    ||·||_F = Frobenius norm
    ||·||_1 = element-wise L1

  Optimization:
    - Outer loop: augmented Lagrangian for h(W^(k)) = 0
    - Inner loop: Adam (lr=1e-3, betas=(0.9, 0.999))
    - Early stopping: patience=20 epochs on regime-internal val split (20%)
    - Max epochs: 500 per regime

Sub-stage 3c — Edge thresholding:
  - Apply threshold τ = 0.1 to W^(k) to obtain binary DAG G^(k)
  - τ tuned on synthetic validation set (see Section 4.2)
```

**Hyperparameters:**
| Parameter | Default | Search range |
|-----------|---------|--------------|
| lag p | 5 | {3, 5, 10} |
| λ (LLM prior weight) | 0.1 | {0.01, 0.1, 0.5, 1.0} |
| α (acyclicity weight) | 1.0 | {0.5, 1.0, 2.0} |
| γ (sparsity weight) | 0.05 | {0.01, 0.05, 0.1} |
| GNN hidden dim | 64 | {32, 64, 128} |
| GNN layers | 2 | {1, 2, 3} |
| τ (edge threshold) | 0.1 | {0.05, 0.1, 0.2} |

**Implementation:** `PyTorch`, `torch-geometric`, custom NOTEARS augmented Lagrangian loop

---

### 2.6 Stage 4 — Confidence-Weighted Ensemble

**Goal:** Merge K regime-local DAGs into a single global causal graph G*.

```
Input:  {G^(k), W^(k), T_k} for k=1..K
Output: G* ∈ {0,1}^{N×N}, confidence C ∈ [0,1]^{N×N}

Steps:
  1. Compute regime weight:
       w_k = (T_k / T') × (1 / uncertainty_k)
     where uncertainty_k = mean(var of W^(k) entries across bootstrap runs)

  2. Aggregate weighted adjacency:
       C[i,j] = Σ_k w_k × W^(k)[i,j] / Σ_k w_k

  3. Threshold to binary DAG:
       G*[i,j] = 1  if C[i,j] > τ_ensemble (default: 0.3)

  4. Enforce acyclicity on G* via greedy cycle-breaking:
       while G* contains a cycle:
         remove edge (i,j) with lowest C[i,j] in the cycle

  5. Annotate each edge with stability label:
       "stable"   : present in > 80% of regime graphs
       "transient": present in 20–80% of regime graphs
       "spurious" : present in < 20% of regime graphs (filtered out)
```

**Implementation:** `networkx` (cycle detection, DAG enforcement), `numpy`

---

### 2.7 Full Algorithm Pseudocode

```python
def REIGN(X, timestamps, variable_names, domain_desc,
          lambda_prior=0.1, alpha_dag=1.0, gamma_sparse=0.05):

    # Stage 1: Preprocessing
    X_clean = preprocess(X, timestamps)

    # Stage 2a: Regime segmentation
    regimes = detect_regimes_PELT(X_clean)

    # Stage 2b: LLM prior (runs in parallel with 2a)
    A_llm = generate_llm_prior(variable_names, domain_desc)
    A_llm = apply_causal_order_correction(A_llm, variable_names)

    # Stage 3: Per-regime CUTS+ Granger
    local_graphs = []
    for k, (start, end) in enumerate(regimes):
        X_k = X_clean[start:end]
        E_coarse = coarse_discovery_VAR(X_k)
        W_k = train_mpgnn_granger(X_k, E_coarse, A_llm,
                                   lambda_prior, alpha_dag, gamma_sparse)
        G_k = threshold(W_k, tau=0.1)
        local_graphs.append((G_k, W_k, end - start))

    # Stage 4: Ensemble
    G_star, confidence = confidence_weighted_ensemble(local_graphs)
    G_star = enforce_dag(G_star, confidence)

    return G_star, confidence
```

---

## 3. Evaluation Plan

### 3.1 Datasets

#### Tier 1 — Synthetic (controlled, ground-truth known)

| Dataset | N vars | T steps | Key challenge | Source |
|---------|--------|---------|---------------|--------|
| TimeGraph-Linear | 10–50 | 1000–5000 | Baseline, linear deps | KDD 2025 |
| TimeGraph-Nonlinear | 10–50 | 1000–5000 | Nonlinear Granger | KDD 2025 |
| TimeGraph-Nonstationary | 10–50 | 1000–5000 | 3 regime switches | KDD 2025 |
| TimeGraph-Irregular | 10–50 | variable | 30% missing obs | KDD 2025 |
| Lorenz-96 | 10, 20 | 2000 | Chaotic nonlinear | Standard |
| VAR-Regime (custom) | 15 | 3000 | 4 regimes, known edges | Generated |

**VAR-Regime generation protocol:**
```python
# Generate synthetic regime-switching VAR data
for k in range(4):
    A_k = random_sparse_DAG(N=15, density=0.2)  # ground truth
    X_k = simulate_VAR(A_k, T=750, noise='gaussian')
regimes = [X_1, X_2, X_3, X_4]
X_full = concatenate(regimes)  # known changepoints at 750, 1500, 2250
```

#### Tier 2 — Real-world causal benchmarks (ground-truth available)

| Dataset | N vars | Description | Source |
|---------|--------|-------------|--------|
| Sachs | 11 | Protein signaling, 853 cells | Science 2005 |
| ALARM | 37 | ICU monitoring BN | AIME 1989 |
| HEPAR II | 70 | Hepatitis diagnosis BN | Onisko 2003 |
| CausalRivers-small | 20 | River discharge, 15-min | ICLR 2025 |

#### Tier 3 — Financial application datasets

| Dataset | N vars | Description | Source |
|---------|--------|-------------|--------|
| Telecom Churn Panel | 12–20 | Monthly KPIs: churn rate, ARPU, NPS, ad spend, competitor price | SmartOSC / UCI Telco |
| Yahoo Finance Equities | 20 | Daily returns, 10-year window, 5 sectors | Yahoo Finance API |
| M4 Competition (finance subset) | 15 | Quarterly macro + sector variables | M4 Competition |

**Note:** For SmartOSC data, anonymize variable names before publishing; include a synthetic surrogate dataset in the public repo.

---

### 3.2 Evaluation Metrics

#### Structural metrics (compare G* to ground-truth G_true)

| Metric | Formula | Notes |
|--------|---------|-------|
| **AUROC** | Area under ROC curve on edge scores C[i,j] | Primary metric — threshold-free |
| **AUPR** | Area under Precision-Recall curve | Better for sparse graphs |
| **SHD** | Structural Hamming Distance = FP + FN + reversals | Standard DAG distance |
| **Precision** | TP / (TP + FP) | Edge precision |
| **Recall** | TP / (TP + FN) | Edge recall — emphasized per LLM-CD finding |
| **F1** | 2 × P × R / (P + R) | Harmonic mean |

```python
# Threshold-free AUROC computation
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(C, G_true):
    y_true = G_true.flatten()
    y_score = C.flatten()
    return {
        'AUROC': roc_auc_score(y_true, y_score),
        'AUPR':  average_precision_score(y_true, y_score),
        'SHD':   shd(threshold(C, 0.3), G_true),
        'F1':    f1(threshold(C, 0.3), G_true),
    }
```

#### Regime-specific metrics

| Metric | Description |
|--------|-------------|
| **Regime-AUROC** | AUROC computed separately per regime, then averaged |
| **Stability score** | Fraction of edges labeled "stable" that are true positives |
| **Transient precision** | Precision of "transient" edge labels vs. ground truth |

#### Computational metrics

| Metric | Description |
|--------|-------------|
| **Wall-clock time** | Total runtime in seconds (CPU + GPU) |
| **Memory peak** | Peak RAM usage in GB |
| **Scalability slope** | Runtime growth as N increases: 10→20→50 variables |

---

### 3.3 Baselines

All baselines use the same preprocessed X̃ as input. No baseline receives A_LLM.

#### Group A — Classical constraint-based

| Baseline | Key reference | Notes |
|----------|--------------|-------|
| PC | Spirtes et al. 2000 | `causal-learn` implementation |
| PCMCI+ | Runge 2020 | `tigramite` implementation |
| LPCMCI | Gerhardus & Runge 2020 | Tests confounded setting |

#### Group B — Score-based / continuous

| Baseline | Key reference | Notes |
|----------|--------------|-------|
| NOTEARS | Zheng et al. 2018 | Static, no time-lagged |
| NOTEARS-MLP | Zheng et al. 2020 | Nonlinear version |
| DYNOTEARS | Pamfil et al. 2020 | Closest classical TS baseline |
| GES | Chickering 2002 | `causal-learn` |

#### Group C — Neural Granger

| Baseline | Key reference | Notes |
|----------|--------------|-------|
| cMLP / cLSTM | Tank et al. 2022 | Seminal neural Granger |
| CUTS | Cheng et al. 2023 | REIGN backbone (no C2FD) |
| CUTS+ | Cheng et al. 2024 | REIGN backbone (no LLM, no regime) |
| GVAR | Marcinkevičs & Vogt 2021 | Time-varying coefficients |

#### Group D — LLM-augmented (ablation group)

| Baseline | Key reference | Notes |
|----------|--------------|-------|
| LLM-CD | Du et al. KDD 2025 | LLM + PC, no neural Granger |
| Causal-LLM | Roy et al. EMNLP 2025 | LLM one-shot, no data |
| LLM-DCD | Waxman et al. 2024 | LLM warm-start, closest to REIGN |
| CD-NOD | Huang et al. JMLR 2020 | Regime-aware, no LLM |
| Regime-PCMCI | Saggioro et al. 2020 | Closest regime-aware baseline |

#### Group E — REIGN ablations (internal)

| Variant | What is removed |
|---------|----------------|
| REIGN-noLLM | Remove A_LLM (λ=0); data-only |
| REIGN-noRegime | Remove BOCPD/PELT; single global graph |
| REIGN-hardPrior | A_LLM as hard constraint (not soft regularizer) |
| REIGN-PELT | Use PELT only (vs. BOCPD) |
| REIGN-BOCPD | Use BOCPD only (vs. PELT) |
| REIGN-noC2FD | Remove coarse-to-fine pruning; full O(N²) Granger |

---

### 3.4 Experimental Protocol

#### Train / validation / test split

```
Synthetic data:
  - 5-fold cross-validation across 5 independently generated graphs
  - Report mean ± std across folds

Real causal benchmarks (Sachs, ALARM, HEPAR II):
  - No train/test split (ground truth is the full graph)
  - Bootstrap 100 subsamples of observations → 100 AUROC estimates

Financial data:
  - Temporal split: 70% train (older), 15% val, 15% test (most recent)
  - No data leakage: val/test regimes never seen during λ, α, γ tuning
```

#### Hyperparameter tuning

```
Search method: Bayesian optimization (Optuna, 50 trials per dataset)
Tuning set: TimeGraph-Nonstationary validation split
Fixed across all datasets once tuned (no per-dataset retuning)
Reported hyperparameters: Table in paper appendix
```

#### Statistical significance

```
- Report p-values from Wilcoxon signed-rank test (REIGN vs. each baseline)
- Significance threshold: p < 0.05
- Bonferroni correction applied across 14 baselines
- Effect size: Cohen's d on AUROC differences
```

#### Robustness checks (CausalCompass-inspired)

```
Run REIGN and all baselines under:
  1. Measurement noise: add Gaussian noise σ ∈ {0.1, 0.3, 0.5} × signal std
  2. Missing data: randomly mask 10%, 20%, 30% of observations
  3. Short regimes: reduce min regime length to 15 timesteps
  4. LLM prompt variation: rephrase prior prompt 5 ways; report AUROC variance
  5. Wrong number of regimes: set K_forced ∈ {K_true/2, K_true, 2×K_true}
```

---

## 4. Repository Structure

```
REIGN/
├── README.md
├── REIGN_plan.md               ← this file
├── requirements.txt
├── data/
│   ├── synthetic/
│   │   ├── generate_VAR_regime.py
│   │   └── download_timegraph.py
│   ├── benchmarks/
│   │   ├── sachs.csv
│   │   ├── alarm.bif
│   │   └── hepar2.bif
│   └── financial/
│       ├── telecom_churn_surrogate.csv
│       └── yahoo_finance_download.py
├── src/
│   ├── preprocessing.py        ← Stage 1
│   ├── regime_detection.py     ← Stage 2a (BOCPD + PELT)
│   ├── llm_prior.py            ← Stage 2b (LLM query + calibration)
│   ├── cuts_plus.py            ← Stage 3 (MPGNN + NOTEARS)
│   ├── ensemble.py             ← Stage 4 (confidence-weighted merge)
│   ├── reign.py                ← Main pipeline entry point
│   └── baselines/
│       ├── pcmci_plus.py
│       ├── dynotears.py
│       ├── cuts_baseline.py
│       ├── llm_cd.py
│       └── regime_pcmci.py
├── evaluation/
│   ├── metrics.py              ← AUROC, AUPR, SHD, F1, regime metrics
│   ├── run_experiments.py      ← Full experiment runner
│   └── statistical_tests.py   ← Wilcoxon, Bonferroni, Cohen's d
├── experiments/
│   ├── configs/
│   │   ├── timegraph.yaml
│   │   ├── sachs.yaml
│   │   └── financial.yaml
│   └── results/               ← Auto-generated CSV + JSON outputs
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_regime_visualization.ipynb
│   ├── 03_llm_prior_analysis.ipynb
│   └── 04_results_tables.ipynb
└── tests/
    ├── test_preprocessing.py
    ├── test_regime_detection.py
    ├── test_llm_prior.py
    ├── test_cuts_plus.py
    └── test_ensemble.py
```

---

## 5. Implementation Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Data pipeline | `preprocessing.py`, `generate_VAR_regime.py`, all datasets downloaded |
| 1 | Regime detection | `regime_detection.py`, unit tests, visualization notebook |
| 2 | LLM prior module | `llm_prior.py`, prompt templates, calibration logic, caching |
| 2 | CUTS+ backbone | `cuts_plus.py`, NOTEARS augmented Lagrangian, MPGNN architecture |
| 3 | Full REIGN pipeline | `reign.py`, end-to-end run on TimeGraph-Linear |
| 3 | Baselines | All 14 baselines runnable, results on TimeGraph |
| 4 | Ablations | 6 REIGN variants, ablation table |
| 4 | Robustness checks | Noise/missing-data experiments |
| 5 | Financial experiments | Telecom churn + Yahoo Finance results |
| 5 | Statistical tests | Wilcoxon + Bonferroni tables |
| 6 | Paper draft | Methodology + Experiments sections using all results |

---

## 6. Key Implementation Notes

### LLM prior caching
```python
# Cache all LLM queries to avoid re-querying during ablations
# Store as: cache/{dataset_name}_{model_name}_prior.json
# Format: {"(i,j)": [q1, q2, q3], "mean": float, "var": float}
```

### NOTEARS numerical stability
```python
# Use matrix exponential via scipy.linalg.expm, not naive power series
# Clip W entries to [-3, 3] before expm to prevent overflow
# Augmented Lagrangian rho schedule: rho *= 2 when h(W) not decreasing
```

### Regime minimum length guard
```python
# If a detected regime has T_k < min_regime_length:
#   - If K > 2: merge with adjacent regime of most similar mean
#   - If K == 2: fall back to single global graph (REIGN-noRegime mode)
```

### LLM prior for unnamed variables
```python
# If variable names are uninformative (e.g., "var_1", "var_2"):
#   - Set A_LLM = 0.5 × ones(N,N) - 0.5 × eye(N)  [uniform prior]
#   - Set lambda_prior = 0  [effectively REIGN-noLLM]
#   - Log warning to user
```

---

## 7. Expected Results (Targets)

Based on the literature survey, REIGN should achieve:

| Dataset | Metric | Target | Reference ceiling |
|---------|--------|--------|-------------------|
| TimeGraph-Nonstationary | AUROC | > 0.82 | CUTS+: ~0.76 |
| TimeGraph-Irregular | AUROC | > 0.79 | CUTS+: ~0.74 |
| Sachs | AUROC | > 0.85 | LLM-CD: ~0.83 |
| CausalRivers-small | AUROC | > 0.72 | VAR: ~0.70 |
| Telecom Churn | F1 | > 0.65 | DYNOTEARS: ~0.55 |

**Minimum publishable bar:** REIGN outperforms CUTS+ on TimeGraph-Nonstationary and outperforms DYNOTEARS on at least one financial dataset.

---

## 8. Dependencies

```txt
# requirements.txt
torch>=2.2.0
torch-geometric>=2.5.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
scipy>=1.12.0
ruptures>=1.1.9
networkx>=3.2.0
optuna>=3.6.0
anthropic>=0.25.0
tigramite>=5.2.0        # PCMCI+
causal-learn>=0.1.3.8   # PC, GES, LiNGAM
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0
tqdm>=4.66.0
pytest>=8.0.0
```

---

*Document version: 1.0 — May 2026*
*Authors: Quang-Vinh Dang, British University Vietnam / SmartOSC*