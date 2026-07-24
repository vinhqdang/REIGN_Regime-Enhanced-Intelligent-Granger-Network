# Datasets and Baseline Models

## Datasets

| # | Dataset | Type | Size | Source | Used for |
|---|---------|------|------|--------|----------|
| 1 | Synthetic interventional streams | binary A,B→C (null / AND / XOR / 3-var triple) | ≤ 4,000 steps | generated (`data_generators.py`) | ANTE validation E1–E4 (type-I, power, subset refinement) |
| 2 | Synthetic common-factor VAR | additive / own-nonlinear (A²) / synergistic (A·B) + market factor | T≈2,500–4,000 | generated | ANTE-SG validation S1–S4, benchmark AUC, sample efficiency, power calibration |
| 3 | US sector-ETF + macro panel | 16 daily: SPY, XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLB, TLT, GLD, USO, HYG, UUP, ^VIX | 2,512 days (2016–2026) | Yahoo chart API (`download_financial.py`) | main real scans, portfolio-variance synergy, systemic-risk groups |
| 4 | World equity indices | 6: S&P 500, DAX, FTSE, Nikkei, Hang Seng, KOSPI | 2,138 days | Yahoo (`world_indices_returns.csv`) | cross-region lead-lag / synergy |
| 5 | "Famous stocks" basket | 13 daily: TSLA, NVDA, AAPL, MSFT, AMZN, META, JPM, XOM, SPY, GLD, USO, TLT, VIX | 2,512 days | Yahoo (`famous_returns.csv`) | plain-English lead-lag + group-causality scans |
| 6 | Famous stocks — intraday | 12 assets, hourly | 3,328 hours | Yahoo (`famous_hourly_returns.csv`) | intraday group causality |
| 7 | Crypto — intraday | 10 coins: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LTC, AVAX, LINK, hourly | 16,761 hours | Yahoo (`download_crypto_hourly.py`) | high-sample group-synergy scan (returns + volatility) |
| 8 | Climate teleconnections | 6 monthly indices: ENSO, SOI, NAO, PDO, PNA, AMO | 865 months (1951–2023) | NOAA PSL (`climate_indices.csv`) | non-finance group-causality test |
| 9 | Turbulence energy cascade | 4 scale-resolved energy signals | 21,760 samples | SURD repo (real fluid-dynamics data) | **positive** multi-series group causality (physics) |

## Baseline models

| # | Method | Year | Type | How obtained | Role |
|---|--------|------|------|--------------|------|
| 1 | **PEID** (Syn^EID) | 2026 | batch estimator | re-implemented (no code released); validated vs paper — AND=0.19, XOR=1.0, continuous α-trend | main 2026 synergy baseline |
| 2 | **SURD** | 2024 | batch decomposition | cloned reference code (Nat. Commun.) | main nonparametric SOTA; turbulence head-to-head |
| 3 | **Jha DGC** (Distributional Granger Causality) | 2026 | **sequential** test (single-source) | re-implemented; validated vs Table 1 — size 0.035, power 1.0 | only other sequential method; shown to miss group synergy |
| 4 | **O-information** (`hoi`) | 2024 (≈ 2026 finance OIR) | batch estimator | pip package | higher-order-info baseline |
| 5 | **PID (Williams–Beer, I_min)** | classical | batch decomposition | re-implemented | canonical PID synergy |
| 6 | **Interaction information** (Gaussian co-information) | classical | batch estimator | re-implemented | linear synergy baseline (misses products) |
| 7 | **Pairwise / conditional Granger F-test** | classical | batch test | re-implemented | individual (non-group) causality reference |
| 8 | **Peeking permutation test** | — | batch test, monitored | re-implemented | "naive continuous monitoring" foil (type-I inflates to 0.14) |

**ANTE / ANTE-SG / `ante_group`** (ours) — sequential, anytime-valid group-synergy
test — is evaluated against all of the above.
