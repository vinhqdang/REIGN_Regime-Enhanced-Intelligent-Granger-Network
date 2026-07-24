# ANTE-SG — Results

All numbers reproducible from `experiments/` (JSON in `experiments/results/`).

## Synthetic validation (`synthetic_synergy.py`)

Common-factor VAR: sources A,B with their own AR dynamics and a shared market
factor Z; target Y depends on A,B additively / own-nonlinearly / synergistically.
All tests condition on Z (so we isolate synergy beyond the common factor).

| Experiment | Result | Reading |
|---|---|---|
| **S1 discrimination** | rejection rate — additive **0.00**, own-nonlinear $Y{=}A^2$ **0.01**, synergistic $Y{=}A{\cdot}B$ **0.71** | correctly ignores additive effects *and* within-variable nonlinearity; fires only on genuine cross-interaction |
| **S2 anytime-validity** | e-process type-I **0.01** ≤ α=0.05 (peeking **0.018**) | error controlled under continuous monitoring |
| **S3 power / latency** | power **0 → 0.79** as A·B strength 0→0.9; median detection ~**1800 → 900** steps | power rises and detection accelerates with synergy strength |
| **S4 trajectories** | capital flat under both nulls, grows under synergy | visual confirmation |

The **S1 own-nonlinearity discriminator** is the key correctness property: a
mechanism $Y=g(A)$, however nonlinear, is absorbed by the marginal model and is
*not* mistaken for synergy. Only irreducible $A\times B$ structure is flagged.

## Real financial data — the positive result (`portfolio_variance_synergy.py`)

**Contemporaneous synergy in portfolio realized variance.** A portfolio's variance
is not the sum of its constituents' variances — the covariance (interaction) term
is first-order. For a 50/50 portfolio, $\mathrm{RV}=\big(\tfrac{R_A+R_B}{2}\big)^2
=\tfrac14R_A^2+\tfrac14R_B^2+\tfrac12R_AR_B$; the $R_AR_B$ term is genuine synergy
no single asset's (even squared) return captures. ANTE's contemporaneous variant
recovers it from **real** daily returns, and — crucially — **discriminates**:

| | pairs | result |
|---|---|---|
| **Cross-asset-class** (distinct covariance) | XLK/GLD, TLT/UUP, SPY/GLD, GLD/USO, HYG/TLT, XLF/TLT, SPY/TLT, USO/UUP | **family-wise significant** (log₁₀E up to **10.6**; detected as early as t\*≈230–360 days) |
| **Redundant within-equity** (both cyclical) | XLE/XLF, XLE/XLB, XLF/XLI, XLK/XLI | **not flagged** (synergy ≤ 0 — cross-term redundant with individual variances) |

Overall **8/14** pairs family-wise significant, **10/14** raw — and every one is an
economically-distinct cross-asset-class pair, exactly where the covariance term is
a real, separate component of risk (the diversification / Markowitz effect). This
is a genuine positive detection of synergy in real market data, with anytime-valid
error control. Figures: `pv_synergy_bars.png`, `pv_event_trajectories.png`; data
`portfolio_variance_results.json`.

## Real financial data — the lagged (Granger) direction is additive+redundant

**Data.** 16-asset daily panel, 2016-07-26 … 2026-07-23 (2,512 days): SPY, 9
sector ETFs, TLT, GLD, USO, HYG, UUP, ^VIX (`data/download_financial.py`), plus 6
world equity indices (US/Europe/Asia, 2,138 aligned days,
`data/world_indices_returns.csv`). Analysis on a daily volatility proxy
(|return|, standardized) and on standardized returns, conditioned on the market.

**Finding — daily cross-asset dependence is additive + redundant, not
synergistic.**

- **Exhaustive volatility scan** (1,365 target×source-pair triples, beyond
  market): **2** raw sequential rejections, **0** family-wise significant
  (e-value Bonferroni). Strongest evidence only log₁₀E ≈ 0.8 (e-value ≈ 6, below
  the 1/α = 20 bar). The in-sample top-ranked triples have **negative
  out-of-sample synergy** — they do not replicate on held-out data — while their
  pairwise Granger p-values are ≈ 0 (strong *individual* spillover).
- **Cross-region indices** (prior-day US & Europe → next-day Asia): strong
  pairwise lead-lag (p ≈ 0 for both) but **negative** synergy — the two regions
  carry *redundant* information about Asia, so the joint model does not beat the
  sum of parts.
- **Economically-motivated mechanisms** (oil×dollar→energy/gold/materials,
  vol×credit→financials, leverage: signed market-return×VIX→sector volatility):
  all batch-synergy estimates tiny (≤ 0.014 standardized-loss units), **none**
  rejected. The largest hint is the leverage effect on energy-sector volatility.
- **Hourly crypto** (`crypto_case_study.py`; 16,761 hours × 10 assets, BTC-
  conditioned): 360 triples in each of returns and volatility, **0** family-wise
  significant in either (top log₁₀E 0.21 returns / 0.34 volatility). Even in the
  inefficient, high-sample crypto regime, lagged super-additive synergy is absent.

**Interpretation.** The two real-data findings together tell a coherent story.
*Contemporaneously*, genuine synergy exists and ANTE detects it strongly and
selectively — portfolio variance is irreducibly joint through the covariance term
(diversification). *In the lagged/predictive (Granger) direction*, synergy is
absent across daily equities and hourly crypto: apparent in-sample synergies do
not replicate out-of-sample and none survive anytime-valid control. So ANTE both
(i) fires on real synergy that is present and (ii) refuses to over-report where the
structure is really additive + common-factor redundancy — a cautionary counterpoint
to in-sample batch synergy rankings (e.g. PID over triplets, Scagliarini et al.
2020). Substantive empirical claim: **cross-asset dependence is synergistic
*contemporaneously* (through covariance) but additive + redundant in the
*predictive* direction at daily/hourly frequency.**

**Figures.** `sg_s3_power_latency.png`, `sg_s4_paths.png` (synthetic);
`fin_synergy_network.png`, `fin_oos_validation.png`, `fin_event_trajectory.png`,
`fin_mechanism_probe.png` (real data).

## Caveats / next steps

- Effects are tested at daily frequency, lag p=1–2, with linear+interaction
  forgetting-RLS predictors. Synergy could be larger intraday, at longer lags, or
  under nonlinear (kernel/neural) predictors — all drop into the same betting
  scaffold and are natural extensions.
- The negative real-data result strengthens, rather than weakens, the
  methodological contribution (a rigorous instrument that does not over-report);
  a positive real-data demonstration at higher frequency remains desirable before
  submission.
