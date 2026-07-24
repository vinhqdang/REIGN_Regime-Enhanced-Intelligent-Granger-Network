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

## Real financial data (`financial_case_study.py`, `mechanism_probe.py`)

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

**Interpretation.** ANTE-SG, by requiring *out-of-sample super-additivity* and
*anytime-valid* evidence, correctly declines to declare synergy where the
apparent higher-order structure is explained by a common market factor plus
pairwise spillovers. This is a specificity result on real data and a cautionary
counterpoint to in-sample batch synergy rankings (e.g. PID synergy over triplets,
Scagliarini et al. 2020), which can surface "top synergistic" circuits that need
not survive out-of-sample testing. It is also a substantive empirical claim:
**at daily frequency, cross-asset financial dependence is dominated by additive
and redundant effects; irreducible super-additive synergy is negligible.**

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
