# ANTE-SG: Anytime-valid Detection of Synergistic Granger Causality in Financial Time Series

This is the flagship algorithm of the ANTE paper — a new method for the financial
time-series setting. It sits on the same testing-by-betting foundation as the
interventional ANTE test (`methodology.md`) but targets a different, observational
object: **super-additive (synergistic) Granger causality among groups of assets**,
detected sequentially and anytime-valid, robust to market nonstationarity.

---

## 1. What problem it solves (and why existing methods don't)

**Question.** Do two series $A,B$ *jointly* Granger-cause a target $Y$ **beyond the
sum of their individual contributions** — i.e. is there predictive structure in
the pair that neither carries alone — and can we detect it *as the data streams*,
stopping the moment the evidence is decisive, under nonstationary markets?

**State of the art (see `literature_review_finance.md`).** The synergy has been
*measured* in markets — Scagliarini et al. (2020, *Entropy*) compute PID synergy
over index triplets; Stramaglia et al. (2014, *New J. Phys.*) show pairwise
Granger *misses* synergy and needs conditioning; SURD (Martínez-Sánchez et al.
2024, *Nat. Commun.*) formalizes synergistic/unique/redundant causal information.
But these are **batch, low-order, and give no hypothesis test with error control**.
On the other axis, anytime-valid sequential inference has reached Granger
causality (Jha 2026, distributional channels of a *single* source) but **not
synergy among groups**. No method combines (i) an explicit super-additive
synergy null, (ii) an e-process / testing-by-betting anytime-valid guarantee, and
(iii) regime-adaptivity for finance. **ANTE-SG occupies exactly that hole.**

---

## 2. The synergy object

Panel of (standardized) returns; target $Y$, candidate sources $A,B$, optional
conditioning set $Z$ (e.g. the market), lag order $p$. For a source set
$S\subseteq\{A,B\}$ let $L(S)$ be the expected one-step-ahead loss of the best
predictor of $Y_t$ from $\{Y,Z\}$-past **plus** the lagged histories of $S$, and
the predictive gain $\Delta(S)=L(\varnothing)-L(S)$. Define

$$\mathrm{Syn}(A,B\!\to\!Y)\;=\;\Delta(\{A,B\})-\Delta(\{A\})-\Delta(\{B\}).$$

$\mathrm{Syn}>0$ means the joint predictor beats the sum of the marginal
predictors — genuine synergy. For Gaussian variables Granger causality equals
transfer entropy (Barnett–Barrett–Seth 2009), so $\mathrm{Syn}$ is an interaction
of conditional transfer entropies — the same quantity SURD/PID call synergistic
information, here recast as an *out-of-sample predictive* contrast.

**Isolating synergy from within-variable nonlinearity.** Each marginal model
receives its source's own linear and squared lags; only the joint model receives
the **cross-products** $A_{t-i}B_{t-i}$. Hence a mechanism $Y=g(A)$ (however
nonlinear) is absorbed by the $A$-model and is *not* counted as synergy — only
irreducible $A\times B$ interaction is. (Validated in synthetic experiment S1:
additive and $Y=A^2$ DGPs are not flagged; $Y=A\cdot B$ is.)

---

## 3. The algorithm

Online, at each $t$ we hold four **forgetting recursive-least-squares** predictors
of $Y_t$ (base $+Z$, $+A$, $+B$, $+AB$), fit only on the past. We predict $Y_t$
*before* updating (so losses are genuinely out-of-sample), observe $Y_t$, and form
the per-step synergy score

$$s_t=\ell^{A}_t+\ell^{B}_t-\ell^{AB}_t-\ell^{\text{base}}_t,\qquad
\mathbb{E}[s_t\mid\mathcal F_{t-1}]=\mathrm{Syn}\text{ (prequentially)}.$$

**Null (game-theoretic / prequential).** $H_0:\mathbb E[s_t\mid\mathcal F_{t-1}]\le0$
for all $t$ — the joint predictor has no super-additive advantage given the past.
This requires **neither stationarity nor correct model specification**; it is a
statement about realized predictability, which is what makes it valid on
nonstationary financial data.

**Test by betting.** With a predictable, one-sided betting fraction $\lambda_t\ge0$
and a scale-normalized, clipped score $u_t=\mathrm{clip}(s_t/\hat\sigma_{t-1},-b,b)$,

$$E_t=\prod_{s\le t}\bigl(1+\lambda_s u_s\bigr)$$

is a nonnegative supermartingale under $H_0$; by Ville's inequality
$P_{H_0}(\exists t:E_t\ge1/\alpha)\le\alpha$. Reject (declare synergy) the first
time capital crosses $1/\alpha$ — valid under continuous monitoring and optional
stopping. $\lambda_t$ uses a truncated-Kelly/GRAPA rule on past scores.

**Regime-adaptivity.** The forgetting factor in the RLS predictors down-weights
stale data, so the test tracks synergy that switches on/off across market regimes
(e.g. appears only in crises) rather than averaging it away.

**Pseudocode.**
```
init 4 forgetting-RLS predictors (base+Z, +A, +B, +AB); E = 1
for t = p+1 .. T:
    predict Y_t from each model  ->  losses l_base,l_A,l_B,l_AB   (out-of-sample)
    s_t = l_A + l_B - l_AB - l_base
    u_t = clip(s_t / scale_{t-1}, -b, b)                          # predictable scale
    lambda_t = clip(max(mean_{<t}(s)/var_{<t}(s),0), 0, gamma/b)  # predictable
    E *= (1 + lambda_t * u_t)
    if E >= 1/alpha: report synergy at t (anytime-valid)
    update all predictors and running stats with (features_t, Y_t)
```

**Groups and networks.** Scanning every (target; source-pair) triple yields a
*synergy network*; per-triple e-values are combined with an e-value Bonferroni
(family-wise: report $E\ge m/\alpha$) or e-BH for FDR, both anytime-valid under
dependence. Higher-order groups follow via the same score with larger cross-terms
(O-information style) — future work.

---

## 4. Baselines it is compared against

| Baseline | What it represents | Why ANTE-SG differs |
|---|---|---|
| Pairwise Granger F-test | individual $A\!\to\!Y$, $B\!\to\!Y$ | cannot express *joint super-additivity*; misses synergy (Stramaglia 2014) |
| Batch synergy estimate ($\mathrm{Syn}$ + permutation) | fixed-sample synergy à la Scagliarini/SURD | no anytime-valid monitoring; one-shot |
| Continuously-monitored fixed-sample synergy $z$-test | naive "peek every day" | inflates type-I error massively (shown in S2) |
| Conditioned (market-controlled) Granger | removes common-factor redundancy | still additive, not a synergy test |

---

## 5. Experiments

**Synthetic (`synthetic_synergy.py`).**
- **S1 discrimination:** additive and $Y=A^2$ DGPs not flagged; $Y=A\cdot B$ flagged.
- **S2 anytime-validity:** e-process type-I $\le\alpha$ vs. inflated peeking test.
- **S3 power/latency:** power rises and detection time falls with synergy strength.
- **S4 trajectories:** capital flat under nulls, grows under synergy.

**Real financial data (`financial_case_study.py`).** 16-asset, 10-year daily panel
(sector ETFs + macro/vol/commodity drivers), conditioned on SPY so discoveries are
synergy *beyond* the market factor. We report: the ranked synergy network with
family-wise e-value control; a held-out **out-of-sample predictive-gain** check
(positive ⇒ the synergy is real structure, not overfit — the field-standard
no-ground-truth evaluation); the **running e-value across 2016–2026** annotated
with the COVID crash and 2022 rate shock (regime/event validation); and a contrast
with pairwise Granger to expose pairs whose *joint* effect is significant while
individual effects are weak.

---

## 6. Results (summary — full detail in `RESULTS.md`)

**Synthetic.** ANTE-SG controls type-I (rejection 0.00 additive, 0.01 for
$Y=A^2$) and powers up on genuine cross-interaction (0.71 on $Y=A\cdot B$), with
power rising to ~0.8 and detection latency falling as synergy strengthens. The
own-nonlinearity discriminator confirms it isolates irreducible $A\times B$
structure rather than within-variable nonlinearity.

**Real financial data (10y, 16-asset panel + 6 world indices).** Across an
exhaustive 1,365-triple volatility scan, cross-region index lead-lag, and curated
economic mechanisms (oil×dollar, rates×dollar, vol×credit, leverage), daily
cross-asset dependence is **additive and redundant, not synergistic**: apparent
in-sample synergies do not replicate out-of-sample and none survive anytime-valid
family-wise control. ANTE-SG thus acts as a rigorous filter that avoids the false
synergy discoveries a batch PID ranking can make — a specificity result on real
data and a substantive empirical claim about daily markets. See `RESULTS.md`.
