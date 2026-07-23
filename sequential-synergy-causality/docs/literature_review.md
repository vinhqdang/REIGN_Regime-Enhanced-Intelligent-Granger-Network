# Literature Review: Group (Synergistic) Causality and the Case for a Sequential Test

**Scope.** This review surveys the three research traditions that formalize
*group causality* — the idea that a set of causes acts jointly, not merely as the
sum of pairwise $A\!\to\!C$ effects — and the sequential-testing machinery we
propose to bring to bear on it. It closes with a gap analysis motivating a
sequential, anytime-valid (e-process) test for a synergy null.

> **Verification note.** Every citation below was checked against a primary
> source (publisher page, arXiv, CrossRef, or PubMed). Items are tagged
> **[V]** verified or **[P]** preprint/not-yet-in-journal. Two second-hand
> attributions from prior correspondence were corrected during verification and
> are flagged inline.

---

## 1. Sufficient-cause / synergy interaction (epidemiology & biostatistics)

The oldest formalization of "$A$ and $B$ together cause $C$" is Rothman's
**sufficient-component cause** ("causal pie") model **[V]** (Rothman, 1976,
*Am. J. Epidemiology*), in which synergy means two factors are component causes
of the *same* sufficient cause, so their joint presence produces an effect
exceeding the sum of their separate contributions.

The definitive modern axiomatization is **VanderWeele & Richardson (2012)**,
*"General theory for interactions in sufficient cause models with dichotomous
exposures,"* *Annals of Statistics* 40(4):2128–2161 **[V]** — counterfactual and
empirical conditions for sufficient-cause interaction (and *singular*
interaction) among an arbitrary number of binary causes. **This is the batch
formalization our sequential null generalizes.**

On the estimation side, additive-scale interaction is measured by the **relative
excess risk due to interaction (RERI)**, whose canonical inferential treatment is
Hosmer & Lemeshow (1992), *Epidemiology* 3(5):452–456 **[V]**, with a widely used
modern tutorial by VanderWeele & Knol (2014), *Epidemiologic Methods* **[V]**.
For mechanistic (sufficient-cause) interaction under case-control sampling, the
**PRISM** measure (Lee, 2013, *PLoS ONE* 8(6):e67424 **[V]**) and its
non-rare-disease case-control test (Lin & Lee, 2018, *Sci. Rep.* 8:9223 **[V]**)
are the reference methods.

The most recent movement generalizes synergy beyond two factors:
**La Torre & D'Urso (2026)**, *"Evaluating synergistic effects among multiple
factors in disease causation: a new approach using a generalized synergy index,"*
*European Journal of Epidemiology*, DOI 10.1007/s10654-026-01405-2 **[V]** —
extends Rothman's two-factor additive synergy index $S$ to an arbitrary number of
dichotomous factors. Framed as an essay with a single small validation cohort;
the concept and formula are solid but not yet stress-tested at scale.

**Common limitation.** Every method here is **fixed-sample**: a point estimate
and CI at a pre-specified $n$. The papers themselves note multi-way interaction
becomes unstable at small $n$ — exactly the regime where a sequential method that
spends evidence adaptively would help.

---

## 2. Joint / multi-cause causal effects (statistics)

Rather than *testing for* synergy, this tradition *estimates the effect of a
joint intervention* $\mathrm{do}(\{A,B,\dots\})$.

* **Causal aggregation** — Roquero Gimenez & Rothenhäusler (2023), *JMLR* 24
  (arXiv:2106.03024) **[V]**: estimates joint-intervention effects by fusing
  experiments that each manipulate only a few variables — directly a
  "sum-of-parts vs. joint" question. *This is the correct citation for the
  joint-intervention-effect estimand* (an earlier second-hand attribution of
  "Average Joint Effect" to the cGNF paper arXiv:2401.06864, Balgi et al. 2024
  *"Deep Learning With DAGs"* **[P]**, was inaccurate — that paper does not
  define such an estimand).
* **Joint potential-outcome identification** — Wu & Mao (2025), *"The Promises of
  Multiple Experiments: Identifying Joint Distribution of Potential Outcomes,"*
  arXiv:2504.20470 **[P]**: nonparametric identification of the *joint*
  distribution of potential outcomes using cross-experiment variation; the
  closest thing to a "probability of joint causation" estimator. A single-study
  companion appears as arXiv:2509.20506 **[P]**.

**Common limitation.** One-shot, fixed-design estimators. No sequential
estimator with time-uniform coverage for a joint effect exists, even though the
single-treatment anytime-valid ATE literature (§4) is mature.

---

## 3. Information-theoretic synergy (PID and dynamical extensions)

Here synergy is an information quantity separable from redundancy and unique
information, via **Partial Information Decomposition (PID)** — Williams & Beer
(2010), *"Nonnegative Decomposition of Multivariate Information,"* arXiv:1004.2515
**[P]** (foundational; never journal-published).

Two recent dynamical/causal extensions:

* **SURD** — Martínez-Sánchez, Arranz & Lozano-Durán (2024), *"Decomposing
  causality into its synergistic, unique, and redundant components,"* *Nature
  Communications* 15:9296, DOI 10.1038/s41467-024-53373-4 **[V]**: decomposes
  causality into redundant/unique/synergistic increments of information, robust to
  nonlinearity, colliders, and exogenous influence.
* **PEID** — Yang, Wang & Zhang (2026), *"Partial Effective Information
  Decomposition for Synergistic Causality,"* arXiv:2605.03267 **[P]** (Beijing
  Normal University): an *interventional* analogue of PID. Its key result — under
  independent maximum-entropy interventions, source-side redundancy vanishes, so
  synergy is read off directly as $EI(A,B\!\to\!C)-\sum_i EI(T^{(i)}\!\to\!C)$ —
  and its **hierarchical additivity** theorem ($\mathrm{Syn}_{\mathcal P'} =
  \mathrm{Syn}_{\mathcal P}+\mathrm{Syn}_{\mathcal R}$) give exactly the additive
  increment structure our subset-refinement e-process exploits. Supports causal
  hypergraphs / downward causation.

A cautionary result motivating a test-based (rather than lattice-based) approach:
Lyu, Clark & Raviv (2026), *"Multivariate Partial Information Decomposition:
Constructions, Inconsistencies, and Alternative Measures,"* *Phys. Rev. E* 113:034102
**[V]** — the PID lattice becomes inconsistent for 4+ variables.

**Common limitation.** These are *estimators*, not tests: PEID/SURD report a
point synergy value with no null distribution, p-value, or stopping rule. There
is no way to say "we now have enough evidence" for a synergy claim.

---

## 4. Sequential testing, e-processes, testing by betting (the backbone)

The machinery we propose to import.

* **Ville (1939)** **[V]** — martingales and the supermartingale maximal
  inequality $P(\sup_t M_t \ge 1/\alpha)\le\alpha$ underlying all anytime-valid
  inference.
* **Shafer (2021)**, *"Testing by betting,"* *JRSS-A* 184(2):407–431, DOI
  10.1111/rssa.12647 **[V]** — the betting reframing of testing.
* **Waudby-Smith & Ramdas (2024)**, *"Estimating means of bounded random
  variables by betting,"* *JRSS-B* 86(1):1–27, DOI 10.1093/jrsssb/qkad009 **[V]**
  — the hedged-capital construction and GRAPA betting strategy our interaction
  e-process is built on.
* **Ramdas, Grünwald, Vovk & Shafer (2023)**, *"Game-theoretic statistics and
  safe anytime-valid inference,"* *Statistical Science* 38(4):576–601, DOI
  10.1214/23-STS894 **[V]** — the survey of e-processes / test martingales / CS.
* **E-values & multiple testing** — Vovk & Wang (2021), *"E-values: calibration,
  combination and applications,"* *Ann. Statist.* 49(3):1736–1754 **[V]**; Wang &
  Ramdas (2022), *"False discovery rate control with e-values"* (e-BH), *JRSS-B*
  84(3):822–852 **[V]** — how we merge/calibrate the per-subset e-values in the
  refinement search under arbitrary dependence.

**Closest prior art (anytime-valid causal inference).** Waudby-Smith, Wu,
Ramdas, Karampatziakis & Mineiro (2024), *"Anytime-valid off-policy inference for
contextual bandits,"* DOI 10.1145/3643693 **[V]**; and anytime-valid ATE / bandit
work (Molitor & Gordon, arXiv:2408.09598; arXiv:2311.05794) **[P]**. All target
**single-effect** estimands (ATE, off-policy mean) — none an interaction or
synergy contrast.

---

## 5. Gap analysis and positioning

A focused prior-art search (terms: *anytime-valid interaction test*, *e-process
causal interaction*, *sequential test synergy*, *confidence sequence interaction
effect*, *test martingale synergistic*) found **no** sequential / anytime-valid /
e-process test whose null is "the joint causal effect of a set of causes equals
the sum of its parts." The two relevant bodies of work do not intersect:

| Literature | Handles synergy? | Sequential / anytime-valid? |
|---|---|---|
| Sufficient-cause / RERI / PRISM / generalized synergy index (§1) | ✅ | ❌ fixed-$n$ |
| Joint effects / causal aggregation / joint PO (§2) | ✅ (joint effect) | ❌ one-shot |
| PID / SURD / PEID (§3) | ✅ (info scale) | ❌ point estimate, no inference |
| Testing by betting / e-processes / anytime-valid ATE (§4) | ❌ single effects only | ✅ |

**Near-misses** (cite to show diligence): anytime-valid off-policy/ATE inference
(§4, single effects); betting tests for other structured nulls — stochastic
dominance (arXiv:2604.21851 **[P]**), sequential independence (arXiv:2305.13818
**[P]**), interactive rank tests (arXiv:2009.05892 **[P]**); sequential
randomization/e-value trial monitoring (arXiv:2512.04366 **[P]**). These show the
betting scaffold has been pushed to composite/structural nulls — good
construction templates — but **none test additive/multiplicative interaction or
sufficient-cause synergy.**

**Contribution positioning.**
1. A single betting-martingale test of the synergy null that instantiates on
   *both* the additive causal-pie scale (RERI-type contrast, exactly identified
   and bounded) and the information-theoretic scale (PEID-type increment) —
   showing they share one testable null.
2. An anytime-valid **subset-refinement** search that exploits PEID's
   hierarchical-additivity to avoid the exponential PID lattice, with e-value
   merging for family-wise/FDR control under dependence.
3. **Observational** (AIPW, cross-fitted) and **graph/GNN** instantiations — the
   latter answering "do nodes $A,B$ *jointly* cause the anomaly at $C$?", which no
   existing group-causality method addresses sequentially.

**Honest framing for the paper.** Since this is an absence-of-evidence claim, state
it as *"we are unaware of any anytime-valid test for causal interaction/synergy,"*
cite VanderWeele–Richardson (2012) as the batch null we generalize, and cite
Waudby-Smith & Ramdas (2024) + the anytime-valid ATE papers as the machinery we
extend.

---

## 6. Consolidated reference list

1. Rothman, K. J. (1976). Causes. *American Journal of Epidemiology*, 104(6), 587–592. https://doi.org/10.1093/oxfordjournals.aje.a112335
2. VanderWeele, T. J., & Richardson, T. S. (2012). General theory for interactions in sufficient cause models with dichotomous exposures. *The Annals of Statistics*, 40(4), 2128–2161. https://doi.org/10.1214/12-AOS1019
3. Hosmer, D. W., & Lemeshow, S. (1992). Confidence interval estimation of interaction. *Epidemiology*, 3(5), 452–456. https://doi.org/10.1097/00001648-199209000-00012
4. VanderWeele, T. J., & Knol, M. J. (2014). A tutorial on interaction. *Epidemiologic Methods*, 3(1), 33–72. https://doi.org/10.1515/em-2013-0005
5. Lee, W.-C. (2013). Assessing causal mechanistic interactions: A peril ratio index of synergy based on multiplicativity. *PLoS ONE*, 8(6), e67424. https://doi.org/10.1371/journal.pone.0067424
6. Lin, J.-H., & Lee, W.-C. (2018). Testing for sufficient-cause interactions in case-control studies of non-rare diseases. *Scientific Reports*, 8, 9223. https://doi.org/10.1038/s41598-018-27660-2
7. La Torre, G., & D'Urso, P. (2026). Evaluating synergistic effects among multiple factors in disease causation: a new approach using a generalized synergy index. *European Journal of Epidemiology*. https://doi.org/10.1007/s10654-026-01405-2
8. Roquero Gimenez, J., & Rothenhäusler, D. (2023). Causal aggregation: estimation and inference of causal effects by constraint-based data fusion. *Journal of Machine Learning Research*, 24. https://arxiv.org/abs/2106.03024
9. Wu, P., & Mao, X. (2025). The promises of multiple experiments: Identifying joint distribution of potential outcomes. arXiv:2504.20470.
10. Balgi, S., Daoud, A., Peña, J. M., Wodtke, G. T., & Zhou, J. (2024). Deep learning with DAGs. arXiv:2401.06864.
11. Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv:1004.2515.
12. Martínez-Sánchez, Á., Arranz, G., & Lozano-Durán, A. (2024). Decomposing causality into its synergistic, unique, and redundant components. *Nature Communications*, 15, 9296. https://doi.org/10.1038/s41467-024-53373-4
13. Yang, M., Wang, S., & Zhang, J. (2026). Partial effective information decomposition for synergistic causality. arXiv:2605.03267.
14. Lyu, A., Clark, A., & Raviv, N. (2026). Multivariate partial information decomposition: Constructions, inconsistencies, and alternative measures. *Physical Review E*, 113, 034102. https://doi.org/10.1103/8rzp-w5z1
15. Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
16. Shafer, G. (2021). Testing by betting: A strategy for statistical and scientific communication. *Journal of the Royal Statistical Society: Series A*, 184(2), 407–431. https://doi.org/10.1111/rssa.12647
17. Waudby-Smith, I., & Ramdas, A. (2024). Estimating means of bounded random variables by betting. *Journal of the Royal Statistical Society: Series B*, 86(1), 1–27. https://doi.org/10.1093/jrsssb/qkad009
18. Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601. https://doi.org/10.1214/23-STS894
19. Vovk, V., & Wang, R. (2021). E-values: Calibration, combination and applications. *The Annals of Statistics*, 49(3), 1736–1754. https://doi.org/10.1214/20-AOS2020
20. Wang, R., & Ramdas, A. (2022). False discovery rate control with e-values. *Journal of the Royal Statistical Society: Series B*, 84(3), 822–852. https://doi.org/10.1111/rssb.12489
21. Waudby-Smith, I., Wu, L., Ramdas, A., Karampatziakis, N., & Mineiro, P. (2024). Anytime-valid off-policy inference for contextual bandits. *ACM/IMS Journal of Data Science*. https://doi.org/10.1145/3643693

*(Preprints without a journal DOI — items 9, 10, 13, and the near-miss arXiv
entries in §5 — should be re-checked for a peer-reviewed version before final
submission.)*
