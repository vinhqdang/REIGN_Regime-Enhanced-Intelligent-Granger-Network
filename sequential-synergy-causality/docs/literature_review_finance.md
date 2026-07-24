# SOTA Review: Synergistic Granger Causality in Finance (for ANTE-SG)

Verification tags: **[V]** confirmed on a primary source (arXiv/publisher/DBLP);
**[U]** appeared in listings but not opened this pass — re-verify before citing.

## 1. Synergy / higher-order information causality in finance
- **Scagliarini, Faes, Marinazzo, Stramaglia, Mantegna (2020).** "Synergistic
  Information Transfer in the Global System of Financial Markets." *Entropy*
  22(9):1000. DOI 10.3390/e22091000. **[V]** — PID synergy over index *triplets*
  across 17 world markets; finds European+American pairs synergistically drive
  Asian markets. **Batch, low-order, no sequential inference — the primary
  position-and-beat target.**
- **Stramaglia, Cortes, Marinazzo (2014).** "Synergy and redundancy in the
  Granger causal analysis of dynamical networks." *New J. Phys.* 16:105003.
  arXiv:1403.5156. **[V]** — pairwise Granger *misses* synergy; conditioning is
  needed. Motivates why "group beyond pairwise" needs its own estimator.
- **Barnett, Barrett, Seth (2009).** "Granger Causality and Transfer Entropy Are
  Equivalent for Gaussian Variables." *PRL* 103:238701. arXiv:0910.4514. **[V]** —
  lets us move between the VAR-Granger and transfer-entropy framings of synergy.
- **Rosas, Mediano, Gastpar, Jensen (2019).** O-information. *Phys. Rev. E*
  100:032305. arXiv:1902.11239. **[V]** — scalable synergy/redundancy balance for
  groups > 3; the natural tool for higher-order extensions.
- **Santoro, Battiston, Petri, Amico (2023).** "Unveiling the higher-order
  organization of multivariate time series." *Nature Physics*. arXiv:2203.10702.
  DOI 10.1038/s41567-022-01852-0. **[V]**
- **Martínez-Sánchez, Arranz, Lozano-Durán (2024).** SURD. *Nat. Commun.* 15.
  arXiv:2405.12411. DOI 10.1038/s41467-024-53373-4. **[V]** — SOTA synergistic/
  unique/redundant causal-information decomposition (not financial, batch).

## 2. Granger causality with group / interaction structure
- **Tank, Covert, Foti, Shojaie, Fox (2021).** Neural Granger Causality. *IEEE
  TPAMI*. arXiv:1802.05842. DOI 10.1109/TPAMI.2021.3065601. **[V]** — nonlinear,
  group-lasso graph recovery; returns a binary graph, does not test super-
  additivity.
- **Lozano, Abe, Liu, Rosset (2009).** Grouped graphical Granger. *KDD*. DOI
  10.1145/1557019.1557085. **[U]** — group sparsity for variable selection, *not*
  a synergy test (a common confusion to disambiguate).

## 3. Regime-aware / nonstationary causal discovery in finance
- **Lee (2026).** "Regime-Dependent Predictive Structure Between Equity Factors:
  Evidence from Granger Causality." arXiv:2601.10732. **[V]** — HMM regimes +
  Granger on Fama-French factors; regime-aware but pairwise and batch.
- **Huang, Zhang, Glymour, Schölkopf et al.** CD-NOD, "Causal Discovery from
  Heterogeneous/Nonstationary Data." *JMLR* 21 (2020). **[U]**

## 4. Sequential / anytime-valid inference (the backbone) and near-misses
- **Jha (2026).** "Distributional Granger Causality: Identification, Sequential
  Inference, and Adaptive Testing." arXiv:2606.22230. **[V]** — the closest
  sequential-Granger work, but tests *distributional channels of a single
  source*, not group synergy. **Nearest miss.**
- **Waudby-Smith & Ramdas (2024).** "Estimating means of bounded random variables
  by betting." *JRSS-B* 86(1). arXiv:2010.09686. **[V]** — the betting/e-process
  machinery ANTE-SG's engine is built on.
- **Diebold & Yilmaz (2014).** "On the network topology of variance
  decompositions." *J. Econometrics* 182(1):119–134. DOI
  10.1016/j.jeconom.2014.04.012. **[V]** — the econometrics-standard connectedness
  benchmark; pairwise/variance-decomposition, not synergy-aware.

## GAP (confirmed)
No method combines (i) an explicit **super-additive synergy** Granger null, (ii)
an **e-process / testing-by-betting anytime-valid** guarantee, and (iii)
**regime-adaptivity** for nonstationary markets. Scagliarini 2020 = synergy +
finance, missing sequential + regime; Jha 2026 = sequential Granger, missing
synergy + group. **ANTE-SG threads the hole.**

## Baselines to compare against (BEAT LIST)
Scagliarini 2020 (PID triplet synergy) · Stramaglia 2014 (conditioned Granger) ·
Jha 2026 (sequential distributional Granger) · Tank 2021 (neural Granger) · SURD
2024 (batch synergy decomposition) · Diebold–Yilmaz 2014 (connectedness foil).

## Datasets used here
16-asset SPDR sector + macro/vol/commodity ETF panel and 6 world equity indices
(Yahoo Finance chart API, 10y daily). Other credible sources for extensions:
Ken French factor library, FRED-MD macro panel, Diebold–Yilmaz connectedness data.

> Items tagged **[U]** (Lozano 2009 KDD, CD-NOD/JMLR) appeared in search listings
> but were not opened this pass; confirm on the publisher/DBLP before submission.
