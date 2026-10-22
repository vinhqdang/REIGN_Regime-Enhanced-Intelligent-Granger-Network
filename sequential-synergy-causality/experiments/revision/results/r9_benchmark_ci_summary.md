# R9 summary: benchmark with CIs

Part A: 100 reps, T=3000. Part C: 250 reps, 15 looks, T=3000, alpha=0.05.

## Detection AUC (syn vs additive+own-nonlinear), bootstrap 95% CI

| method | AUC | CI | AUC syn vs own-nl | CI |
|---|---|---|---|---|
| ANTE_legacy | 0.992 | [0.983, 0.998] | 0.991 | [0.981, 0.998] |
| ANTE_v2_addjoint | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| SURD | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| PEID | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| InteractionInfo | 0.865 | [0.824, 0.906] | 0.730 | [0.655, 0.802] |

## Type-I under 15 looks (additive null), Wilson 95% CI

| procedure | rejections | rate | CI |
|---|---|---|---|
| ANTE-legacy | 5/250 | 0.020 | [0.009, 0.046] |
| ANTE-v2 addjoint (max E>=20) | 0/250 | 0.000 | [0.000, 0.015] |
| peeking batch permutation | 29/250 | 0.116 | [0.082, 0.162] |
| GS Pocock on raw addjoint score (one-sided) | 0/250 | 0.000 | [0.000, 0.015] |
| GS OBF on raw addjoint score (one-sided) | 0/250 | 0.000 | [0.000, 0.015] |

## Group-sequential boundaries (Monte Carlo, 200k paths)

| look | horizon | c Pocock | c OBF |
|---|---|---|---|
| 1 | 400 | 2.310 | inf |
| 2 | 585 | 2.432 | 4.462 |
| 3 | 771 | 2.412 | 3.673 |
| 4 | 956 | 2.380 | 3.307 |
| 5 | 1142 | 2.364 | 3.007 |
| 6 | 1328 | 2.332 | 2.780 |
| 7 | 1513 | 2.308 | 2.611 |
| 8 | 1699 | 2.303 | 2.458 |
| 9 | 1885 | 2.270 | 2.321 |
| 10 | 2070 | 2.253 | 2.204 |
| 11 | 2256 | 2.240 | 2.098 |
| 12 | 2442 | 2.223 | 2.023 |
| 13 | 2627 | 2.213 | 1.948 |
| 14 | 2813 | 2.214 | 1.885 |
| 15 | 2999 | 2.204 | 1.822 |

MC total crossing probability: Pocock 0.0500, OBF 0.0500

## Power on syn DGP (100 reps), Wilson 95% CI

| procedure | rejections | rate | CI |
|---|---|---|---|
| ANTE-legacy | 81/100 | 0.810 | [0.722, 0.875] |
| ANTE-v2 addjoint (max E>=20) | 100/100 | 1.000 | [0.963, 1.000] |
| peeking batch permutation | 63/100 | 0.630 | [0.532, 0.718] |
| GS Pocock on raw addjoint score (one-sided) | 99/100 | 0.990 | [0.946, 0.998] |
| GS OBF on raw addjoint score (one-sided) | 100/100 | 1.000 | [0.963, 1.000] |
