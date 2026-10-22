# R7 summary: Billio-style pairwise Granger connectedness

Pairwise tests: pairwise_granger_f, ordered pairs i->j, raw alpha=.05 and Bonferroni alpha/(N(N-1)).

## (i) Full-sample network

| panel | series | p | N | raw edges | raw DGC | Bonferroni edges | Bonferroni DGC |
|---|---|---|---|---|---|---|---|
| etf16 | ret | 1 | 16 | 134/240 | 0.558 | 47/240 | 0.196 |
| etf16 | ret | 5 | 16 | 172/240 | 0.717 | 82/240 | 0.342 |
| etf16 | vol | 1 | 16 | 215/240 | 0.896 | 186/240 | 0.775 |
| etf16 | vol | 5 | 16 | 197/240 | 0.821 | 157/240 | 0.654 |
| stocks13 | ret | 1 | 13 | 62/156 | 0.397 | 15/156 | 0.096 |
| stocks13 | ret | 5 | 13 | 69/156 | 0.442 | 16/156 | 0.103 |
| stocks13 | vol | 1 | 13 | 128/156 | 0.821 | 95/156 | 0.609 |
| stocks13 | vol | 5 | 13 | 117/156 | 0.750 | 84/156 | 0.538 |

## (ii) Rolling 252-day DGC (returns, p=1, raw .05, step 21)

| panel | windows | min DGC (end) | max DGC (end) | median | covid-overlap windows: mean [min, max] | 2022-overlap windows: mean [min, max] |
|---|---|---|---|---|---|---|
| etf16 | 108 | 0.004 (2023-06-28) | 0.546 (2020-05-27) | 0.104 | 0.398 [0.075, 0.546] (n=16) | 0.060 [0.004, 0.154] (n=24) |
| stocks13 | 108 | 0.019 (2022-02-24) | 0.404 (2021-02-25) | 0.080 | 0.312 [0.045, 0.404] (n=16) | 0.046 [0.019, 0.083] (n=24) |

## (iii) stocks13 target 'vol', lag1, cond SPY (660 triples)

- scan source: /home/user/REIGN_Regime-Enhanced-Intelligent-Granger-Network/sequential-synergy-causality/experiments/revision/results/r7_scan_stocks13_vol_lag1_recomputed.csv (recomputed)
- addjoint rejections raw (max E>=20): 4/660; family-wise (max E>=m/alpha=13200): 0/660
- four rejections raw: 6/660
- single-source F-tests: 132 unique (A->C) tests; Bonferroni alpha = 3.79e-04
- triples with BOTH A->C and B->C significant, unconditional F: raw 467, Bonferroni 260
- triples with BOTH significant, conditional on SPY F: raw 145, Bonferroni 29
- contingency (both F conditional Bonferroni) x (addjoint raw): {'both': 0, 'x_only': 29, 'y_only': 4, 'neither': 627}
- contingency (both F unconditional Bonferroni) x (addjoint raw): {'both': 2, 'x_only': 258, 'y_only': 2, 'neither': 398}
- median addjoint max log10E +0.04; median four max log10E +0.06
- per-triple table: results/r7_triples_stocks13_vol_lag1.csv
