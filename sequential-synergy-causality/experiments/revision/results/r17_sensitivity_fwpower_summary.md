# R17 summary

## (a) sensitivity to truncation level and winsorisation (200 hypotheses each)
### stocks13_vol
| variant | raw rejections | median final log10E | max log10E |
|---|---|---|---|
| base (c=3, winsor 5) | 1/200 | -0.23 | 1.55 |
| c=2 | 2/200 | -0.33 | 1.46 |
| c=5 | 0/200 | -0.15 | 0.65 |
| no winsorisation | 0/200 | -0.23 | 1.16 |
| c=5, no winsorisation | 0/200 | -0.18 | 0.97 |
### etf16_ret
| variant | raw rejections | median final log10E | max log10E |
|---|---|---|---|
| base (c=3, winsor 5) | 1/200 | -0.25 | 1.80 |
| c=2 | 1/200 | -0.31 | 1.37 |
| c=5 | 0/200 | -0.20 | 1.08 |
| no winsorisation | 0/200 | -0.26 | 1.23 |
| c=5, no winsorisation | 0/200 | -0.23 | 0.88 |

## (b) synthetic power at family-wise thresholds (150 reps)
| c | 1/alpha | m=660 | m=4400 |
|---|---|---|---|
| 0.15 | 0.37 [0.29,0.45] | 0.01 [0.00,0.04] | 0.00 [0.00,0.02] |
| 0.2 | 0.83 [0.77,0.88] | 0.06 [0.03,0.11] | 0.03 [0.01,0.08] |
| 0.3 | 0.99 [0.96,1.00] | 0.77 [0.69,0.83] | 0.61 [0.53,0.68] |
| 0.45 | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] |
