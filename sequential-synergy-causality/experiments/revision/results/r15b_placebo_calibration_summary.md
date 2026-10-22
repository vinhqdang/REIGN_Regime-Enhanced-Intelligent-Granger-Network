# R15 summary: real vs circular-shift placebo (all nulls true under placebo)

quantiles (5,25,50,75,95,100%) of log10 of running-max E; raw rejection = max E >= 20

## crypto10_ret
- real   log10 max E quantiles: [0.0, 0.04, 0.2, 1.62, 14.3, 15.0]; raw rejection 53/200
- placebo log10 max E quantiles: [0.0, 0.04, 0.4, 4.26, 15.0, 15.0]; raw rejection 76/200 [0.316, 0.449]
- real   log10 final E quantiles: [-1.21, -0.59, -0.16, 0.0, 11.12, 15.0]
- placebo log10 final E quantiles: [-1.0, -0.44, -0.03, 2.68, 14.48, 15.0]

## crypto10_vol
- real   log10 max E quantiles: [0.0, 0.19, 1.73, 6.46, 15.0, 15.0]; raw rejection 108/200
- placebo log10 max E quantiles: [0.0, 0.08, 0.44, 6.67, 15.0, 15.0]; raw rejection 79/200 [0.330, 0.464]
- real   log10 final E quantiles: [-1.01, -0.43, -0.14, 4.16, 15.0, 15.0]
- placebo log10 final E quantiles: [-1.08, -0.48, -0.1, 4.88, 15.0, 15.0]

## etf16_tail
- real   log10 max E quantiles: [0.0, 0.07, 0.4, 2.12, 9.19, 15.0]; raw rejection 68/200
- placebo log10 max E quantiles: [0.0, 0.03, 0.24, 0.75, 2.81, 14.14]; raw rejection 28/200 [0.099, 0.195]
- real   log10 final E quantiles: [-2.31, -0.92, -0.35, -0.06, 0.36, 15.0]
- placebo log10 final E quantiles: [-1.57, -0.86, -0.4, -0.1, 0.53, 4.68]

## stocks13_tail
- real   log10 max E quantiles: [0.0, 0.06, 0.42, 1.53, 6.27, 15.0]; raw rejection 56/200
- placebo log10 max E quantiles: [0.0, 0.02, 0.12, 0.54, 2.05, 15.0]; raw rejection 19/200 [0.062, 0.144]
- real   log10 final E quantiles: [-3.13, -1.08, -0.52, -0.19, 0.0, 4.92]
- placebo log10 final E quantiles: [-1.62, -0.82, -0.44, -0.14, 0.0, 14.89]

## hourly12_tail
- real   log10 max E quantiles: [0.0, 0.06, 0.39, 1.77, 6.68, 12.16]; raw rejection 65/200
- placebo log10 max E quantiles: [0.0, 0.05, 0.38, 1.56, 7.31, 15.0]; raw rejection 56/200 [0.222, 0.346]
- real   log10 final E quantiles: [-3.61, -1.29, -0.45, -0.11, 0.0, 3.44]
- placebo log10 final E quantiles: [-2.88, -0.93, -0.4, -0.1, 0.23, 3.24]

## hourly12_ret
- real   log10 max E quantiles: [0.0, 0.0, 0.02, 0.12, 0.38, 1.34]; raw rejection 1/200
- placebo log10 max E quantiles: [0.0, 0.0, 0.03, 0.15, 0.46, 1.63]; raw rejection 1/200 [0.001, 0.028]
- real   log10 final E quantiles: [-1.14, -0.57, -0.2, -0.01, 0.0, 0.3]
- placebo log10 final E quantiles: [-0.91, -0.48, -0.23, -0.05, 0.0, 0.03]
