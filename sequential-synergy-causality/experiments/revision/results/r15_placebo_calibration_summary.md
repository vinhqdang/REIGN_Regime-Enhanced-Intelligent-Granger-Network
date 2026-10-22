# R15 summary: real vs circular-shift placebo (all nulls true under placebo)

quantiles (5,25,50,75,95,100%) of log10 of running-max E; raw rejection = max E >= 20

## stocks13_vol
- real   log10 max E quantiles: [0.0, 0.0, 0.04, 0.19, 0.51, 1.59]; raw rejection 2/250
- placebo log10 max E quantiles: [0.0, 0.0, 0.04, 0.19, 0.6, 1.3]; raw rejection 0/250 [0.000, 0.015]
- real   log10 final E quantiles: [-1.1, -0.62, -0.27, -0.06, 0.0, 0.67]
- placebo log10 final E quantiles: [-1.09, -0.55, -0.24, -0.02, 0.0, 0.9]

## etf16_ret
- real   log10 max E quantiles: [0.0, 0.0, 0.06, 0.19, 0.62, 1.8]; raw rejection 2/250
- placebo log10 max E quantiles: [0.0, 0.0, 0.04, 0.15, 0.49, 1.01]; raw rejection 0/250 [0.000, 0.015]
- real   log10 final E quantiles: [-1.08, -0.65, -0.33, -0.09, 0.0, 0.45]
- placebo log10 final E quantiles: [-0.99, -0.56, -0.26, -0.06, 0.0, 0.56]

## hourly12_vol
- real   log10 max E quantiles: [0.0, 0.0, 0.06, 0.29, 1.15, 3.82]; raw rejection 10/250
- placebo log10 max E quantiles: [0.0, 0.0, 0.04, 0.25, 1.18, 2.88]; raw rejection 9/250 [0.019, 0.067]
- real   log10 final E quantiles: [-1.12, -0.54, -0.25, -0.04, 0.0, 0.84]
- placebo log10 final E quantiles: [-0.98, -0.46, -0.2, -0.0, 0.0, 1.01]
