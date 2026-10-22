# R8 summary: portfolio-variance control (explicit RV target)

Target RV_t=(0.5 r_A+0.5 r_B)^2, predictable standardisation, feat_clip=5, p=1, no conditioning. Thresholds: raw max E>=20; family-wise max E>=m/alpha=280 (m=14).

## Per pair

| pair | description | corr(rA,rB) | cov sign | R^2(rA rB on 1,rA,rB,rA^2,rB^2) | contemp addjoint final / max log10E | contemp four final / max log10E | lagged addjoint final / max log10E | lagged four final / max log10E |
|---|---|---|---|---|---|---|---|---|
| GLD/USO | gold x oil | +0.066 | +1 | 0.026 | +13.73 / +13.73 | +5.84 / +6.05 | +0.00 / +0.00 | +0.00 / +0.00 |
| XLF/TLT | financials x bonds | -0.262 | -1 | 0.404 | +15.00 / +15.00 | +14.98 / +15.00 | -0.11 / +0.00 | +0.21 / +0.27 |
| SPY/TLT | equity x bonds | -0.138 | -1 | 0.327 | +15.00 / +15.00 | +14.78 / +15.00 | -0.50 / +0.00 | +1.62 / +1.65 |
| GLD/TLT | gold x bonds | +0.246 | +1 | 0.101 | +13.51 / +15.00 | +5.29 / +7.62 | +0.77 / +1.04 | +0.81 / +1.15 |
| USO/UUP | oil x dollar | +0.020 | +1 | 0.196 | +10.97 / +10.97 | +4.22 / +6.10 | +0.00 / +0.00 | -0.71 / +0.05 |
| TLT/UUP | bonds x dollar | -0.240 | -1 | 0.446 | +15.00 / +15.00 | +14.83 / +15.00 | -0.00 / +0.00 | -0.19 / +0.80 |
| SPY/GLD | equity x gold | +0.106 | +1 | 0.241 | +14.76 / +15.00 | +7.06 / +8.89 | -0.12 / +0.21 | -0.18 / +0.00 |
| XLE/TLT | energy x bonds | -0.226 | -1 | 0.179 | +0.54 / +15.00 | +14.99 / +15.00 | -1.12 / +1.70 | +1.08 / +2.00 |
| HYG/TLT | credit x bonds | +0.088 | +1 | 0.105 | +15.00 / +15.00 | +9.68 / +12.56 | +0.31 / +1.14 | +1.08 / +1.30 |
| XLK/GLD | tech x gold | +0.118 | +1 | 0.216 | +15.00 / +15.00 | +8.28 / +9.40 | -0.18 / +0.00 | +0.00 / +0.00 |
| XLE/XLF | energy x financials (both cyclical) | +0.661 | +1 | 0.973 | +4.28 / +5.79 | -0.75 / +1.16 | -1.13 / +0.94 | -0.98 / +1.06 |
| XLK/XLI | tech x industrials (both equity) | +0.723 | +1 | 0.973 | +4.36 / +5.44 | -0.44 / +1.05 | -0.60 / +0.65 | -0.44 / +0.57 |
| XLF/XLI | financials x industrials (both equity) | +0.857 | +1 | 0.995 | +0.83 / +1.73 | -0.62 / +0.14 | +0.00 / +0.00 | -0.05 / +0.16 |
| XLE/XLB | energy x materials (both cyclical) | +0.638 | +1 | 0.955 | +8.86 / +8.86 | -0.39 / +0.18 | -1.02 / +0.10 | -1.26 / +0.06 |

## Rejection counts

| setting / score | raw (max E>=20) | family-wise (max E>=280) | median max log10E |
|---|---|---|---|
| contemp_addjoint | 14/14 | 13/14 | +15.00 |
| contemp_four | 10/14 | 10/14 | +8.25 |
| lagged_addjoint | 1/14 | 0/14 | +0.05 |
| lagged_four | 3/14 | 0/14 | +0.42 |

## Explanation numbers

- Spearman(cross R^2, contemp addjoint max log10E) = -0.59
- Spearman(corr, contemp addjoint max log10E) = -0.70
- Spearman(cross R^2, contemp four max log10E) = -0.39
- Spearman(corr, contemp four max log10E) = -0.86
- cross-term R^2 range [0.026, 0.995]; corr range [-0.262, +0.857]; 10/14 pairs with positive covariance term.
