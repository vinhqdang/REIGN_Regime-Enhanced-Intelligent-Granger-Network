# R1 Type-I stress tests (auto-generated)

n_rep = 200 per null (T=2500), alpha = 0.05, rejection = max E >= 20.

## (a) Toy counterexample (legacy-style clipped bettor)

| sigma_hat | n_rep | T | P(max E>=20) | 95% Wilson CI | median log10 final E | median log10 max E | E[s] | E[u] |
|---|---|---|---|---|---|---|---|---|
| adaptive (expanding rms) | 2000 | 2000 | 0.0040 | [0.0020, 0.0079] | -0.044 | 0.002 | -0.65 | n/a |
| fixed = 1 (reviewer example) | 2000 | 2000 | 0.9775 | [0.9700, 0.9831] | -2.549 | 8.873 | -0.65 | 0.85 |

Clipping at -5 with sigma=1 makes E[u] > 0 although E[s] < 0, so the fixed-sigma bettor is a genuine Type-I failure of the clipped score; the adaptive version normalises the -20 outliers and does not grow. The unclipped bounded alternative is not applicable to this toy score.

## (b) VAR nulls

| id | DGP | test | rejection | 95% Wilson CI | median log10 final E | median log10 max E | mean lambda>0 | mean raw s |
|---|---|---|---|---|---|---|---|---|
| N1 | gaussian additive rho=.2 | addjoint | 0.000 (0/200) | [0.000, 0.019] | -0.176 | 0.024 | 0.272 | -0.0066 |
| N1 | gaussian additive rho=.2 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.007 | -0.0971 |
| N1 | gaussian additive rho=.2 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.1426 |
| N2 | heavy t3 + skew additive rho=.2 | addjoint | 0.000 (0/200) | [0.000, 0.019] | -0.172 | 0.025 | 0.237 | -0.0074 |
| N2 | heavy t3 + skew additive rho=.2 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.005 | -0.1021 |
| N2 | heavy t3 + skew additive rho=.2 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.1117 |
| N3 | heavy + garch additive rho=.2 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.128 | 0.016 | 0.227 | -0.0071 |
| N3 | heavy + garch additive rho=.2 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.004 | -0.1051 |
| N3 | heavy + garch additive rho=.2 | legacy | 0.000 (0/200) | [0.000, 0.019] | -0.000 | 0.000 | n/a | 0.1051 |
| N4 | additive rho=.95 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.192 | 0.029 | 0.256 | -0.0044 |
| N4 | additive rho=.95 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.000 | -0.8500 |
| N4 | additive rho=.95 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -1.0090 |
| N5 | additive rho=.99 | addjoint | 0.010 (2/200) | [0.003, 0.036] | -0.269 | 0.038 | 0.272 | -0.0030 |
| N5 | additive rho=.99 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.000 | -0.9060 |
| N5 | additive rho=.99 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -1.0586 |
| N6 | cubic additive rho=.5 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.178 | 0.031 | 0.270 | -0.0079 |
| N6 | cubic additive rho=.5 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.007 | -0.2538 |
| N6 | cubic additive rho=.5 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.5535 |
| N7 | cubic rho=.9 | addjoint | 0.000 (0/200) | [0.000, 0.019] | -0.212 | 0.035 | 0.247 | -0.0061 |
| N7 | cubic rho=.9 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.001 | -0.6347 |
| N7 | cubic rho=.9 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -3.7302 |
| N8 | break (sign flip at T/2) rho=.2 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.149 | 0.021 | 0.233 | -0.0065 |
| N8 | break (sign flip at T/2) rho=.2 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.006 | -0.0810 |
| N8 | break (sign flip at T/2) rho=.2 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.1110 |
| N9 | own nonlinearity rho=.5 | addjoint | 0.000 (0/200) | [0.000, 0.019] | -0.172 | 0.018 | 0.303 | -0.0060 |
| N9 | own nonlinearity rho=.5 | four | 0.000 (0/200) | [0.000, 0.019] | -0.055 | 0.000 | 0.034 | -0.0490 |
| N9 | own nonlinearity rho=.5 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.2547 |
| N10 | additive ar=0.9 sources rho=.5 | addjoint | 0.000 (0/200) | [0.000, 0.019] | -0.125 | 0.015 | 0.197 | -0.0078 |
| N10 | additive ar=0.9 sources rho=.5 | four | 0.170 (34/200) | [0.124, 0.228] | -0.035 | 0.091 | 0.252 | -0.0035 |
| N10 | additive ar=0.9 sources rho=.5 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.1202 |
| N11 | additive forget=0.98 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.123 | 0.019 | 0.158 | -0.1476 |
| N11 | additive forget=0.98 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.008 | -0.0358 |
| N11 | additive forget=0.98 | legacy | 0.000 (0/200) | [0.000, 0.019] | -0.000 | 0.000 | n/a | -9197597.1787 |
| N12 | additive forget=0.999 | addjoint | 0.005 (1/200) | [0.001, 0.028] | -0.200 | 0.030 | 0.278 | -0.0049 |
| N12 | additive forget=0.999 | four | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | 0.009 | -0.0946 |
| N12 | additive forget=0.999 | legacy | 0.000 (0/200) | [0.000, 0.019] | 0.000 | 0.000 | n/a | -0.1476 |

Runtime: 16.3 min. Figure: `/home/user/REIGN_Regime-Enhanced-Intelligent-Granger-Network/sequential-synergy-causality/experiments/revision/results/r1_stress_typeI.png`
