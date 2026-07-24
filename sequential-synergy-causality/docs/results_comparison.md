# Results: Is ANTE the Best? (honest, per-axis)

**Short answer: not on every metric — SURD/PEID edge us on raw detection accuracy
(1.00 vs 0.98) and fixed-N power — but ANTE is the only method that satisfies the
full capability set the problem requires, and it matches SOTA on detection.**

## Per-axis results

| Axis / metric | ANTE (ours) | SURD 2024 | PEID 2026 | Jha DGC 2026 | Interaction-Info | Winner |
|---|---|---|---|---|---|---|
| Canonical XOR/AND | ✓ 1.00 | ✓ 1.00 | ✓ 1.00 | ✗ misses | partial | tie |
| Detection AUC (syn vs non), synthetic | 0.98 | **1.00** | **1.00** | — | 0.87 | SURD/PEID |
| Fixed-N power @ N=250 | low | **0.99** | — | — | 0.20 | **SURD** |
| Discriminate own-nonlinearity (A²) | **✓** | partial | partial | — | ✗ | **ANTE** |
| Isolate super-additive interaction (≈0 on A+B) | **✓** | ✗ | ✗ | — | ✗ | **ANTE** |
| Type-I under continuous monitoring | **0.016** | 0.14 | 0.14 | ✓(1-src) | 0.14 | **ANTE** |
| Detect **group** synergy, sequential | **1.00** | batch only | batch only | **0.01** | — | **ANTE** |
| Real portfolio-variance separation AUC | **1.00** | 0.85 | 0.90 | — | 1.00 | ANTE (tie II) |
| Real turbulence 3-way group causality | **✓ log₁₀E 4.0** | ✓ agrees | — | — | — | tie ANTE/SURD |

## Capability matrix (see `capability_matrix.png`)

| Capability | ANTE | SURD | PEID | Jha DGC | O-info/PID/II |
|---|---|---|---|---|---|
| Detects group synergy | ✓ | ✓ | ✓ | ✗ | ✓ |
| Sequential (streaming) | ✓ | ✗ | ✗ | ✓ | ✗ |
| Anytime-valid error control | ✓ | ✗ | ✗ | ✓ | ✗ |
| Isolates interaction (not joint-dependence) | ✓ | ✗ | ✗ | ✗ | ✗ |
| k-way groups (>2) | ✓ | ✓ | ✓ | ✗ | ✓ |
| No stationarity / model-spec needed | ✓ | ✗ | ✗ | ~ | ✗ |
| **All of the above** | **✓ (only one)** | ✗ | ✗ | ✗ | ✗ |

## Verdict
- **Raw detection:** SURD/PEID ≥ ANTE (0.98 vs 1.00) — we are competitive, not #1.
- **Fixed-N power:** batch SURD > ANTE — the standard cost of anytime-validity.
- **Everything the task actually needs** — sequential monitoring with error
  control, on *groups*, isolating genuine interaction, without stationarity
  assumptions — **ANTE is the only method that does it.** It is the best tool for
  sequential group-causality; it does not claim to be a better point estimator.
