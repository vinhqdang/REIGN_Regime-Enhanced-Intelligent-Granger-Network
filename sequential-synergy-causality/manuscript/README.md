# ANTE manuscript — Statistics and Computing (Springer)

Full manuscript for Paper 2 (ANTE), prepared with the official Springer Nature
`sn-jnl` LaTeX template (author–year `sn-mathphys-ay` reference style, as required
by *Statistics and Computing*, which cites by name and year).

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | The manuscript (single self-contained source, per Springer policy) |
| `main.pdf` | Compiled output (21 pages) |
| `sn-bibliography.bib` | Bibliography (author–year, verified DOIs) |
| `main.bbl` | Generated bibliography (included for submission) |
| `sn-jnl.cls`, `sn-mathphys-ay.bst`, `bst/` | Springer class and style files |
| `figures/` | 300-dpi PNG figures |

## Build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Structure

Storytelling introduction → related work and the gap (three synergy literatures
vs. the anytime-valid backbone) → problem setup and the synergy null on two scales
→ the ANTE test (interaction score, betting e-process, **validity theorem**,
**Ville anytime-validity corollary**, **consistency/power theorem** with an
explicit detection-time bound, **k-way group proposition**, **interaction-isolation
proposition**) → algorithms → experiments (anytime-valid control, SOTA benchmark,
real portfolio-variance synergy, turbulence positive, and the evidence-of-absence
finance finding) → discussion → conclusion → declarations → proofs appendix.

Target journal: *Statistics and Computing* (Springer, Q1). Free to publish
(no submission or publication fee; open access optional).
