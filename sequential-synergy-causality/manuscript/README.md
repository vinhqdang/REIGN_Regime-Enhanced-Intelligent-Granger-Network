# ANTE manuscript (revision) — Discover Applied Mathematics

Revised manuscript for Paper 2 (ANTE), prepared with the Springer Nature `sn-jnl`
LaTeX template (author-year `sn-mathphys-ay` references), together with the
point-by-point response to the editor and reviewers.

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | Revised manuscript (single self-contained source) |
| `main.pdf` | Compiled revised manuscript |
| `response_to_reviewers.tex` / `.pdf` | Point-by-point response letter |
| `sn-bibliography.bib`, `main.bbl` | Bibliography |
| `sn-jnl.cls`, `sn-mathphys-ay.bst`, `bst/` | Springer class and style files |
| `figures/` | Figures (`fig_*.png` are drawn from the saved results by `experiments/revision/make_paper_figures.py`) |
| `cover_letter.tex/.pdf` | Cover letter of the first submission (superseded) |

## Build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Where the numbers come from

Every number and figure in the manuscript is produced by a script in
`../experiments/revision/` (results in `../experiments/revision/results/`).
`../src/ante_v2.py` is the revised test (truncated-loss score with a predictable bound,
exclusion processes and the time-uniform average-synergy bound).

## Structure

Introduction and scope of claims → related work and positioning → setup (interventional and
predictive nulls) → Part A: interventional ANTE → Part B: predictive ANTE-SG (null as a
comparison of two forecasters, exactly valid truncated-loss score, isolation analysis, exclusion,
multiplicity, table of guarantees) → experiments (validity and stress tests, correlated
sources and nonlinear alternatives, benchmark, positive controls, a fully specified financial
search with placebo calibration) → discussion → declarations → appendices (proofs, bounds table).
