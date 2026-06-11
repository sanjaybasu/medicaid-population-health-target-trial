# Correcting enrollment-triggered bias in evaluations of Medicaid population health programs

Analysis code for a target-trial emulation that diagnoses, quantifies, and corrects
regression-to-the-mean bias arising when enrollment in high-risk care/population-health
programs is triggered by an acute event.

## Method
- **Bias diagnosis:** event study of acute care relative to program activation (the month −1 trigger spike).
- **Correction:** sequential target-trial emulation with Callaway–Sant'Anna staggered
  difference-in-differences (not-yet-treated controls), triangulated with overlap-weighted /
  IPTW / matched difference-in-differences.
- **Robustness:** parallel-pre-trends falsification, E-value, Honest difference-in-differences
  (Rambachan & Roth) relative-magnitude sensitivity.
- **Heterogeneity (secondary):** doubly robust CATE, rank-weighted ATE (AUTOC).

## Code
See `code/`. Scripts are ordered: cohort build (`build_cspanel.py`), primary CS-DiD
(`cs_run2.py`), triangulation/falsification (`did.py`, `confirm.py`), Honest-DiD (`honest.py`),
heterogeneity (`hte.py`), bias quantification (`naive_vs_corrected.py`, `prewindow.py`),
tables and figures (`tables.py`, `fix_figures.py`).

## Reproducibility
Python 3; see `requirements.txt`. The Callaway–Sant'Anna estimator uses the `differences` package.
Scripts read from a project data warehouse through an internal Vault-backed connector
(`wm_conn`) that is **not** included; table/column references are specific to that warehouse.

## Data availability
Individual-level Medicaid data cannot be shared under state data-use agreements and HIPAA.
This repository contains **code only** — no individual-level data, no results, no manuscript.

## Citation
[Add on publication.]
