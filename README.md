# Correcting enrollment-triggered bias in evaluations of Medicaid population health programs

Analysis code for a target-trial emulation that diagnoses, quantifies, and corrects
regression-to-the-mean bias arising when enrollment in high-risk care/population-health
programs is triggered by an acute event.

## Method
- **Bias diagnosis:** event study of acute care relative to program activation (the month −1 trigger spike).
- **Correction:** sequential target-trial emulation with Callaway–Sant'Anna staggered
  difference-in-differences (not-yet-treated controls), triangulated with overlap-weighted /
  IPTW / matched difference-in-differences. Because the comparison group comprises not-yet-activated
  patients who themselves activate later, the estimand is the effect of earlier versus later activation
  (a timing contrast holding activation propensity fixed).
- **Robustness:** parallel-pre-trends falsification, E-value, Honest difference-in-differences
  (Rambachan & Roth) relative-magnitude sensitivity curve, ATT-vs-overlap (ATO) estimand concordance,
  negative-binomial / two-part specifications, and baseline-rate-by-subgroup context.
- **Heterogeneity (secondary):** doubly robust CATE, rank-weighted ATE (AUTOC).

## Code
See `code/` (run via `code/run_all.sh`), ordered:
- `00_pull_data.py` — cohort/panel pull (only DB-dependent step) → `cache/`
- `01_compute_results.py` — overlap-weighted / IPTW / matched DiD, by-outcome, equity, mechanism, baseline table
- `02_cs_dr_honest.py` — Callaway–Sant'Anna doubly robust group-time ATT + Honest-DiD inputs
- `03_make_artifacts.py` — display items
- `04_checks.py` — invariant guardrails
- `05_secondary_analyses.py` — cost, effect by condition, ED severity, negative-binomial / hurdle
- `06_care_gaps.py`, `07_social_needs.py` — care-gap and documented-social-need characterization
- `08_revision_robustness.py` — IPW full-precision CI, ATT-vs-ATO concordance, Honest-DiD relative-magnitude curve, baseline rates by subgroup, cohort funnel

## Reproducibility
Python 3; see `requirements.txt`. The Callaway–Sant'Anna estimator uses the `differences` package.
Scripts read from a project data warehouse through an internal Vault-backed connector
(`wm_conn`) that is **not** included; table/column references are specific to that warehouse.

## Data availability
Individual-level Medicaid data cannot be shared under state data-use agreements and HIPAA.
This repository contains **code only** — no individual-level data, no results, no manuscript.

## Citation
[Add on publication.]
