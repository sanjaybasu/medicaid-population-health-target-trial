# Medicaid Population Health Target-Trial Emulation — analysis code

Reproducible pipeline for "Effect of an Artificial Intelligence–Directed, Multidisciplinary Population Health
Program on Avoidable Acute Care in Medicaid: A Target-Trial Emulation."

**Single source of truth:** all reported numbers are produced by this pipeline and
written to `results.json`; tables and figures are rendered from that file. There are
no hardcoded results.

## Steps (see `run_all.sh`)
1. `build_zip_adi.py` — build the ZIP→ADI crosswalk (`cache/zip_adi.csv`) from two public
   files placed in `cache/`: the UW Neighborhood Atlas block-group ADI
   (`BlockGroupToADI.feather`) and the Census 2020 ZCTA↔tract relationship file
   (`zcta_tract.txt`); sources documented in the script header.
2. `00_pull_data.py` — pull the person-month panel + attributes from coredb (the only
   DB-dependent step) → `cache/panel.parquet`, `cache/attrs.parquet`.
3. `01_compute_results.py` — cohort (rising-risk `rr_flag=1`, claims mature through Dec 2025),
   event study, naive-by-window, overlap-weighted Poisson DiD (primary rate ratio) with
   trimmed-IPW and 1:5 matched triangulation, disaggregated/ACS outcomes, equity
   (race/sex/ADI), mechanism (primary-care by baseline connection), and Table 1 → `results.json`.
4. `02_cs_dr_honest.py` — Callaway–Sant'Anna doubly-robust additive ATT and Honest-DiD
   M̄ (requires the `differences` package).
5. `03_make_artifacts.py` — render Table 1, Table 2, and Figures 1–3 to `outputs/`.
6. `04_checks.py` — invariant guardrails (trigger peak at −1; RR<1; null pre-trends;
   ALL-pathway validation reproduces the locked estimate).
7. `05_secondary_analyses.py` — acute-care spending (CMS 99th-percentile truncation), effect
   by clinical condition (with within-stratum falsification), ED visits by NYU severity, and
   zero-event robustness (negative-binomial, two-part hurdle) → `results.json`.
8. `06_care_gaps.py` — care-gap taxonomy at rising-risk identification operationalized in
   claims, pharmacy fills, and diagnoses across cascade domains (access, guideline-concordant
   chronic-disease treatment, behavioral-health/SUD, medication safety, transitions, social
   needs); renders eTable 9 and underlies the conceptual model (eFigure 1) → `results.json`.
   DB-dependent (pulls `pharmacy_claim` and `condition`).
9. `07_social_needs.py` — documented health-related social needs (PRAPARE/AHC domains) by
   transparent keyword search of free-text "meet the patient" intake notes (Lighthouse), among
   the ~36% of the cohort with such a note; an explicit lower bound (eTable 9 social section).
   DB-dependent (Lighthouse); note text (PHI) is processed only in aggregate.

## Notes
- Output location is `$ARTIFACTS_DIR` (defaults to the repo root): `cache/`, `outputs/`, `results.json`.
- Individual-level Medicaid data cannot be shared (state data-use agreements, HIPAA);
  only code is published here.
- IRB: WCG Institutional Review Board (tracking ID 20253751), waiver of consent and HIPAA authorization.
