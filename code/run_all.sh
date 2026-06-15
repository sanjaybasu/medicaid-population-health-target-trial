#!/usr/bin/env bash
# Reproduce the analysis end-to-end. Requires project coredb access for 00_pull_data.py
# (the only DB-dependent step); all later steps run from the cached parquet in cache/.
# ADI crosswalk: run build_zip_adi.py first (needs the two public files noted in its header) -> cache/zip_adi.csv
set -euo pipefail
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
python 00_pull_data.py          # DB pull -> cache/panel.parquet, cache/attrs.parquet
python 01_compute_results.py    # -> results.json (overlap/IPTW/matched, outcomes, equity, mechanism, Table 1)
python 02_cs_dr_honest.py       # CS-DR additive ATT + Honest-DiD M-bar (differences venv)
python 03_make_artifacts.py     # -> outputs/ (Table 1, Table 2, Figures 1-3)
python 04_checks.py             # invariant guardrails
python 05_secondary_analyses.py # cost (per-patient-per-month; none/95th/99th-pct truncation), effect by condition, ED severity, NB/hurdle -> results.json
python 06_care_gaps.py          # care-gap taxonomy at identification (eTable 9; eFigure 1) -> results.json
python 07_social_needs.py       # documented social needs from intake notes (eTable 9 social section) -> results.json
python 08_revision_robustness.py # IPW full-precision CI, ATT/ATO concordance, Honest-DiD RM curve, subgroup baselines, funnel -> results.json
