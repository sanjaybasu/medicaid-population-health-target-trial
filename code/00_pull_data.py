import os, pathlib
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd
core = coredb("prod")
print("pulling monthly panel...")
panel = query(core, """
SELECT person_id, months_since_zero_date AS ms, enrolled_days, zero_date,
  emergency_department_ct AS ed, acute_inpatient_ct AS ip,
  (COALESCE(ed_non_emergent,0)+COALESCE(ed_primary_care_treatable,0)+COALESCE(ed_preventable,0)) AS acsed,
  (COALESCE(pqi_01,0)+COALESCE(pqi_03,0)+COALESCE(pqi_05,0)+COALESCE(pqi_07,0)+COALESCE(pqi_08,0)
   +COALESCE(pqi_11,0)+COALESCE(pqi_12,0)+COALESCE(pqi_14,0)+COALESCE(pqi_15,0)+COALESCE(pqi_16,0)) AS pqi,
  pc_office_visit AS pcp, office_visit_ct AS office, telehealth_ct AS tele,
  (COALESCE(outpatient_psych_ct,0)+COALESCE(outpatient_substance_use_ct,0)) AS bh, pharmacy_ct AS rx
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -27 AND 12""")
panel["acute"]=panel.ed.fillna(0)+panel.ip.fillna(0)
panel.to_parquet(str(CACHE/"panel.parquet"))
print("  panel rows:",len(panel),"persons:",panel.person_id.nunique())
print("pulling attrs...")
attrs = query(core, """
SELECT person_id, MAX(rr_flag) rr, MAX(no_pcp_last_10mo) nopcp, MAX(market) market, MAX(postal_code) postal,
 MAX(postpartum) postpartum,MAX(prenatal) prenatal,MAX(diabetes) diabetes,MAX(htn) htn,MAX(chf) chf,MAX(copd) copd,
 MAX(sud) sud,MAX(any_bh) any_bh,MAX(mdd) mdd,MAX(asthma) asthma,MAX(polypharmacy) polypharmacy,MAX(high_ed_ip) high_ed_ip,
 MAX(risk_percentile) risk,MAX(tier1_flg) tier1,MAX(age) age,MAX(gender) gender,MAX(race) race,MAX(state) state,
 MIN(first_eligible_date::date) elig,MAX(zero_date) zero_date
FROM dbt.outcomes_with_enrollment__months_since WHERE ever_activated=1 GROUP BY person_id""")
attrs.to_parquet(str(CACHE/"attrs.parquet"))
print("  attrs persons:",len(attrs))
print("  rr_flag=1:",int((attrs.rr==1).sum()),"| nopcp=1:",int((attrs.nopcp==1).sum()),"| has postal:",int(attrs.postal.notna().sum()))
