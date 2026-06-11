import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
core = coredb("prod")
# calendar panel for ever-activated; event window -15..+12; enrolled months only
panel = query(core, """
SELECT person_id, zero_date, months_since_zero_date AS ms, enrolled_days,
       (emergency_department_ct+acute_inpatient_ct) AS acute
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -15 AND 12 AND enrolled_days>0""")
attrs = query(core, """
SELECT person_id,MAX(postpartum) postpartum,MAX(prenatal) prenatal,MAX(diabetes) diabetes,MAX(htn) htn,
 MAX(chf) chf,MAX(copd) copd,MAX(sud) sud,MAX(any_bh) any_bh,MAX(mdd) mdd,MAX(asthma) asthma,
 MAX(polypharmacy) polypharmacy,MAX(high_ed_ip) high_ed_ip,MAX(risk_percentile) risk,
 MAX(age) age,MAX(gender) gender,MAX(race) race,MAX(state) state
FROM dbt.outcomes_with_enrollment__months_since WHERE ever_activated=1 GROUP BY person_id""")
panel["zero_date"]=pd.to_datetime(panel.zero_date,errors="coerce")
panel=panel.dropna(subset=["zero_date"])
# calendar month index (months since 2023-01)
zi = (panel.zero_date.dt.year-2023)*12 + (panel.zero_date.dt.month-1)
panel["cohort"]=zi.astype(int)              # activation calendar month = first-treated period
panel["time"]=(panel["cohort"]+panel["ms"]).astype(int)
panel=panel.merge(attrs,on="person_id",how="left")
panel["perinatal"]=((panel.postpartum==1)|(panel.prenatal==1)).astype(int)
# rate per 1000 member-months as outcome (acute scaled by enrolled exposure)
panel["acute_p1k"]=1000*panel.acute/(panel.enrolled_days/30.44)
for c in ["gender","race","state"]: panel[c]=panel[c].fillna("u")
panel["age"]=panel.age.fillna(panel.age.median()); panel["risk"]=panel.risk.fillna(panel.risk.median())
panel.to_parquet("/tmp/cs_panel.parquet")
print("rows:",len(panel),"persons:",panel.person_id.nunique(),
      "cohorts:",panel.cohort.nunique(),"time range:",panel.time.min(),panel.time.max())
print("cohort sizes (activation month -> n persons):")
print(panel.groupby("cohort").person_id.nunique().describe()[["min","50%","max"]].round(0).to_string())
