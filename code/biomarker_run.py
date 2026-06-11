import warnings; warnings.filterwarnings("ignore")
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
core = coredb("prod")

# activation dates + covariates for ever-activated and targeted-not-activated (VA lab pop)
pp = query(core, """
SELECT person_id, MAX(ever_activated) act, MAX(zero_date) adate,
   MAX(risk_percentile) risk, MAX(age) age, MAX(diabetes) diabetes, MAX(htn) htn
FROM dbt.outcomes_with_enrollment__months_since
WHERE (ever_activated=1 OR ever_targeted=1) GROUP BY person_id""")
pp["adate"]=pd.to_datetime(pp.adate,errors="coerce")

labs = query(core, """
SELECT person_id, result_date, result, source_description
FROM dbt_tuva_core.lab_result
WHERE data_source='abhva_lab' AND (
  source_description ILIKE '%ldl%' OR source_description ILIKE '%egfr%'
  OR source_description ILIKE '%a1c%' OR source_description ILIKE '%hemoglobin a%')""")
labs["result_date"]=pd.to_datetime(labs.result_date,errors="coerce")
def kind(s):
    s=s.lower()
    if "egfr" in s: return "eGFR"
    if "ldl" in s: return "LDL"
    return "A1c"
labs["bm"]=labs.source_description.map(kind)
def parseval(x):
    if x is None: return np.nan
    m=re.search(r'-?\d+\.?\d*', str(x))
    return float(m.group()) if m else np.nan
labs["val"]=labs.result.map(parseval)
# plausibility filters
labs=labs.dropna(subset=["val","result_date"])
ok=((labs.bm=="LDL")&labs.val.between(20,400))|((labs.bm=="A1c")&labs.val.between(3,18))|((labs.bm=="eGFR")&labs.val.between(5,150))
labs=labs[ok]

d=labs.merge(pp,on="person_id",how="inner")
d["days_rel"]=(d.result_date-d.adate).dt.days     # NaT for controls -> NaN
# time-varying post (activated only), washout: drop peri-activation [-15,+45]
d["post"]=np.where((d.act==1)&(d.days_rel>=45),1,0)
peri=(d.act==1)&(d.days_rel.between(-15,45))
d=d[~peri]
d["cal"]=((d.result_date.dt.year-2023)*12+(d.result_date.dt.month-1))/12.0
d["age"]=d.age.fillna(d.age.median()); d["risk"]=d.risk.fillna(d.risk.median())

print("### CKM biomarker mixed-effects models: activation (time-varying) vs not, VA lab population")
print("    'post' coef = within-person change after activation, net of secular trend + person RE.")
for bm,better in [("LDL","lower"),("A1c","lower"),("eGFR","higher")]:
    s=d[d.bm==bm].copy()
    npost=s[s.post==1].person_id.nunique()
    try:
        m=smf.mixedlm("val ~ post + cal + age + risk", s, groups=s.person_id).fit(method="lbfgs")
        b=m.params["post"]; lo,hi=m.conf_int().loc["post"]
        direction="IMPROVED" if ((better=="lower" and b<0) or (better=="higher" and b>0)) else "worse"
        sig="*" if (lo>0 or hi<0) else ""
        print(f"\n  {bm}: post coef = {b:+.2f} ({lo:+.2f},{hi:+.2f}) {sig}  [{direction} if sig; better={better}]")
        print(f"       n_obs={len(s)}, n_persons={s.person_id.nunique()}, n_with_post_measures={npost}")
    except Exception as e:
        print(f"\n  {bm}: ERR {str(e)[:100]}")
