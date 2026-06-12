import os, pathlib
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json
panel=pd.read_parquet(str(CACHE/"panel.parquet")); attrs=pd.read_parquet(str(CACHE/"attrs.parquet"))
for c in ["diabetes","htn","chf","copd","sud","any_bh","rr"]: attrs[c]=pd.to_numeric(attrs[c],errors="coerce").fillna(0)
attrs["risk"]=pd.to_numeric(attrs.risk,errors="coerce"); attrs["age"]=pd.to_numeric(attrs.age,errors="coerce")
panel["zd"]=pd.to_datetime(panel.zero_date,errors="coerce")
panel["zmon"]=panel.zd.dt.to_period("M").apply(lambda p:p.ordinal if pd.notna(p) else np.nan)
panel["calmon"]=panel.zmon+panel.ms
CUT=pd.Period("2025-12","M").ordinal
keep=set(attrs[attrs.rr==1].person_id)
p=panel[(panel.calmon<=CUT)&(panel.person_id.isin(keep))].copy()
p["acute_p1k"]=1000*p.acute/(p.enrolled_days.clip(lower=0.1)/30.44)
p=p[p.enrolled_days>0]
p["cohort"]=p.zmon.astype(int); p["time"]=(p.zmon+p.ms).astype(int)
a=attrs[["person_id","risk","age","diabetes","htn","chf","copd","sud","any_bh"]].copy()
a["risk"]=a.risk.fillna(a.risk.median()); a["age"]=a.age.fillna(a.age.median())
df=p[["person_id","time","cohort","acute_p1k","ms"]].merge(a,on="person_id",how="left")
df=df.drop_duplicates(["person_id","time"]).set_index(["person_id","time"]).sort_index()
print("CS panel rows:",len(df),"persons:",df.reset_index().person_id.nunique())
from differences import ATTgt
att=ATTgt(data=df, cohort_column="cohort", base_period="universal", anticipation=1)
att.fit(formula="acute_p1k ~ risk + age + diabetes + htn + chf + copd + sud + any_bh",
        control_group="not_yet_treated", est_method="dr", n_jobs=4, progress_bar=False)
es=att.aggregate("event")
es.columns=['_'.join([str(x) for x in c if str(x)!='']).strip('_') for c in es.columns.values]
es=es.reset_index().rename(columns={es.reset_index().columns[0]:"e"})
attcol=[c for c in es.columns if c.endswith("ATT") or c=="ATT"][0]
secol=[c for c in es.columns if "std_error" in c][0]
es["e"]=pd.to_numeric(es["e"],errors="coerce"); es=es.dropna(subset=["e"]); es=es[es[secol]<1e6]
pre=es[(es.e<=-2)&(es.e>=-12)]; post=es[(es.e>=1)&(es.e<=12)]
eff=post[attcol].mean(); eff_se=np.sqrt((post[secol]**2).mean()/len(post)); eff_lo=abs(eff)-1.96*eff_se
maxpre=pre[attcol].abs().max(); mbar=abs(eff)/maxpre; mbar_lo=eff_lo/maxpre
R=json.load(open(str(ROOT/"results.json")))
R["cs_dr"]={"post_ATT_per1000mm":round(float(eff),1),"ci_halfwidth":round(float(1.96*eff_se),1),
            "trigger_e-1_ATT":round(float(es.loc[es.e==-1,attcol].values[0]),1) if (es.e==-1).any() else None,
            "max_pre_abs":round(float(maxpre),1),"honest_mbar":round(float(mbar),2),"honest_mbar_ci_lo":round(float(mbar_lo),2)}
json.dump(R,open(str(ROOT/"results.json"),"w"),indent=1,default=str)
print("CS-DR post ATT/1000mm:",R["cs_dr"]["post_ATT_per1000mm"],"+/-",R["cs_dr"]["ci_halfwidth"])
print("trigger e=-1 ATT:",R["cs_dr"]["trigger_e-1_ATT"])
print("Honest-DiD M-bar:",R["cs_dr"]["honest_mbar"],"(CI-lo",R["cs_dr"]["honest_mbar_ci_lo"],")")
