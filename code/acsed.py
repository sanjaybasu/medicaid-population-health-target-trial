import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
core=coredb("prod")
panel=query(core,"""SELECT person_id, months_since_zero_date ms, enrolled_days,
  (COALESCE(ed_non_emergent,0)+COALESCE(ed_primary_care_treatable,0)+COALESCE(ed_preventable,0)) acsed
 FROM dbt.outcomes_with_enrollment__months_since
 WHERE ever_activated=1 AND months_since_zero_date BETWEEN -27 AND 12""")
attrs=query(core,"""SELECT person_id,MAX(diabetes) diabetes,MAX(htn) htn,MAX(chf) chf,MAX(copd) copd,MAX(sud) sud,
 MAX(any_bh) any_bh,MAX(risk_percentile) risk,MAX(age) age,MAX(gender) gender,MAX(race) race,MAX(state) state,
 MIN(first_eligible_date::date) elig,MAX(zero_date) zero_date
 FROM dbt.outcomes_with_enrollment__months_since WHERE ever_activated=1 GROUP BY person_id""")
attrs["elig"]=pd.to_datetime(attrs.elig,errors="coerce");attrs["zero_date"]=pd.to_datetime(attrs.zero_date,errors="coerce")
attrs["e2a"]=((attrs.zero_date-attrs.elig).dt.days/30.44)
def mat(c):
    m=panel.pivot_table(index="person_id",columns="ms",values=c,aggfunc="sum",fill_value=0)
    return m.reindex(columns=range(-27,13),fill_value=0)
A=mat("acsed");D=mat("enrolled_days");cols=list(range(-27,13));idx=A.index
def wsum(M,lo,hi):
    c=[x for x in cols if lo<=x<=hi];return M[c].sum(axis=1) if c else pd.Series(0.0,index=idx)
WASH=1; rows=[]
for s in range(-12,1):
    treat=1 if s==0 else 0; plo=s+1+WASH if treat else s+1; phi=12 if treat else -1-WASH
    if plo>phi: continue
    d=pd.DataFrame({"person_id":idx,"s":s,"treat":treat,"b":wsum(A,s-12,s-7).values,
      "mm_b":(wsum(D,s-12,s-7)/30.44).values,"p":wsum(A,plo,phi).values,"mm_p":(wsum(D,plo,phi)/30.44).values,
      "obs":(D[s]>0).values})
    rows.append(d[(d.obs)&(d.mm_p>0.3)&(d.mm_b>0.3)])
lm=pd.concat(rows,ignore_index=True).merge(attrs,on="person_id",how="left")
lm["mse"]=lm.e2a+lm.s; lm=lm[lm.mse>=0].copy()
for c in ["gender","race","state"]: lm[c]=lm[c].fillna("u")
lm["age"]=lm.age.fillna(lm.age.median());lm["risk"]=lm.risk.fillna(lm.risk.median())
cf=["diabetes","htn","chf","copd","sud","any_bh"]; lm["base_rate"]=1000*lm.b/lm.mm_b.clip(lower=0.1)
Xn=lm[["base_rate","mse","risk","age"]+cf].astype(float)
Xc=pd.get_dummies(lm[["gender","race","state"]],drop_first=True).astype(float)
Z=pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1); Z=((Z-Z.mean())/Z.std(ddof=0).replace(0,1)).fillna(0)
ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,lm.treat).predict_proba(Z)[:,1],1e-3,1-1e-3)
w=np.where(lm.treat==1,1-ps,ps)
pre=lm[["person_id","treat"]].copy();pre["y"]=lm.b;pre["mm"]=lm.mm_b;pre["post"]=0;pre["w"]=w
po =lm[["person_id","treat"]].copy();po["y"]=lm.p;po["mm"]=lm.mm_p;po["post"]=1;po["w"]=w
s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
b=m.params["treat:post"];se=m.bse["treat:post"]; t=lm[lm.treat==1]
print(f"### NYU ambulatory-care-sensitive ED (non-emergent + primary-care-treatable + preventable), n_treated={int(lm.treat.sum())}")
print(f"  corrected RR = {np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f}) [{100*(1-np.exp(b)):.0f}% reduction]")
print(f"  treated post ACS-ED rate = {1000*t.p.sum()/t.mm_p.sum():.1f}/1000mm")
