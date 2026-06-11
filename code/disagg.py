import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
core=coredb("prod")
panel=query(core,"""SELECT person_id, months_since_zero_date ms, enrolled_days,
   emergency_department_ct ed, acute_inpatient_ct ip
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
ED=mat("ed");IP=mat("ip");D=mat("enrolled_days");cols=list(range(-27,13));idx=ED.index
def wsum(M,lo,hi):
    c=[x for x in cols if lo<=x<=hi];return M[c].sum(axis=1) if c else pd.Series(0.0,index=idx)
WASH=1
rows=[]
for s in range(-12,1):
    treat=1 if s==0 else 0
    plo=s+1+WASH if treat else s+1; phi=12 if treat else -1-WASH
    if plo>phi: continue
    d=pd.DataFrame({"person_id":idx,"s":s,"treat":treat,
      "ed_b":wsum(ED,s-12,s-7).values,"ip_b":wsum(IP,s-12,s-7).values,"mm_b":(wsum(D,s-12,s-7)/30.44).values,
      "ed_p":wsum(ED,plo,phi).values,"ip_p":wsum(IP,plo,phi).values,"mm_p":(wsum(D,plo,phi)/30.44).values,
      "obs":(D[s]>0).values})
    rows.append(d[(d.obs)&(d.mm_p>0.3)&(d.mm_b>0.3)])
lm=pd.concat(rows,ignore_index=True).merge(attrs,on="person_id",how="left")
lm["mse"]=lm.e2a+lm.s; lm=lm[lm.mse>=0].copy()
lm[" edrate"]=1000*lm.ed_b/lm.mm_b  # for PS
for c in ["gender","race","state"]: lm[c]=lm[c].fillna("u")
lm["age"]=lm.age.fillna(lm.age.median());lm["risk"]=lm.risk.fillna(lm.risk.median())
cf=["diabetes","htn","chf","copd","sud","any_bh"]
def didrr(lm,bcol,pcol):
    d=lm.copy()
    d["base_rate"]=1000*d[bcol]/d.mm_b.clip(lower=0.1)
    Xn=d[["base_rate","mse","risk","age"]+cf].astype(float)
    Xc=pd.get_dummies(d[["gender","race","state"]],drop_first=True).astype(float)
    Z=pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1)
    Z=((Z-Z.mean())/Z.std(ddof=0).replace(0,1)).fillna(0)
    ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,d.treat).predict_proba(Z)[:,1],1e-3,1-1e-3)
    w=np.where(d.treat==1,1-ps,ps)
    pre=d[["person_id","treat"]].copy();pre["y"]=d[bcol];pre["mm"]=d.mm_b;pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d[pcol];po["mm"]=d.mm_p;po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w
              ).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"];return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)
print("### Disaggregated corrected effect (overlap-weighted DiD rate ratio), n_treated=%d"%int(lm.treat.sum()))
for nm,bc,pc in [("ED + hospitalization (combined)","ed_b","ed_p")]:
    pass
# combined for check
lm["ac_b"]=lm.ed_b+lm.ip_b; lm["ac_p"]=lm.ed_p+lm.ip_p
for nm,bc,pc in [("ED visits","ed_b","ed_p"),("Acute inpatient admissions","ip_b","ip_p"),("Combined (check)","ac_b","ac_p")]:
    rr,lo,hi=didrr(lm,bc,pc)
    # descriptive post rate by arm
    t=lm[lm.treat==1];c=lm[lm.treat==0]
    tpost=1000*t[pc].sum()/t.mm_p.sum(); cpost=1000*c[pc].sum()/c.mm_p.sum()
    print(f"  {nm:30s} RR={rr:.3f} ({lo:.3f}-{hi:.3f})  [{100*(1-rr):.0f}% reduction]  treated post={tpost:.0f}/1000mm")
