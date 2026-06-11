import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
core = coredb("prod")
panel = query(core, """
SELECT person_id, months_since_zero_date AS ms, enrolled_days,
       (emergency_department_ct+acute_inpatient_ct) AS acute
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -27 AND 12""")
attrs = query(core, """
SELECT person_id, MAX(diabetes) diabetes,MAX(htn) htn,MAX(chf) chf,MAX(copd) copd,MAX(sud) sud,
 MAX(any_bh) any_bh,MAX(mdd) mdd,MAX(asthma) asthma,MAX(postpartum) postpartum,MAX(prenatal) prenatal,
 MAX(polypharmacy) polypharmacy,MAX(high_ed_ip) high_ed_ip,MAX(risk_percentile) risk,MAX(tier1_flg) tier1,
 MAX(age) age,MAX(gender) gender,MAX(race) race,MAX(state) state,
 MIN(first_eligible_date::date) elig,MAX(zero_date) zero_date
FROM dbt.outcomes_with_enrollment__months_since WHERE ever_activated=1 GROUP BY person_id""")
attrs["elig"]=pd.to_datetime(attrs.elig,errors="coerce");attrs["zero_date"]=pd.to_datetime(attrs.zero_date,errors="coerce")
attrs["e2a"]=((attrs.zero_date-attrs.elig).dt.days/30.44)
def mat(c):
    m=panel.pivot_table(index="person_id",columns="ms",values=c,aggfunc="sum",fill_value=0)
    return m.reindex(columns=range(-27,13),fill_value=0)
A=mat("acute");D=mat("enrolled_days");cols=list(range(-27,13));idx=A.index
def wsum(M,lo,hi):
    c=[x for x in cols if lo<=x<=hi];return M[c].sum(axis=1) if c else pd.Series(0.0,index=idx)
WASH=1   # exclude activation month +/- 1
PRE_LO,PRE_HI=-12,-7      # far-pre baseline (before trigger run-up)
rows=[]
for s in range(-12,1):
    treat=1 if s==0 else 0
    post_lo=s+1+WASH if treat else s+1
    post_hi=12 if treat else -1-WASH
    pre_lo,pre_hi=s+PRE_LO,s+PRE_HI
    if post_lo>post_hi: continue
    base=wsum(A,pre_lo,pre_hi);base_mm=wsum(D,pre_lo,pre_hi)/30.44
    post=wsum(A,post_lo,post_hi);post_mm=wsum(D,post_lo,post_hi)/30.44
    obs=(D[s]>0).values
    d=pd.DataFrame({"person_id":idx,"s":s,"treat":treat,
      "base":base.values,"base_mm":base_mm.values,"post":post.values,"post_mm":post_mm.values,"obs":obs})
    d=d[(d.obs)&(d.post_mm>0.3)&(d.base_mm>0.3)]   # require BOTH windows observed
    rows.append(d)
lm=pd.concat(rows,ignore_index=True).merge(attrs,on="person_id",how="left")
lm["mse"]=lm.e2a+lm.s; lm=lm[lm.mse>=0].copy()
lm["base_rate"]=1000*lm.base/lm.base_mm
for c in ["gender","race","state"]: lm[c]=lm[c].fillna("u")
lm["age"]=lm.age.fillna(lm.age.median());lm["risk"]=lm.risk.fillna(lm.risk.median());lm["tier1"]=lm.tier1.fillna(0)
print("DiD landmark units:",len(lm),"treated:",int(lm.treat.sum()),"control:",int((lm.treat==0).sum()))

condflags=["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","postpartum","prenatal"]
def did(d):
    d=d.copy()
    # PS on far-pre baseline (NOT trigger window) + risk + conditions + demo
    Xn=d[["base_rate","mse","risk","tier1","age"]+condflags].astype(float)
    Xc=pd.get_dummies(d[["gender","race","state"]],drop_first=True).astype(float)
    Z=pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1)
    Z=((Z-Z.mean())/Z.std(ddof=0).replace(0,1)).fillna(0)
    ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,d.treat).predict_proba(Z)[:,1],1e-3,1-1e-3)
    d["w"]=np.where(d.treat==1,1-ps,ps)
    pre=d[["person_id","treat","w","base","base_mm"]].rename(columns={"base":"y","base_mm":"mm"});pre["post"]=0
    po =d[["person_id","treat","w","post","post_mm"]].rename(columns={"post":"y","post_mm":"mm"});po["post"]=1
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,
              freq_weights=s.w).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"]
    return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)
out=[]
for name,c in [("ALL",None)]+[(c,c) for c in condflags]:
    d=lm if c is None else lm[lm[c]==1]
    if d.treat.sum()<40: out.append([name,int(d.treat.sum()),np.nan,np.nan,np.nan]);continue
    r,lo,hi=did(d); out.append([name,int(d.treat.sum()),round(r,3),round(lo,3),round(hi,3)])
res=pd.DataFrame(out,columns=["cohort","n_treat","DiD_RR","lo95","hi95"]).sort_values("DiD_RR")
res["verdict"]=np.where(res.hi95<1,"BENEFIT**",np.where(res.lo95>1,"harm",""))
print("\n### DiD-within-sequential-trial (pre-trend differenced out, activation month +/-1 washout)")
print("    DiD_RR<1 = earlier engagement reduces acute care, net of pre-trend.")
print(res.to_string(index=False))
