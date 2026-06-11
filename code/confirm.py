import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
core = coredb("prod")
panel = query(core, """
SELECT person_id, months_since_zero_date AS ms, enrolled_days,
       (emergency_department_ct+acute_inpatient_ct) AS acute
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -27 AND 12""")
attrs = query(core, """
SELECT person_id,MAX(postpartum) postpartum,MAX(prenatal) prenatal,MAX(diabetes) diabetes,MAX(htn) htn,
 MAX(chf) chf,MAX(copd) copd,MAX(sud) sud,MAX(any_bh) any_bh,MAX(mdd) mdd,MAX(asthma) asthma,
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
WASH=1
rows=[]
for s in range(-12,1):
    treat=1 if s==0 else 0
    post_lo=s+1+WASH if treat else s+1; post_hi=12 if treat else -1-WASH
    if post_lo>post_hi: continue
    base=wsum(A,s-12,s-7);base_mm=wsum(D,s-12,s-7)/30.44
    prepre=wsum(A,s-18,s-13);prepre_mm=wsum(D,s-18,s-13)/30.44
    post=wsum(A,post_lo,post_hi);post_mm=wsum(D,post_lo,post_hi)/30.44
    d=pd.DataFrame({"person_id":idx,"s":s,"treat":treat,"base":base.values,"base_mm":base_mm.values,
      "prepre":prepre.values,"prepre_mm":prepre_mm.values,"post":post.values,"post_mm":post_mm.values,
      "obs":(D[s]>0).values})
    rows.append(d[(d.obs)&(d.post_mm>0.3)&(d.base_mm>0.3)])
lm=pd.concat(rows,ignore_index=True).merge(attrs,on="person_id",how="left")
lm["mse"]=lm.e2a+lm.s; lm=lm[lm.mse>=0].copy()
lm["base_rate"]=1000*lm.base/lm.base_mm
lm["perinatal"]=((lm.postpartum==1)|(lm.prenatal==1)).astype(int)
for c in ["gender","race","state"]: lm[c]=lm[c].fillna("u")
lm["age"]=lm.age.fillna(lm.age.median());lm["risk"]=lm.risk.fillna(lm.risk.median());lm["tier1"]=lm.tier1.fillna(0)
lm.to_parquet("/tmp/did_lm.parquet")

cf=["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","postpartum","prenatal"]
def design(d):
    Xn=d[["base_rate","mse","risk","tier1","age"]+cf].astype(float)
    Xc=pd.get_dummies(d[["gender","race","state"]],drop_first=True).astype(float)
    Z=pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1)
    Z=((Z-Z.mean())/Z.std(ddof=0).replace(0,1)).fillna(0)
    ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,d.treat).predict_proba(Z)[:,1],1e-3,1-1e-3)
    return ps,Z
def did_poisson(d,w,ycol="post",mmcol="post_mm",basecol="base",basemm="base_mm"):
    pre=d[["person_id","treat"]].copy();pre["y"]=d[basecol];pre["mm"]=d[basemm];pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d[ycol];po["mm"]=d[mmcol];po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w
              ).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"];return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)
def evalue(rr,hi):
    def e(x): x=min(x,1/x); return 1/x+np.sqrt((1/x)*(1/x-1))
    return round(e(rr),2), round(e(hi),2)   # point + CI-bound nearest null

for name,mask in [("PERINATAL",lm.perinatal==1),("ALL",pd.Series(True,index=lm.index))]:
    d=lm[mask].copy(); ps,Z=design(d)
    ow=np.where(d.treat==1,1-ps,ps)                                   # overlap
    sw=np.where(d.treat==1,1/ps,1/(1-ps)); sw=np.clip(sw,None,np.quantile(sw,0.99))  # IPTW trimmed
    r_o=did_poisson(d,ow); r_i=did_poisson(d,sw)
    # 1:5 PS matching (treated to controls)
    t=d[d.treat==1]; c=d[d.treat==0]
    nn=NearestNeighbors(n_neighbors=min(5,len(c))).fit(ps[d.treat==0].reshape(-1,1))
    _,iidx=nn.kneighbors(ps[d.treat==1].reshape(-1,1))
    keep=np.unique(np.concatenate([c.index.values[iidx.ravel()], t.index.values]))
    dm=d.loc[keep]; r_m=did_poisson(dm,np.ones(len(dm)))
    # formal parallel pre-trend: DiD on prepre vs base (should be ~1)
    r_p=did_poisson(d,ow,ycol="base",mmcol="base_mm",basecol="prepre",basemm="prepre_mm")
    ev=evalue(r_o[0],r_o[2])
    print(f"\n### {name}  (n_treated_lm={int(d.treat.sum())})")
    print(f"  Overlap-wt DiD   RR={r_o[0]:.3f} ({r_o[1]:.3f}-{r_o[2]:.3f})  [PRIMARY]")
    print(f"  IPTW DiD         RR={r_i[0]:.3f} ({r_i[1]:.3f}-{r_i[2]:.3f})")
    print(f"  1:5 PS-matched   RR={r_m[0]:.3f} ({r_m[1]:.3f}-{r_m[2]:.3f})")
    print(f"  E-value (pt, CI): {ev[0]} , {ev[1]}")
    print(f"  PARALLEL PRE-TREND test (prepre->base, want ~1, CI incl 1): RR={r_p[0]:.3f} ({r_p[1]:.3f}-{r_p[2]:.3f})")
