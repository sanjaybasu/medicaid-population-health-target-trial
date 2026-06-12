import os, pathlib
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

panel=pd.read_parquet(str(CACHE/"panel.parquet"))
attrs=pd.read_parquet(str(CACHE/"attrs.parquet"))
for c in ["postpartum","prenatal","diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","tier1","nopcp","rr"]:
    attrs[c]=pd.to_numeric(attrs[c],errors="coerce").fillna(0)
attrs["elig"]=pd.to_datetime(attrs.elig,errors="coerce");attrs["zd"]=pd.to_datetime(attrs.zero_date,errors="coerce")
attrs["e2a"]=((attrs.zd-attrs.elig).dt.days/30.44)
attrs["zmon"]=attrs.zd.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan)
panel["zd"]=pd.to_datetime(panel.zero_date,errors="coerce")
panel["zmon"]=panel.zd.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan)
panel["calmon"]=panel.zmon+panel.ms
CUT=pd.Period("2025-12","M").ordinal
OUTS=["acute","ed","ip","acsed","pqi","pcp","office","tele","bh","rx"]
cf6=["diabetes","htn","chf","copd","sud","any_bh"]
cf12=cf6+["mdd","asthma","polypharmacy","high_ed_ip","postpartum","prenatal"]

def cohort(rrflag=True, mature=True):
    # Censor immature person-MONTHS only (calmon>Dec2025); keep later-activating patients'
    # mature pre-period as not-yet-treated controls (sequential-trial design). Their own
    # activation month is censored, so they cannot enter as treated units with immature post-data.
    p=panel.copy()
    if mature: p=p[p.calmon<=CUT]
    if rrflag:
        keep=set(attrs[attrs.rr==1].person_id); p=p[p.person_id.isin(keep)]
    return p, attrs

def build_lm(p,a):
    cols=list(range(-27,13));
    M={c:p.pivot_table(index="person_id",columns="ms",values=c,aggfunc="sum",fill_value=0).reindex(columns=cols,fill_value=0) for c in OUTS+["enrolled_days"]}
    idx=M["acute"].index
    def ws(MM,lo,hi):
        cc=[x for x in cols if lo<=x<=hi]; return MM[cc].sum(axis=1) if cc else pd.Series(0.0,index=idx)
    WASH=1; rows=[]
    for s in range(-12,1):
        tr=1 if s==0 else 0; plo=s+1+WASH if tr else s+1; phi=12 if tr else -1-WASH
        if plo>phi: continue
        d=pd.DataFrame({"person_id":idx,"s":s,"treat":tr,"obs":(M["enrolled_days"][s]>0).values,
            "mm_b":(ws(M["enrolled_days"],s-12,s-7)/30.44).values,"mm_p":(ws(M["enrolled_days"],plo,phi)/30.44).values,
            "prepre":ws(M["acute"],s-18,s-13).values,"prepre_mm":(ws(M["enrolled_days"],s-18,s-13)/30.44).values})
        for c in OUTS: d[c+"_b"]=ws(M[c],s-12,s-7).values; d[c+"_p"]=ws(M[c],plo,phi).values
        rows.append(d[(d.obs)&(d.mm_p>0.3)&(d.mm_b>0.3)])
    lm=pd.concat(rows,ignore_index=True).merge(a,on="person_id",how="left")
    lm["mse"]=lm.e2a+lm.s; lm=lm[lm.mse>=0].copy()
    for c in ["gender","race","state"]: lm[c]=lm[c].fillna("u")
    lm["age"]=lm.age.fillna(lm.age.median()); lm["risk"]=lm.risk.fillna(lm.risk.median())
    return lm

def ps_weights(d, numeric, cond, clip=(1e-3,1-1e-3)):
    Xn=d[numeric+cond].astype(float)
    Xc=pd.get_dummies(d[["gender","race","state"]],drop_first=True).astype(float)
    Z=pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1)
    Z=((Z-Z.mean())/Z.std(ddof=0).replace(0,1)).fillna(0)
    ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,d.treat).predict_proba(Z)[:,1],*clip)
    return ps

def did(d,bcol,pcol,w,mmb="mm_b",mmp="mm_p"):
    pre=d[["person_id","treat"]].copy();pre["y"]=d[bcol].values;pre["mm"]=d[mmb].values;pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d[pcol].values;po["mm"]=d[mmp].values;po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"];return round(np.exp(b),3),round(np.exp(b-1.96*se),3),round(np.exp(b+1.96*se),3)

def evalue(rr): x=min(rr,1/rr); return round(1/x+np.sqrt((1/x)*(1/x-1)),2)

def primary_block(lm, label):
    d=lm.copy(); d["base_rate"]=1000*d.acute_b/d.mm_b.clip(lower=0.1)
    ps=ps_weights(d,["base_rate","mse","risk","age"],cf6)   # 6-cond spec (matches locked v3 rerun_mature)
    ow=np.where(d.treat==1,1-ps,ps)
    sw=np.where(d.treat==1,1/ps,1/(1-ps)); sw=np.clip(sw,None,np.quantile(sw,0.99))
    r_o=did(d,"acute_b","acute_p",ow); r_i=did(d,"acute_b","acute_p",sw)
    t=d[d.treat==1];c=d[d.treat==0]
    nn=NearestNeighbors(n_neighbors=min(5,len(c))).fit(ps[(d.treat==0).values].reshape(-1,1))
    _,ii=nn.kneighbors(ps[(d.treat==1).values].reshape(-1,1))
    keep=np.unique(np.concatenate([c.index.values[ii.ravel()],t.index.values])); dm=d.loc[keep]
    r_m=did(dm,"acute_b","acute_p",np.ones(len(dm)))
    r_p=did(d,"prepre","acute_b",ow,mmb="prepre_mm",mmp="mm_b")  # pre-trend prepre->base
    return dict(label=label,n_treated=int(d.treat.sum()),n_control_periods=int((d.treat==0).sum()),
                overlap=r_o,iptw=r_i,matched=r_m,evalue=evalue(r_o[0]),pretrend=r_p)

def sec_block(lm,bcol,pcol):  # 6-cond spec, overlap only
    d=lm.copy(); d["base_rate"]=1000*d[bcol]/d.mm_b.clip(lower=0.1)
    ow=np.where(d.treat==1,1-ps_weights(d,["base_rate","mse","risk","age"],cf6),ps_weights(d,["base_rate","mse","risk","age"],cf6))
    return did(d,bcol,pcol,ow)

R={}
# ---------- CANONICAL: rr_flag=1, mature ----------
p,a=cohort(rrflag=True,mature=True); lm=build_lm(p,a)
R["cohort"]={"activated_persons":int(attrs[(attrs.rr==1)&(attrs.zmon<=CUT)].shape[0]),"landmark_treated":int(lm.treat.sum()),"comparison_periods":int((lm.treat==0).sum())}
R["primary"]=primary_block(lm,"rr_flag=1 mature")
# event study (mature rr_flag); descriptive rates use enrolled months only
pe=p[p.enrolled_days>0]
es=pe.groupby("ms").apply(lambda x:1000*x.acute.sum()/(x.enrolled_days.sum()/30.44))
R["event_study"]={int(m):round(float(es.get(m,np.nan)),0) for m in range(-12,13)}
R["event_study_summary"]={"trigger_-1":round(float(es.get(-1))),"baseline_-6to-2":round(float(es.loc[-6:-2].mean())),"post_1to12":round(float(es.loc[1:12].mean()))}
# naive windows (mature rr_flag)
def rate(lo,hi):
    x=pe[pe.ms.between(lo,hi)]; return 1000*x.acute.sum()/(x.enrolled_days.sum()/30.44)
post=rate(1,12)
R["naive"]={"post_1to12":round(post),
  "quarter":round(100*(1-post/rate(-1,-1))),"m3":round(100*(1-post/rate(-3,-1))),
  "m6":round(100*(1-post/rate(-6,-1))),"m12":round(100*(1-post/rate(-12,-1)))}
# disaggregated outcomes
R["by_outcome"]={"ED":sec_block(lm,"ed_b","ed_p"),"hospitalization":sec_block(lm,"ip_b","ip_p"),
  "ACS_ED_NYU":sec_block(lm,"acsed_b","acsed_p"),"ACS_hosp_PQI":sec_block(lm,"pqi_b","pqi_p")}
# equity race/gender
eq={}
for col,vals in [("race",["Black or African American","White","Hispanic","Asian"]),("gender",["Female","Male"])]:
    for v in vals:
        d=lm[lm[col]==v]; nt=int(d.treat.sum())
        if nt<30: eq[v]={"n_treated":nt,"rr":None}; continue
        d2=d.copy(); d2["base_rate"]=1000*d2.acute_b/d2.mm_b.clip(lower=0.1)
        ow=np.where(d2.treat==1,1-ps_weights(d2,["base_rate","mse","risk","age"],cf6),ps_weights(d2,["base_rate","mse","risk","age"],cf6))
        eq[v]={"n_treated":nt,"rr":did(d2,"acute_b","acute_p",ow)}
R["equity_race_gender"]=eq
# mechanism: overall process outcomes (overlap from acute-PS) + PCP by connection
d=lm.copy(); d["base_rate"]=1000*d.acute_b/d.mm_b.clip(lower=0.1)
ow=np.where(d.treat==1,1-ps_weights(d,["base_rate","mse","risk","age"],cf6),ps_weights(d,["base_rate","mse","risk","age"],cf6))
R["mechanism"]={nm:did(d,c+"_b",c+"_p",ow) for nm,c in [("pcp_overall","pcp"),("office","office"),("telehealth","tele"),("bh","bh"),("pharmacy","rx")]}
for lab,mask in [("pcp_unconnected",lm.nopcp==1),("pcp_connected",lm.nopcp==0)]:
    d2=lm[mask].copy(); d2["base_rate"]=1000*d2.pcp_b/d2.mm_b.clip(lower=0.1)
    ow2=np.where(d2.treat==1,1-ps_weights(d2,["base_rate","mse","risk","age"],cf6),ps_weights(d2,["base_rate","mse","risk","age"],cf6))
    R["mechanism"][lab]={"n_treated":int(d2.treat.sum()),"rr":did(d2,"pcp_b","pcp_p",ow2)}
# equity by Area Deprivation Index (UW Neighborhood Atlas national percentile via ZCTA)
zadi=pd.read_csv(str(CACHE/"zip_adi.csv"),dtype={"zip":str})
la=lm.copy(); la["zip"]=la.postal.astype(str).str.extract(r'(\d{5})')[0]
la=la.merge(zadi[["zip","adi_natrank"]],on="zip",how="left")
R.setdefault("equity_adi",{})["match_rate"]=round(float(la.adi_natrank.notna().mean()),3)
la=la.dropna(subset=["adi_natrank"]).copy(); la["adi_t"]=pd.qcut(la.adi_natrank,3,labels=["low","med","high"])
adi={}
for tname in ["low","med","high"]:
    d=la[la.adi_t==tname].copy(); d["base_rate"]=1000*d.acute_b/d.mm_b.clip(lower=0.1)
    ow=np.where(d.treat==1,1-ps_weights(d,["base_rate","mse","risk","age"],cf6),ps_weights(d,["base_rate","mse","risk","age"],cf6))
    adi[tname]={"n_treated":int(d.treat.sum()),"median_adi":round(float(d.adi_natrank.median())),"rr":did(d,"acute_b","acute_p",ow)}
R["equity_adi"]["tertiles"]=adi
# Table 1 baseline characteristics (treated vs comparison; overlap-weighted SMD)
d=lm.copy(); d["base_rate"]=1000*d.acute_b/d.mm_b.clip(lower=0.1)
ps=ps_weights(d,["base_rate","mse","risk","age"],cf6); owt=np.where(d.treat==1,1-ps,ps)
t=d[d.treat==1].copy(); c=d[d.treat==0].copy(); wt=owt[(d.treat==1).values]; wc=owt[(d.treat==0).values]
def smd(col): sd=np.sqrt((t[col].var()+c[col].var())/2); return round((t[col].mean()-c[col].mean())/sd,3) if sd>0 else 0.0
def wsmd(col):
    sd=np.sqrt((t[col].var()+c[col].var())/2); return round((np.average(t[col],weights=wt)-np.average(c[col],weights=wc))/sd,3) if sd>0 else 0.0
t1={"n_treated":int(len(t)),"n_comparison":int(len(c)),
    "age":[round(t.age.mean(),1),round(t.age.std(),1),round(c.age.mean(),1),round(c.age.std(),1),smd("age"),wsmd("age")],
    "female_pct":[round(100*(t.gender=="Female").mean(),1),round(100*(c.gender=="Female").mean(),1)],
    "risk":[round(t.risk.mean(),1),round(c.risk.mean(),1),smd("risk"),wsmd("risk")],
    "base_acute":[round(1000*t.acute_b.sum()/t.mm_b.sum()),round(1000*c.acute_b.sum()/c.mm_b.sum()),smd("base_rate"),wsmd("base_rate")]}
for r in ["Black or African American","White","Hispanic","Asian"]:
    t1["race_"+r]=[round(100*(t.race==r).mean(),1),round(100*(c.race==r).mean(),1)]
for x in cf6:
    t1["cond_"+x]=[round(100*t[x].mean(),1),round(100*c[x].mean(),1),smd(x),wsmd(x)]
t1["max_abs_wsmd"]=round(max(abs(wsmd(x)) for x in ["age","risk","base_rate"]+cf6),3)
R["table1"]=t1
# ---------- immature contrast (rr_flag=1, NO maturity) ----------
p2,a2=cohort(rrflag=True,mature=False); lm2=build_lm(p2,a2)
R["immature_contrast"]=primary_block(lm2,"rr_flag=1 all-data")
# ---------- VALIDATION: ALL pathways, mature (should reproduce locked v3 ~0.79) ----------
pA,aA=cohort(rrflag=False,mature=True); lmA=build_lm(pA,aA)
R["validation_all_pathways"]=primary_block(lmA,"ALL mature")["overlap"]
json.dump(R,open(str(ROOT/"results.json"),"w"),indent=1,default=str)
print("=== CANONICAL rr_flag=1 mature ===")
print("cohort:",R["cohort"])
print("event study summary:",R["event_study_summary"])
print("naive:",R["naive"])
print("primary overlap RR:",R["primary"]["overlap"],"IPTW:",R["primary"]["iptw"],"matched:",R["primary"]["matched"],"E-value:",R["primary"]["evalue"],"pretrend:",R["primary"]["pretrend"])
print("by_outcome:",{k:v for k,v in R["by_outcome"].items()})
print("equity:",{k:v["rr"] for k,v in R["equity_race_gender"].items()})
print("mechanism:",{k:(v if isinstance(v,tuple) else v) for k,v in R["mechanism"].items()})
print("immature contrast overlap RR:",R["immature_contrast"]["overlap"])
