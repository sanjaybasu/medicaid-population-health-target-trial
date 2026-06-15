"""08_revision_robustness.py — supplementary robustness/estimand analyses.

Runs after 01 (primary results) and 02 (Callaway-Sant'Anna + Honest-DiD inputs).
Reads cache/panel.parquet, cache/attrs.parquet, and results.json; appends a
`revision_robustness` block to results.json. Adds nothing to the data pull.

Quantities:
  1. Inverse-probability-weighted DiD confidence interval at full precision
     (reconciles 2-decimal rounding across reported tables).
  2. ATT-reweighted Poisson DiD (inverse-odds weights) as a concordance check for
     the overlap-weighted (ATO) headline estimand.
  3. Honest-DiD relative-magnitude sensitivity curve, from the cached CS-DR event
     study (Rambachan & Roth): robust CIs for the additive effect across a grid of
     M (multiples of the largest pre-period coefficient) plus the breakdown M-bar.
  4. Baseline acute-care rate by race and Area Deprivation Index tertile among the
     treated (context for the equity / absolute-difference reading of the subgroup RRs).
  5. Cohort funnel and median time from eligibility to activation (estimand framing:
     the comparison group is composed of later-activating patients).

A validation gate reproduces the primary overlap rate ratio from results.json before
emitting any new quantity; if it fails, downstream outputs should not be trusted.
"""
import os, pathlib, warnings, json
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"
panel = pd.read_parquet(str(CACHE / "panel.parquet"))
attrs = pd.read_parquet(str(CACHE / "attrs.parquet"))

for c in ["postpartum","prenatal","diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","tier1","nopcp","rr"]:
    attrs[c]=pd.to_numeric(attrs[c],errors="coerce").fillna(0)
attrs["elig"]=pd.to_datetime(attrs.elig,errors="coerce"); attrs["zd"]=pd.to_datetime(attrs.zero_date,errors="coerce")
attrs["e2a"]=((attrs.zd-attrs.elig).dt.days/30.44)
attrs["zmon"]=attrs.zd.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan)
panel["zd"]=pd.to_datetime(panel.zero_date,errors="coerce")
panel["zmon"]=panel.zd.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan)
panel["calmon"]=panel.zmon+panel.ms
CUT=pd.Period("2025-12","M").ordinal
OUTS=["acute","ed","ip","acsed","pqi","pcp","office","tele","bh","rx"]
cf6=["diabetes","htn","chf","copd","sud","any_bh"]

def cohort(rrflag=True, mature=True):
    p=panel.copy()
    if mature: p=p[p.calmon<=CUT]
    if rrflag:
        keep=set(attrs[attrs.rr==1].person_id); p=p[p.person_id.isin(keep)]
    return p, attrs

def build_lm(p,a):
    cols=list(range(-27,13))
    M={c:p.pivot_table(index="person_id",columns="ms",values=c,aggfunc="sum",fill_value=0).reindex(columns=cols,fill_value=0) for c in OUTS+["enrolled_days"]}
    idx=M["acute"].index
    def ws(MM,lo,hi):
        cc=[x for x in cols if lo<=x<=hi]; return MM[cc].sum(axis=1) if cc else pd.Series(0.0,index=idx)
    WASH=1; rows=[]
    for s in range(-12,1):
        tr=1 if s==0 else 0; plo=s+1+WASH if tr else s+1; phi=12 if tr else -1-WASH
        if plo>phi: continue
        d=pd.DataFrame({"person_id":idx,"s":s,"treat":tr,"obs":(M["enrolled_days"][s]>0).values,
            "mm_b":(ws(M["enrolled_days"],s-12,s-7)/30.44).values,"mm_p":(ws(M["enrolled_days"],plo,phi)/30.44).values})
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
    return np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,d.treat).predict_proba(Z)[:,1],*clip)

def did(d,bcol,pcol,w,mmb="mm_b",mmp="mm_p",ndp=3):
    pre=d[["person_id","treat"]].copy();pre["y"]=d[bcol].values;pre["mm"]=d[mmb].values;pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d[pcol].values;po["mm"]=d[mmp].values;po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"]
    return round(np.exp(b),ndp),round(np.exp(b-1.96*se),ndp),round(np.exp(b+1.96*se),ndp)

R=json.load(open(str(ROOT/"results.json")))
out={}

# build canonical cohort
p,a=cohort(rrflag=True,mature=True); lm=build_lm(p,a)
lm["base_rate"]=1000*lm.acute_b/lm.mm_b.clip(lower=0.1)
ps=ps_weights(lm,["base_rate","mse","risk","age"],cf6)
ow=np.where(lm.treat==1,1-ps,ps)

# ---- validation gate ----
overlap=did(lm,"acute_b","acute_p",ow)
canon=R["primary"]["overlap"][0]
assert abs(overlap[0]-canon)<0.003, f"reproduction failed: {overlap[0]} vs {canon}"
out["validation"]={"overlap_rr_reproduced":overlap[0],"canonical":canon}

# 1. IPW full-precision CI
sw=np.where(lm.treat==1,1/ps,1/(1-ps)); sw=np.clip(sw,None,np.quantile(sw,0.99))
ipw=did(lm,"acute_b","acute_p",sw,ndp=4)
out["ipw_full_precision"]={"rr":ipw[0],"lo":ipw[1],"hi":ipw[2]}

# 2. ATT-reweighted (inverse-odds) concordance vs ATO overlap
attw=np.where(lm.treat==1,1.0,ps/(1-ps))
attrr=did(lm,"acute_b","acute_p",attw)
out["att_reweighted"]={"rr":attrr[0],"lo":attrr[1],"hi":attrr[2],"overlap_ATO_rr":overlap[0]}

# 3. Honest-DiD relative-magnitude sensitivity curve (from cached cs_dr)
cs=R.get("cs_dr")
if cs:
    eff=cs["post_ATT_per1000mm"]; se=cs["ci_halfwidth"]/1.96; mp=cs["max_pre_abs"]
    curve=[]
    for Mb in [0,0.5,1,1.5,2,2.5,3,3.5,4]:
        w=1.96*se+Mb*mp
        curve.append({"M":Mb,"lo":round(eff-w,1),"hi":round(eff+w,1),"excludes_null":bool((eff-w)>0 or (eff+w)<0)})
    out["honest_did_curve"]={"max_pre_abs_per1000mm":mp,"breakdown_Mbar_point":round(abs(eff)/mp,2),
                             "breakdown_Mbar_ci":round((abs(eff)-1.96*se)/mp,2),"curve":curve}

# 4. baseline acute-care rate by race and ADI tertile (treated)
t=lm[lm.treat==1]
def br(s): return round(1000*s.acute_b.sum()/s.mm_b.sum()) if len(s) and s.mm_b.sum()>0 else None
out["baseline_acute_by_race"]={r:{"n":int((t.race==r).sum()),"rate_per1000mm":br(t[t.race==r])}
                               for r in t.race.dropna().unique() if (t.race==r).sum()>=30}
try:
    zadi=pd.read_csv(str(CACHE/"zip_adi.csv"),dtype={"zip":str})
    la=t.copy(); la["zip"]=la.postal.astype(str).str.extract(r'(\d{5})')[0]
    la=la.merge(zadi[["zip","adi_natrank"]],on="zip",how="left").dropna(subset=["adi_natrank"])
    la["adi_t"]=pd.qcut(la.adi_natrank,3,labels=["low","med","high"])
    out["baseline_acute_by_adi"]={x:{"n":int((la.adi_t==x).sum()),"rate_per1000mm":br(la[la.adi_t==x])} for x in ["low","med","high"]}
except Exception as e:
    out["baseline_acute_by_adi"]={"error":str(e)}

# 5. cohort funnel + median time to activation
rr=attrs[attrs.rr==1]; act=rr[rr.zmon<=CUT]
out["cohort_funnel"]={"rr_cohort":int(len(rr)),"ever_activate_frac":round(float(rr.zmon.notna().mean()),3),
    "activated_by_cutoff":int(len(act)),"landmark_treated":int(lm.treat.sum()),"comparison_periods":int((lm.treat==0).sum()),
    "median_months_elig_to_activation":round(float(act.e2a.median()),1),
    "iqr":[round(float(act.e2a.quantile(.25)),1),round(float(act.e2a.quantile(.75)),1)]}

R["revision_robustness"]=out
json.dump(R,open(str(ROOT/"results.json"),"w"),indent=1,default=str)
print("validation overlap RR:",overlap[0],"(canonical",canon,")")
print("IPW full precision:",out["ipw_full_precision"])
print("ATT-reweighted:",out["att_reweighted"])
if cs: print("Honest-DiD breakdown M-bar:",out["honest_did_curve"]["breakdown_Mbar_point"],"(CI",out["honest_did_curve"]["breakdown_Mbar_ci"],"); max pre-dev",out["honest_did_curve"]["max_pre_abs_per1000mm"])
print("baseline by race:",{k:v["rate_per1000mm"] for k,v in out["baseline_acute_by_race"].items()})
print("cohort funnel:",out["cohort_funnel"])
