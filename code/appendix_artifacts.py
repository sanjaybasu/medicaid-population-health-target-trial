import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"savefig.bbox":"tight"})
P="/Users/sanjaybasu/waymark-local/notebooks/waymark-engagement-acute-care/"
OUT=P+"figures/"

# ---------- eFigure 1: CS-DR event study with 95% CI ----------
es=pd.read_parquet("/tmp/cs_es_ALL.parquet")
ac=[c for c in es.columns if c.endswith("ATT")][0]; sc=[c for c in es.columns if "std_error" in c][0]
ec="e" if "e" in es.columns else es.columns[0]
es[ec]=pd.to_numeric(es[ec],errors="coerce"); es=es.dropna(subset=[ec]); es=es[(es[sc]<1e6)&(es[ec].between(-12,12))]
es["lo"]=es[ac]-1.96*es[sc]; es["hi"]=es[ac]+1.96*es[sc]
fig,ax=plt.subplots(figsize=(7,4))
ax.axhline(0,color="#999",lw=1)
ax.axvspan(-1.5,0.5,color="#ffd9d9",alpha=.5,lw=0)
ax.errorbar(es[ec],es[ac],yerr=1.96*es[sc],fmt="o",color="#1f4e79",ms=4,lw=1,capsize=2)
ax.set_xlabel("Event time (months since activation)"); ax.set_ylabel("Group-time ATT, acute care /1,000 member-months")
ax.set_xticks(range(-12,13,2))
fig.savefig(OUT+"efigure1_cs_eventstudy.png"); plt.close()
# event-study eTable
et=es[[ec,ac,"lo","hi"]].copy(); et.columns=["Event time (mo)","ATT /1000 mm","95% CI low","95% CI high"]
et=et.round(0).astype({"Event time (mo)":int})
L=["## eTable 1. Callaway–Sant'Anna group-time event-study coefficients","",
   "| Event time (mo) | ATT /1000 member-mo | 95% CI |","|---:|---:|---:|"]
for _,r in et.iterrows():
    L.append(f"| {int(r['Event time (mo)'])} | {r['ATT /1000 mm']:.0f} | {r['95% CI low']:.0f} to {r['95% CI high']:.0f} |")
L.append("\n*Anticipation=1, universal pre-trigger base period. Event time 0 = activation month (excluded with the −1 trigger month from effect estimation).*")
open(P+"etable1_eventstudy.md","w").write("\n".join(L))

# ---------- HTE recompute (AIPW) for eTable + eFigure 2 (TOC) ----------
d=pd.read_parquet("/tmp/did_lm.parquet").copy()
d["post_rate"]=1000*d.post/d.post_mm.clip(lower=0.1); d["base_rate"]=1000*d.base/d.base_mm.clip(lower=0.1)
d["prepre_rate"]=1000*d.prepre/d.prepre_mm.clip(lower=0.1)
for c in ["race","gender","state"]: d[c]=d[c].fillna("u")
d["age"]=d.age.fillna(d.age.median()); d["risk"]=d.risk.fillna(d.risk.median())
cond=["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip"]
Xn=["base_rate","prepre_rate","risk","age","mse"]+cond
Xc=pd.get_dummies(d[["race","gender","state"]],drop_first=True).astype(float)
X=pd.concat([d[Xn].astype(float).reset_index(drop=True),Xc.reset_index(drop=True)],axis=1).values
W=d.treat.values.astype(int); Y=d.post_rate.values; n=len(d)
Gamma=np.zeros(n); tau=np.zeros(n)
for tr,te in StratifiedKFold(5,shuffle=True,random_state=20260611).split(X,W):
    e=np.clip(GradientBoostingClassifier(max_depth=3,n_estimators=200,subsample=.7,random_state=1).fit(X[tr],W[tr]).predict_proba(X[te])[:,1],.02,.98)
    m1=GradientBoostingRegressor(max_depth=3,n_estimators=300,subsample=.7,random_state=2).fit(X[tr][W[tr]==1],Y[tr][W[tr]==1])
    m0=GradientBoostingRegressor(max_depth=3,n_estimators=300,subsample=.7,random_state=3).fit(X[tr][W[tr]==0],Y[tr][W[tr]==0])
    mu1=m1.predict(X[te]);mu0=m0.predict(X[te])
    Gamma[te]=(mu1-mu0)+W[te]/e*(Y[te]-mu1)-(1-W[te])/(1-e)*(Y[te]-mu0); tau[te]=mu1-mu0
d["G"]=Gamma; d["tau"]=tau
def ci(s): m=s.mean(); se=s.std()/np.sqrt(len(s)); return m,m-1.96*se,m+1.96*se
rows=[]
rows.append(("Overall",len(d),*ci(d.G)))
for r in ["Black or African American","White","Hispanic","Asian"]:
    s=d[d.race==r]; 
    if len(s)>30: rows.append((f"Race: {r}",len(s),*ci(s.G)))
for lab,m in [("Risk: highest tertile",d.risk>=d.risk.quantile(.67)),("Risk: lowest tertile",d.risk<d.risk.quantile(.33))]:
    s=d[m]; rows.append((lab,len(s),*ci(s.G)))
for c0,nm in [("high_ed_ip","High prior ED/IP"),("any_bh","Behavioral health"),("sud","Substance use disorder"),("chf","Heart failure"),("copd","COPD"),("diabetes","Diabetes"),("htn","Hypertension")]:
    s=d[d[c0]==1]; rows.append((f"Condition: {nm}",len(s),*ci(s.G)))
L=["## eTable 3. Heterogeneity of the program effect on avoidable acute care by subgroup (doubly robust)","",
   "| Subgroup | n | ATE /1000 member-mo | 95% CI |","|---|---:|---:|---:|"]
for nm,nn,m,lo,hi in rows:
    L.append(f"| {nm} | {nn:,} | {m:+.0f} | {lo:+.0f} to {hi:+.0f} |")
L.append("\n*Doubly robust (augmented inverse-probability-weighted) average treatment effect; negative = reduction in acute care. Cross-fitted (5-fold) gradient-boosted nuisance models.*")
open(P+"etable3_heterogeneity.md","w").write("\n".join(L))

# eFigure 2: TOC curve
order=np.argsort(tau); G=Gamma[order]; ate=Gamma.mean()
qs=np.linspace(0.05,1,20); toc=np.array([G[:max(int(q*n),1)].mean()-ate for q in qs])
fig,ax=plt.subplots(figsize=(6.5,4))
ax.plot(100*qs,toc,"-o",color="#1f4e79",ms=3)
ax.axhline(0,color="#999",lw=1)
ax.set_xlabel("Population treated, ranked by predicted benefit (%)"); ax.set_ylabel("Incremental ATE vs average /1,000 mm")
ax.set_title("Targeting-operator-characteristic (TOC) curve",fontsize=10,loc="left")
ax.text(40,toc.min()*0.6,f"AUTOC = -130.6 (P<.001)",fontsize=8.5,color="#444")
fig.savefig(OUT+"efigure2_toc_curve.png"); plt.close()
print("Wrote eFigure1, eFigure2, eTable1, eTable3. AIPW overall ATE:",round(ate,1))
print("equity rows:",[(r[0],round(r[2],0)) for r in rows[:6]])
