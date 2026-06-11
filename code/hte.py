import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
rng=np.random.RandomState(20260611)
d=pd.read_parquet("/tmp/did_lm.parquet").copy()
d["post_rate"]=1000*d.post/d.post_mm.clip(lower=0.1)
d["base_rate"]=1000*d.base/d.base_mm.clip(lower=0.1)
d["prepre_rate"]=1000*d.prepre/d.prepre_mm.clip(lower=0.1)
for c in ["race","gender","state"]: d[c]=d[c].fillna("u")
d["age"]=d.age.fillna(d.age.median()); d["risk"]=d.risk.fillna(d.risk.median())
cond=["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","perinatal"]
Xnum=["base_rate","prepre_rate","risk","age","mse"]+cond
Xcat=pd.get_dummies(d[["race","gender","state"]],drop_first=True).astype(float)
X=pd.concat([d[Xnum].astype(float).reset_index(drop=True),Xcat.reset_index(drop=True)],axis=1).values
W=d.treat.values.astype(int); Y=d.post_rate.values
n=len(d); print("HTE cohort:",n,"treated:",W.sum())

# cross-fitted AIPW scores + T-learner CATE for ranking
folds=StratifiedKFold(5,shuffle=True,random_state=20260611)
Gamma=np.zeros(n); tau=np.zeros(n); ehat=np.zeros(n)
for tr,te in folds.split(X,W):
    e=GradientBoostingClassifier(max_depth=3,n_estimators=200,subsample=.7,random_state=1).fit(X[tr],W[tr]).predict_proba(X[te])[:,1]
    e=np.clip(e,.02,.98); ehat[te]=e
    m1=GradientBoostingRegressor(max_depth=3,n_estimators=300,subsample=.7,random_state=2).fit(X[tr][W[tr]==1],Y[tr][W[tr]==1])
    m0=GradientBoostingRegressor(max_depth=3,n_estimators=300,subsample=.7,random_state=3).fit(X[tr][W[tr]==0],Y[tr][W[tr]==0])
    mu1=m1.predict(X[te]); mu0=m0.predict(X[te])
    Gamma[te]=(mu1-mu0)+W[te]/e*(Y[te]-mu1)-(1-W[te])/(1-e)*(Y[te]-mu0)
    tau[te]=mu1-mu0     # CATE (negative = benefit, reduces acute care)
ate=Gamma.mean(); ate_se=Gamma.std()/np.sqrt(n)
print(f"\nAIPW ATE (post acute /1000mm): {ate:.1f} (95% CI {ate-1.96*ate_se:.1f},{ate+1.96*ate_se:.1f})  [negative=benefit]")

# RATE / TOC (Yadlowsky): rank by CATE (most benefit = most negative tau -> prioritize)
order=np.argsort(tau)            # ascending: most negative (most benefit) first
G=Gamma[order]
qs=np.linspace(0.05,1,20); toc=[]
for q in qs:
    k=max(int(q*n),1)
    toc.append(G[:k].mean()-ate)     # benefit among top-q vs overall (more negative=better targeting)
toc=np.array(toc)
autoc=np.trapz(toc,qs)/ (qs[-1]-qs[0])
# AUTOC SE via influence approx (bootstrap over persons)
boot=[]
for b in range(200):
    idx=rng.randint(0,n,n); Gi=Gamma[idx]; ti=tau[idx]; oi=np.argsort(ti); Gs=Gi[oi]; ai=Gi.mean()
    tv=[Gs[:max(int(q*n),1)].mean()-ai for q in qs]; boot.append(np.trapz(tv,qs)/(qs[-1]-qs[0]))
autoc_se=np.std(boot)
print(f"AUTOC (targeting value, more negative=stronger HTE/targeting gain): {autoc:.1f} (SE {autoc_se:.1f}, p={2*min((np.array(boot)>0).mean(),(np.array(boot)<0).mean()):.3f})")

# Qini-style: cumulative incremental benefit from targeting top-x by CATE
print("\nTOC curve (benefit per 1000mm beyond average, by targeted fraction):")
for q in [0.1,0.2,0.3,0.5,0.7,1.0]:
    k=max(int(q*n),1); print(f"  top {int(q*100):3d}% by predicted benefit: extra {G[:k].mean()-ate:+.1f} vs avg ; abs effect {G[:k].mean():+.1f}")

# who benefits most: top-decile vs rest characteristics
d["tau"]=tau; top=d.nsmallest(int(.1*n),"tau"); rest=d.drop(top.index)
print("\nTop-decile (largest predicted benefit) vs rest — mean covariates:")
for c in ["base_rate","risk","age","high_ed_ip","chf","copd","sud","any_bh","diabetes","htn"]:
    print(f"  {c:12s}: top {top[c].mean():8.2f} | rest {rest[c].mean():8.2f}")

# equity: AIPW ATE by subgroup
print("\nEquity — AIPW ATE by subgroup (negative=benefit):")
d["Gamma"]=Gamma
for col,grps in [("race",d.race.value_counts().head(4).index)]:
    for gv in grps:
        s=d[d[col]==gv]; print(f"  race={gv:10s} n={len(s):5d}  ATE={s.Gamma.mean():+.1f} ({s.Gamma.mean()-1.96*s.Gamma.std()/np.sqrt(len(s)):+.1f},{s.Gamma.mean()+1.96*s.Gamma.std()/np.sqrt(len(s)):+.1f})")
for lab,mask in [("risk top tertile",d.risk>=d.risk.quantile(.67)),("risk bottom tertile",d.risk<d.risk.quantile(.33)),
                 ("high_ed_ip=1",d.high_ed_ip==1),("behavioral health=1",d.any_bh==1)]:
    s=d[mask]; print(f"  {lab:20s} n={len(s):5d}  ATE={s.Gamma.mean():+.1f} ({s.Gamma.mean()-1.96*s.Gamma.std()/np.sqrt(len(s)):+.1f},{s.Gamma.mean()+1.96*s.Gamma.std()/np.sqrt(len(s)):+.1f})")
np.savez("/tmp/hte_curves.npz",qs=qs,toc=toc,tau=tau,Gamma=Gamma)
