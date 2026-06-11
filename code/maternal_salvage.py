import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
lm=pd.read_parquet("/tmp/did_lm.parquet")
peri=lm[lm.perinatal==1].copy()
peri["base_rate"]=1000*peri.base/peri.base_mm.clip(lower=0.1)
peri["prepre_rate"]=1000*peri.prepre/peri.prepre_mm.clip(lower=0.1)
print("perinatal landmark units:",len(peri),"| treated:",int(peri.treat.sum()),"| control:",int((peri.treat==0).sum()))

def did(d,w):
    pre=d[["person_id","treat"]].copy();pre["y"]=d.base;pre["mm"]=d.base_mm;pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d.post;po["mm"]=d.post_mm;po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w
              ).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"];return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)
def falsif(d,w):  # prepre->base : should be ~1
    pre=d[["person_id","treat"]].copy();pre["y"]=d.prepre;pre["mm"]=d.prepre_mm;pre["post"]=0;pre["w"]=w
    po =d[["person_id","treat"]].copy();po["y"]=d.base;po["mm"]=d.base_mm;po["post"]=1;po["w"]=w
    s=pd.concat([pre,po],ignore_index=True);s=s[s.mm>0.05];s["lo"]=np.log(s.mm)
    m=smf.glm("y ~ treat*post",data=s,family=sm.families.Poisson(),offset=s.lo,freq_weights=s.w
              ).fit(cov_type="cluster",cov_kwds={"groups":s.person_id})
    b=m.params["treat:post"];se=m.bse["treat:post"];return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)

feats=["base_rate","prepre_rate","mse","risk","age","htn","diabetes","sud","any_bh"]
peri["age"]=peri.age.fillna(peri.age.median());peri["risk"]=peri.risk.fillna(peri.risk.median())
X=peri[feats].astype(float); Z=((X-X.mean())/X.std(ddof=0).replace(0,1)).fillna(0)
ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Z,peri.treat).predict_proba(Z)[:,1],1e-3,1-1e-3)
peri["ps"]=ps

# (A) overlap-weighted, full
ow=np.where(peri.treat==1,1-ps,ps)
print("\n(A) OVERLAP-WEIGHTED (full perinatal):")
print("    effect DiD RR:",[round(x,3) for x in did(peri,ow)])
print("    falsification (want CI incl 1):",[round(x,3) for x in falsif(peri,ow)])

# (B) 1:3 PS-matching incl pre-trend -> re-test
t=peri[peri.treat==1]; c=peri[peri.treat==0]
k=min(3,len(c)//max(len(t),1) if len(t)>0 else 1); k=max(k,1)
nn=NearestNeighbors(n_neighbors=min(3,len(c))).fit(c[["ps"]].values)
_,idx=nn.kneighbors(t[["ps"]].values)
keep=np.unique(np.concatenate([c.index.values[idx.ravel()],t.index.values]))
mt=peri.loc[keep].copy()
print(f"\n(B) 1:3 PS-MATCHED on pre-trend+baseline (matched n={len(mt)}, treated={int(mt.treat.sum())}):")
# balance on pretrend
for v in ["base_rate","prepre_rate","mse"]:
    smd=(mt[mt.treat==1][v].mean()-mt[mt.treat==0][v].mean())/np.sqrt((mt[mt.treat==1][v].var()+mt[mt.treat==0][v].var())/2)
    print(f"    SMD {v}: {smd:+.3f}")
print("    effect DiD RR:",[round(x,3) for x in did(mt,np.ones(len(mt)))])
print("    falsification (want CI incl 1):",[round(x,3) for x in falsif(mt,np.ones(len(mt)))])
