import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import pyfixest as pf
df=pd.read_parquet("/tmp/cs_panel.parquet")
df["ms"]=df.time-df.cohort
# ---- Borusyak-Jaravel-Spiess imputation estimator ----
# Y(0) model on not-yet-treated obs (ms<=-2, excludes trigger month -1 and activation 0); impute for treated (ms 1..12)
unt=df[df.ms.between(-15,-2)].copy()
tre=df[df.ms.between(1,12)].copy()
m=pf.feols("acute_p1k ~ 1 | person_id + time", data=unt)
yhat0=m.predict(newdata=tre)
tre=tre.assign(yhat0=yhat0).dropna(subset=["yhat0"])
att_bjs=(tre.acute_p1k - tre.yhat0).mean()
# cluster-bootstrap SE by person
rng=np.random.RandomState(20260611); ids=tre.person_id.unique(); bs=[]
for b in range(80):
    samp=rng.choice(ids,len(ids),replace=True)
    sub=tre[tre.person_id.isin(samp)]
    bs.append((sub.acute_p1k - sub.yhat0).mean())
se=np.std(bs)
print(f"Borusyak-Jaravel-Spiess imputation ATT (acute/1000mm): {att_bjs:.1f} (95% CI {att_bjs-1.96*se:.1f}, {att_bjs+1.96*se:.1f}); n_treated_obs={len(tre)}")

# ---- Sun & Abraham interaction-weighted event study ----
try:
    d=df.copy(); d["rel"]=d.ms
    # last cohort as reference (no never-treated); pyfixest sunab
    m2=pf.feols("acute_p1k ~ sunab(cohort, time) | person_id + time", data=d, vcov={"CRV1":"person_id"})
    ct=m2.tidy().reset_index()
    ct["e"]=ct.iloc[:,0].astype(str).str.extract(r'(-?\d+)').astype(float)
    post=ct[(ct.e>=1)&(ct.e<=12)]
    print(f"Sun-Abraham IW event study: POST avg (e=1..12) = {post['Estimate'].mean():.1f}/1000mm (interaction-weighted)")
except Exception as e:
    print("Sun-Abraham via pyfixest failed:",str(e)[:120])
    # manual fallback: cohort x relative-time interactions, aggregate by cohort share
    try:
        d=df[df.ms.between(-12,12)].copy()
        d=d[d.ms!=-1]; d["msf"]=d.ms.astype(int).astype(str)
        import statsmodels.formula.api as smf, statsmodels.api as sm
        # not run fully due to size; report conceptual
        print("  (manual Sun-Abraham not executed; CS is the primary heterogeneity-robust estimator)")
    except Exception as e2: print("  manual also failed")
print(f"\nReference: CS-DR primary ATT ~ -180/1000mm; overlap/IPTW/matched RR 0.60-0.63")
