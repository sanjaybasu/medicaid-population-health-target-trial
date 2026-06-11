import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from differences import ATTgt
pd.set_option("display.width",200)
df=pd.read_parquet("/tmp/cs_panel.parquet")
peri=df[df.perinatal==1].copy()
covars="risk + age + diabetes + htn"
print("perinatal panel rows:",len(peri),"persons:",peri.person_id.nunique(),"cohorts(monthly):",peri.cohort.nunique())

def cs_eventstudy(d,coarsen=None,label=""):
    d=d.copy()
    if coarsen=="Q":
        d["cohort"]=(d.cohort//3)*3; d["time"]=(d.time//3)*3
        d=d.groupby(["person_id","time","cohort"],as_index=False).agg(
            acute_p1k=("acute_p1k","mean"),risk=("risk","first"),age=("age","first"),
            diabetes=("diabetes","first"),htn=("htn","first"))
    d=d.drop_duplicates(["person_id","time"]).set_index(["person_id","time"]).sort_index()
    att=ATTgt(data=d, cohort_column="cohort", base_period="universal", anticipation=1)
    att.fit(formula=f"acute_p1k ~ {covars}", control_group="not_yet_treated", est_method="dr", n_jobs=4, progress_bar=False)
    es=att.aggregate("event")
    es.columns=['_'.join([str(x) for x in c if str(x)!='']).strip('_') for c in es.columns.values]
    es=es.reset_index(); es.rename(columns={es.columns[0]:"e"},inplace=True)
    ac=[c for c in es.columns if c.endswith("ATT")][0]; sc=[c for c in es.columns if "std_error" in c][0]
    es["e"]=pd.to_numeric(es["e"],errors="coerce"); es=es.dropna(subset=["e"]); es=es[es[sc]<1e6]
    return es,ac,sc

for coarsen,lab in [(None,"monthly"),("Q","quarterly")]:
    try:
        es,ac,sc=cs_eventstudy(peri,coarsen,lab)
        scale = 1 if coarsen is None else 1   # event time in months or quarters
        pre=es[(es.e<0)&(es.e>=(-12 if coarsen is None else -4))]
        post=es[(es.e>=1)&(es.e<=(12 if coarsen is None else 4))]
        if len(post)==0: print(f"\n[{lab}] no post periods"); continue
        eff=post[ac].mean(); eff_se=np.sqrt((post[sc]**2).mean()/len(post))
        maxpre=pre[ac].abs().max(); 
        print(f"\n### Perinatal CS-DR event study [{lab}] (conditional parallel trends)")
        print(es[[ "e",ac,sc]].round(1).to_string(index=False))
        print(f"  POST effect (avg): {eff:.1f}/1000mm (95% CI {eff-1.96*eff_se:.1f},{eff+1.96*eff_se:.1f})")
        print(f"  max pre-trend |coef|: {maxpre:.1f}")
        if maxpre>0:
            mbar_pt=abs(eff)/maxpre; mbar_ci=(abs(eff)-1.96*eff_se)/maxpre
            print(f"  Honest-DiD relative-magnitude breakdown M-bar: point {mbar_pt:.2f}, CI-lower {mbar_ci:.2f}")
            print(f"  => maternal effect survives pre-trend violations up to {mbar_ci:.1f}x the largest observed pre-trend")
        break
    except Exception as e:
        print(f"\n[{lab}] CS failed: {str(e)[:80]}")
