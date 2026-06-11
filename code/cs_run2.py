import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from differences import ATTgt
pd.set_option("display.width",220); pd.set_option("display.max_rows",60)
df=pd.read_parquet("/tmp/cs_panel.parquet")
covars="risk + age + diabetes + htn + chf + copd + sud + any_bh"

def run(d,label,anticipation=1):
    d=d.drop_duplicates(["person_id","time"]).set_index(["person_id","time"]).sort_index()
    att=ATTgt(data=d, cohort_column="cohort", base_period="universal", anticipation=anticipation)
    att.fit(formula=f"acute_p1k ~ {covars}", control_group="not_yet_treated", est_method="dr",
            n_jobs=4, progress_bar=False)
    es=att.aggregate("event")
    es.columns=['_'.join([str(x) for x in c if str(x)!='']).strip('_') for c in es.columns.values]
    es=es.reset_index().rename(columns={es.reset_index().columns[0]:"e"})
    attcol=[c for c in es.columns if c.endswith("ATT") or c=="ATT"][0]
    secol=[c for c in es.columns if "std_error" in c][0]
    es["e"]=pd.to_numeric(es["e"],errors="coerce"); es=es.dropna(subset=["e"])
    es=es[es[secol]<1e6]   # drop singular cells
    pre=es[(es.e<=-2)&(es.e>=-12)]; post=es[(es.e>=1)&(es.e<=12)]
    # member-month weighting ~ equal; simple averages of event-time ATTs
    eff=post[attcol].mean(); pretrend=pre[attcol].abs().mean()
    print(f"\n### {label}: CS-DR, anticipation={anticipation}, universal base (pre-trigger)")
    print(f"  POST effect (avg e=1..12), acute/1000mm: {eff:.1f}")
    print(f"  pre-trend (avg |ATT| e=-2..-12):         {pretrend:.1f}   (want small vs effect)")
    print(f"  e=-1 (trigger month) ATT: {es.loc[es.e==-1,attcol].values}")
    print("  event path e=-6..+6:")
    print(es[(es.e>=-6)&(es.e<=6)][["e",attcol,secol]].round(1).to_string(index=False))
    return es

es=run(df,"ALL (program-wide)")
es.to_parquet("/tmp/cs_es_ALL.parquet")
