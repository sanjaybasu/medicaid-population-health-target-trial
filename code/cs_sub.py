import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from differences import ATTgt
pd.set_option("display.width",200)
df=pd.read_parquet("/tmp/cs_panel.parquet")
covars="risk + age + diabetes + htn + chf + copd + sud + any_bh"
def run(d,label):
    d=d.drop_duplicates(["person_id","time"]).set_index(["person_id","time"]).sort_index()
    try:
        att=ATTgt(data=d, cohort_column="cohort", base_period="universal", anticipation=1)
        att.fit(formula=f"acute_p1k ~ {covars}", control_group="not_yet_treated", est_method="dr",
                n_jobs=4, progress_bar=False)
        es=att.aggregate("event")
        es.columns=['_'.join([str(x) for x in c if str(x)!='']).strip('_') for c in es.columns.values]
        es=es.reset_index(); es.rename(columns={es.columns[0]:"e"},inplace=True)
        ac=[c for c in es.columns if c.endswith("ATT")][0]; sc=[c for c in es.columns if "std_error" in c][0]
        es["e"]=pd.to_numeric(es["e"],errors="coerce"); es=es.dropna(subset=["e"]); es=es[es[sc]<1e6]
        post=es[(es.e>=1)&(es.e<=12)]; pre=es[(es.e<=-2)&(es.e>=-12)]
        eff=post[ac].mean(); eff_se=np.sqrt((post[sc]**2).mean()/max(len(post),1)); pt=pre[ac].abs().mean()
        return [label,d.reset_index().person_id.nunique(),round(eff,1),
                round(eff-1.96*eff_se,1),round(eff+1.96*eff_se,1),round(pt,1)]
    except Exception as e:
        return [label,d.reset_index().person_id.nunique(),np.nan,np.nan,np.nan,f"ERR:{str(e)[:30]}"]
rows=[run(df,"ALL")]
for c in ["perinatal","htn","diabetes","chf","copd","sud","any_bh","high_ed_ip"]:
    rows.append(run(df[df[c]==1],c))
rows.append(run(df[df.state=="VIRGINIA"],"VA"))
res=pd.DataFrame(rows,columns=["cohort","n_pers","ATT_post_/1k_mm","lo95","hi95","pretrend_absavg"])
print("### CS doubly-robust by subgroup (anticipation=1, not-yet-treated). ATT acute/1000mm, e=1..12. Negative=benefit.")
print(res.to_string(index=False))
