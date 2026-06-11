import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
P="/Users/sanjaybasu/waymark-local/notebooks/waymark-engagement-acute-care/"
d=pd.read_parquet("/tmp/did_lm.parquet").copy()
d["base_rate"]=1000*d.base/d.base_mm.clip(lower=0.1)
for c in ["race","gender","state"]: d[c]=d[c].fillna("Unknown")
d["age"]=d.age.fillna(d.age.median()); d["risk"]=d.risk.fillna(d.risk.median())
cond=["diabetes","htn","chf","copd","sud","any_bh","mdd","polypharmacy","high_ed_ip"]
feats=["base_rate","risk","age"]+cond
Z=((d[feats].astype(float)-d[feats].astype(float).mean())/d[feats].astype(float).std(ddof=0).replace(0,1)).fillna(0)
Xc=pd.get_dummies(d[["race","gender","state"]],drop_first=True).astype(float)
Zall=pd.concat([Z.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1)
ps=np.clip(LogisticRegression(max_iter=3000,C=0.5).fit(Zall,d.treat).predict_proba(Zall)[:,1],1e-3,1-1e-3)
d["w"]=np.where(d.treat==1,1-ps,ps); t=d[d.treat==1]; c=d[d.treat==0]
def smd(col):
    sd=np.sqrt((t[col].var()+c[col].var())/2); return (t[col].mean()-c[col].mean())/sd if sd>0 else 0
def wsmd(col):
    sd=np.sqrt((t[col].var()+c[col].var())/2)
    return (np.average(t[col],weights=t.w)-np.average(c[col],weights=c.w))/sd if sd>0 else 0
L=["| Characteristic | Activated (now), n=%s | Comparison (not-yet), n=%s | SMD | SMD (weighted) |"%(f"{len(t):,}",f"{len(c):,}"),
   "|---|---|---|---|---|"]
L.append(f"| Age, mean (SD) | {t.age.mean():.1f} ({t.age.std():.1f}) | {c.age.mean():.1f} ({c.age.std():.1f}) | {smd('age'):+.3f} | {wsmd('age'):+.3f} |")
L.append(f"| Female, % | {100*(t.gender=='Female').mean():.1f} | {100*(c.gender=='Female').mean():.1f} | — | — |")
for r in ["Black or African American","White","Hispanic","Asian"]:
    tt=(t.race==r).mean(); cc=(c.race==r).mean(); sd=np.sqrt((tt*(1-tt)+cc*(1-cc))/2); s=(tt-cc)/sd if sd>0 else 0
    L.append(f"| {r}, % | {100*tt:.1f} | {100*cc:.1f} | {s:+.3f} | — |")
L.append(f"| Risk percentile, mean | {t.risk.mean():.1f} | {c.risk.mean():.1f} | {smd('risk'):+.3f} | {wsmd('risk'):+.3f} |")
L.append(f"| Baseline acute care /1000 mm, mean | {t.base_rate.mean():.0f} | {c.base_rate.mean():.0f} | {smd('base_rate'):+.3f} | {wsmd('base_rate'):+.3f} |")
lab={"diabetes":"Diabetes","htn":"Hypertension","chf":"Heart failure","copd":"COPD","sud":"Substance use disorder","any_bh":"Any behavioral health","mdd":"Major depression","polypharmacy":"Polypharmacy","high_ed_ip":"High prior ED/IP"}
for x in cond:
    L.append(f"| {lab[x]}, % | {100*t[x].mean():.1f} | {100*c[x].mean():.1f} | {smd(x):+.3f} | {wsmd(x):+.3f} |")
maxw=max(abs(wsmd(x)) for x in ["age","risk","base_rate"]+cond)
L.append("")
L.append(f"*Sequential-trial landmark units; overlap weighting achieved covariate balance (maximum absolute weighted standardized mean difference {maxw:.3f}; balance target <0.10). Sex coded as recorded; cohort {100*(t.gender=='Female').mean():.0f}% female.*")
open(P+"table1_baseline.md","w").write("\n".join(L))

# TABLE 2 — main results (locked numbers from analysis_results.md)
T2="""## Table 2. Effect of program activation on avoidable acute care: naive vs bias-corrected estimates

| Analysis | Estimate | 95% CI |
|---|---|---|
| **Naive before-and-after (by baseline window)** | apparent reduction | |
| &nbsp;&nbsp;Quarter of enrollment (most common) | 64% | — |
| &nbsp;&nbsp;3 months pre | 52% | — |
| &nbsp;&nbsp;6 months pre | 46% | — |
| &nbsp;&nbsp;12 months pre | 41% | — |
| **Bias-corrected (target-trial emulation)** | rate ratio | |
| &nbsp;&nbsp;Callaway–Sant'Anna / overlap-weighted DiD (primary) | 0.63 (37% reduction) | 0.58–0.69 |
| &nbsp;&nbsp;Inverse-probability-weighted DiD | 0.63 | 0.58–0.69 |
| &nbsp;&nbsp;1:5 propensity-score matched | 0.60 | 0.55–0.65 |
| **Total cost of care** | −$454 PMPM | −$535 to −$373 |
| **Robustness** | | |
| &nbsp;&nbsp;Parallel-pre-trends falsification (should ≈1) | 0.97 | 0.89–1.06 |
| &nbsp;&nbsp;E-value (point estimate) | 2.54 | — |
| &nbsp;&nbsp;Honest-DiD relative-magnitude breakdown (M̄) | 1.72 | — |

PMPM, per member per month. Negative cost = savings. Acute care = emergency department visits + acute inpatient admissions per 1000 member-months. Naive estimates use within-enrollee before-and-after comparison with no concurrent control; the bias-corrected estimate uses not-yet-activated controls anchored at the activation event with the activation/trigger month excluded.
"""
open(P+"table2_results.md","w").write(T2)
print("Wrote table1_baseline.md and table2_results.md")
print("\n--- TABLE 1 ---"); print("\n".join(L[:8]),"\n...")
