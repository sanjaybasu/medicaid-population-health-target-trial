import os, pathlib
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
import json, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
R=json.load(open(str(ROOT/"results.json")))
P=OUT
FIG=P/"figures"
def rr(x): return f"{x[0]:.2f}"
def ci(x): return f"{x[1]:.2f}–{x[2]:.2f}"
pr=R["primary"]; bo=R["by_outcome"]; nv=R["naive"]; im=R["immature_contrast"]["overlap"]

# ---------- Table 1 ----------
t=R["table1"]
L=[f"| Characteristic | Activated, n={t['n_treated']:,} | Comparison periods, n={t['n_comparison']:,} | SMD | SMD (overlap-wt) |","|---|---|---|---|---|"]
a=t["age"]; L.append(f"| Age, mean (SD), y | {a[0]} ({a[1]}) | {a[2]} ({a[3]}) | {a[4]:+.3f} | {a[5]:+.3f} |")
L.append(f"| Female, % | {t['female_pct'][0]} | {t['female_pct'][1]} | — | — |")
for r_ in ["Black or African American","White","Hispanic","Asian"]:
    v=t["race_"+r_]; L.append(f"| {r_}, % | {v[0]} | {v[1]} | — | — |")
rk=t["risk"]; L.append(f"| Risk percentile, mean | {rk[0]} | {rk[1]} | {rk[2]:+.3f} | {rk[3]:+.3f} |")
ba=t["base_acute"]; L.append(f"| Baseline acute care /1000 patient-mo | {ba[0]} | {ba[1]} | {ba[2]:+.3f} | {ba[3]:+.3f} |")
lab={"diabetes":"Diabetes","htn":"Hypertension","chf":"Heart failure","copd":"COPD","sud":"Substance use disorder","any_bh":"Any behavioral health"}
for x in ["diabetes","htn","chf","copd","sud","any_bh"]:
    v=t["cond_"+x]; L.append(f"| {lab[x]}, % | {v[0]} | {v[1]} | {v[2]:+.3f} | {v[3]:+.3f} |")
L.append(f"\n*Rising-risk–targeted (rr_flag=1) sequential-trial landmark units, mature claims (activation through December 2025); overlap weighting achieved balance (maximum absolute weighted standardized mean difference {t['max_abs_wsmd']:.3f}). {t['female_pct'][0]:.0f}% female.*")
open(P/"table1_baseline.md","w").write("\n".join(L))

# ---------- Table 2 ----------
red=lambda x:f"{x[0]:.2f} ({100*(1-x[0]):.0f}% reduction)"
T=["## Table 2. Effect of program activation on avoidable acute care: naive vs bias-corrected estimates (rising-risk cohort, mature claims)","",
"| Analysis | Estimate | 95% CI |","|---|---|---|",
"| **Naive before-and-after (by baseline window)** | apparent reduction | |",
f"| &nbsp;&nbsp;Quarter of enrollment (most common) | {nv['quarter']}% | — |",
f"| &nbsp;&nbsp;3 months pre | {nv['m3']}% | — |",
f"| &nbsp;&nbsp;6 months pre | {nv['m6']}% | — |",
f"| &nbsp;&nbsp;12 months pre | {nv['m12']}% | — |",
f"| &nbsp;&nbsp;Using immature (all-2026) claims | {round(100*(1-im[0]))}% | — |",
"| **Bias-corrected (target-trial, mature claims)** | rate ratio | |",
f"| &nbsp;&nbsp;Overlap-weighted Poisson DiD (primary) | {red(pr['overlap'])} | {ci(pr['overlap'])} |",
f"| &nbsp;&nbsp;Inverse-probability-weighted DiD | {rr(pr['iptw'])} | {ci(pr['iptw'])} |",
f"| &nbsp;&nbsp;1:5 propensity-score matched | {rr(pr['matched'])} | {ci(pr['matched'])} |",
"| **By outcome** | | |",
f"| &nbsp;&nbsp;Emergency department visits | {rr(bo['ED'])} | {ci(bo['ED'])} |",
f"| &nbsp;&nbsp;Acute inpatient admissions | {rr(bo['hospitalization'])} | {ci(bo['hospitalization'])} |",
f"| &nbsp;&nbsp;Ambulatory-care-sensitive ED (NYU algorithm) | {rr(bo['ACS_ED_NYU'])} | {ci(bo['ACS_ED_NYU'])} |",
f"| &nbsp;&nbsp;Ambulatory-care-sensitive hospitalization (AHRQ PQI) | {rr(bo['ACS_hosp_PQI'])} | {ci(bo['ACS_hosp_PQI'])} |",
"| **Robustness** | | |",
f"| &nbsp;&nbsp;Parallel-pre-trends falsification (should ≈ 1) | {rr(pr['pretrend'])} | {ci(pr['pretrend'])} |",
f"| &nbsp;&nbsp;E-value (point estimate) | {pr['evalue']} | — |","",
f"Acute care = emergency department visits + acute inpatient admissions per 1000 patient-months among rising-risk–targeted patients. Naive estimates use within-patient before-and-after comparison with no concurrent control; the bias-corrected estimate uses not-yet-activated controls anchored at the activation event, with the activation/trigger month excluded and analysis restricted to claims mature through December 2025. The rate ratio of {pr['overlap'][0]:.2f} corresponds to a {100*(1-pr['overlap'][0]):.0f}% relative reduction."]
open(P/"table2_results.md","w").write("\n".join(T))

# ---------- Figure 1: event study ----------
es={int(k):v for k,v in R["event_study"].items()}
xs=sorted(es); ys=[es[m] for m in xs]
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.axvspan(-1.5,0.5,color="#ffd9d9",alpha=.6,lw=0)
ax.plot(xs,ys,"-o",color="#1f4e79",ms=4,lw=1.8); ax.axvline(0,color="#888",ls="--",lw=1)
sp=es[-1]; ax.annotate(f"Acute-event trigger: {sp:.0f}",xy=(-1,sp),xytext=(2,sp-6),fontsize=8.5,color="#b30000",va="center",arrowprops=dict(arrowstyle="->",color="#b30000",lw=1.2))
pre=R["event_study_summary"]["baseline_-6to-2"]; post=R["event_study_summary"]["post_1to12"]
ax.hlines(pre,-6,-2,color="#2ca02c",lw=1.4,ls=":");ax.text(-6,pre+9,f"pre-trigger baseline ~{pre:.0f}",color="#2ca02c",fontsize=8)
ax.hlines(post,1,12,color="#2ca02c",lw=1.4,ls=":");ax.text(6.5,post-18,f"sustained post ~{post:.0f}",color="#2ca02c",fontsize=8)
ax.set_xlabel("Months since program activation");ax.set_ylabel("ED + hospitalizations /1,000 patient-months");ax.set_xticks(range(-12,13,2));ax.set_ylim(0,max(ys)*1.08)
fig.suptitle("Enrollment is triggered by an acute event (rising-risk cohort, mature claims): source of regression-to-the-mean",fontsize=9,y=1.02,x=0.02,ha="left")
fig.savefig(FIG/"figure1_event_study.png",dpi=300,bbox_inches="tight");plt.close()

# ---------- Figure 2: naive vs corrected ----------
fig,ax=plt.subplots(figsize=(7,3.6))
labs=["Quarter-of-\nenrollment","3 mo pre","6 mo pre","12 mo pre","Corrected\n(target-trial)"]
corr=round(100*(1-pr['overlap'][0])); vals=[nv['quarter'],nv['m3'],nv['m6'],nv['m12'],corr]
cols=["#b30000","#d9692a","#e0a030","#c9b84a","#1f4e79"]
b=ax.bar(labs,vals,color=cols,width=.62)
for bar,v in zip(b,vals): ax.text(bar.get_x()+bar.get_width()/2,v+1,f"{v}%",ha="center",fontsize=9,fontweight="bold")
ax.axhline(corr,color="#1f4e79",ls="--",lw=1);ax.set_ylabel("Apparent reduction in acute care");ax.set_ylim(0,max(vals)*1.15)
ax.set_title("Naive before-and-after estimates inflate the effect (rising-risk cohort, mature claims)",fontsize=9.3,loc="left")
ax.text(1.5,max(vals)*1.05,"naive (biased by trigger spike + regression to the mean)",fontsize=8,color="#b30000")
fig.savefig(FIG/"figure2_naive_vs_corrected.png",dpi=300,bbox_inches="tight");plt.close()

# ---------- Figure 3: forest (outcomes + equity) ----------
eqr=R["equity_race_gender"]; adi=R["equity_adi"]["tertiles"]
fig,(axA,axB)=plt.subplots(2,1,figsize=(7.6,6.8),gridspec_kw={"height_ratios":[5,8]},sharex=True)
def forest(ax,items,title):
    ax.axvspan(0.3,1.0,color="#eef4fa",lw=0);y=np.arange(len(items))[::-1]
    for yi,(lab,v) in zip(y,items):
        ax.plot([v[1],v[2]],[yi,yi],"-",color="#1f4e79",lw=2);ax.plot(v[0],yi,"s",color="#1f4e79",ms=6.5)
        ax.text(v[2]+0.02,yi,f"{v[0]:.2f} ({v[1]:.2f}–{v[2]:.2f})",va="center",fontsize=8.2)
    ax.axvline(1.0,color="#666",ls="--",lw=1.1);ax.set_yticks(y);ax.set_yticklabels([i[0] for i in items],fontsize=8.6);ax.set_ylim(-0.7,len(items)-0.3);ax.set_title(title,fontsize=9.3,loc="left")
forest(axA,[("All-cause ED visits",bo['ED']),("All-cause hospitalization",bo['hospitalization']),
   ("Combined (primary)",pr['overlap']),("Ambulatory-care-sensitive ED (NYU)",bo['ACS_ED_NYU']),
   ("Ambulatory-care-sensitive hosp (PQI)",bo['ACS_hosp_PQI'])],"A  Effect by outcome")
forest(axB,[("Black",eqr['Black or African American']['rr']),("White",eqr['White']['rr']),("Hispanic",eqr['Hispanic']['rr']),
   ("Female",eqr['Female']['rr']),("Male",eqr['Male']['rr']),
   ("ADI low (least deprived)",adi['low']['rr']),("ADI medium",adi['med']['rr']),("ADI high (most deprived)",adi['high']['rr'])],"B  Effect by subgroup (equity)")
axB.set_xlim(0.3,max(2.7,bo['ACS_hosp_PQI'][2]+0.1));axB.set_xlabel("Adjusted rate ratio for avoidable acute care; RR < 1.0 = reduction")
axB.text(1.0,7.85,"No effect",fontsize=7.3,color="#666",ha="center",va="bottom")
fig.savefig(FIG/"figure3_robustness_forest.png",dpi=300,bbox_inches="tight");plt.close()
print("Artifacts regenerated: table1_baseline.md, table2_results.md, figure1/2/3")
print("Table2 primary:",red(pr['overlap']),ci(pr['overlap']),"| ED",rr(bo['ED']),"| ACS-hosp",rr(bo['ACS_hosp_PQI']),ci(bo['ACS_hosp_PQI']))
