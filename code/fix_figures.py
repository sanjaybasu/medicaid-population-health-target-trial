import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"savefig.bbox":"tight"})
OUT=pathlib.Path("/Users/sanjaybasu/waymark-local/notebooks/waymark-engagement-acute-care/figures")
core=coredb("prod")
d=query(core,"""SELECT months_since_zero_date ms, SUM(emergency_department_ct+acute_inpatient_ct) acute,
  SUM(enrolled_days) days FROM dbt.outcomes_with_enrollment__months_since
  WHERE ever_activated=1 AND months_since_zero_date BETWEEN -12 AND 12 AND enrolled_days>0 GROUP BY 1 ORDER BY 1""")
d["rate"]=1000*d.acute/(d.days/30.44)

# FIGURE 1 (fixed: annotation repositioned, title as padded suptitle)
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.axvspan(-1.5,0.5,color="#ffd9d9",alpha=.6,lw=0)
ax.plot(d.ms,d.rate,"-o",color="#1f4e79",ms=4,lw=1.8)
ax.axvline(0,color="#888",ls="--",lw=1)
sp=d.loc[d.ms==-1,"rate"].values[0]
ax.annotate(f"Acute-event trigger\n(Rising-Risk flag fires): {sp:.0f}",
            xy=(-1,sp),xytext=(2.2,sp-8),fontsize=8.5,color="#b30000",va="center",
            arrowprops=dict(arrowstyle="->",color="#b30000",lw=1.2))
pre=d[d.ms.between(-12,-3)].rate.mean(); post=d[d.ms.between(1,12)].rate.mean()
ax.hlines(pre,-12,-3,color="#2ca02c",lw=1.4,ls=":"); ax.text(-11.5,pre+8,f"pre-baseline ~{pre:.0f}",color="#2ca02c",fontsize=8)
ax.hlines(post,1,12,color="#2ca02c",lw=1.4,ls=":"); ax.text(6.5,post-20,f"sustained post ~{post:.0f}",color="#2ca02c",fontsize=8)
ax.text(-1.3,40,"excluded\nwindow",fontsize=7,color="#b30000",ha="center")
ax.set_xlabel("Months since program activation"); ax.set_ylabel("ED + hospitalizations /1,000 member-months")
ax.set_xticks(range(-12,13,2)); ax.set_ylim(0,470)
fig.suptitle("Enrollment is triggered by an acute crisis: the source of regression-to-the-mean bias",
             fontsize=10,y=1.02,x=0.02,ha="left")
fig.savefig(OUT/"figure1_event_study.png"); plt.close()

# FIGURE 3: robustness forest
fig,ax=plt.subplots(figsize=(7,3.4))
items=[("Overlap-weighted DiD (primary)",0.63,0.58,0.69),
       ("Inverse-probability-weighted DiD",0.63,0.58,0.69),
       ("1:5 propensity-score matched",0.60,0.55,0.65)]
y=np.arange(len(items))[::-1]
for yi,(lab,rr,lo,hi) in zip(y,items):
    ax.plot([lo,hi],[yi,yi],"-",color="#1f4e79",lw=2)
    ax.plot(rr,yi,"s",color="#1f4e79",ms=7)
    ax.text(hi+0.005,yi,f"{rr:.2f} ({lo:.2f}–{hi:.2f})",va="center",fontsize=8.5)
ax.axvline(1.0,color="#999",ls="--",lw=1); ax.text(1.0,len(items)-0.35,"no effect",fontsize=7.5,color="#999",ha="center")
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items],fontsize=9)
ax.set_xlim(0.5,0.85); ax.set_xlabel("Adjusted rate ratio for ED + hospitalization (activation vs not-yet)")
ax.set_title("Corrected effect is consistent across estimators",fontsize=10,loc="left")
ax.text(0.51,-0.9,"Falsification (parallel pre-trends) RR 0.97 (0.89–1.06) · E-value 2.54 · Honest-DiD M̄ 1.72 · cost −$454 PMPM",
        fontsize=7.3,color="#444")
ax.set_ylim(-1.3,len(items)-0.2)
fig.savefig(OUT/"figure3_robustness_forest.png"); plt.close()
print("Saved figure1 (fixed) + figure3. Files:",[p.name for p in sorted(OUT.glob('*.png'))])
