import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})
OUT=pathlib.Path("/Users/sanjaybasu/waymark-local/notebooks/waymark-engagement-acute-care/figures"); OUT.mkdir(exist_ok=True)
core=coredb("prod")
# by-month raw acute rate (descriptive event study)
d=query(core,"""SELECT months_since_zero_date ms, SUM(emergency_department_ct+acute_inpatient_ct) acute,
  SUM(enrolled_days) days FROM dbt.outcomes_with_enrollment__months_since
  WHERE ever_activated=1 AND months_since_zero_date BETWEEN -12 AND 12 AND enrolled_days>0 GROUP BY 1 ORDER BY 1""")
d["rate"]=1000*d.acute/(d.days/30.44)

# FIGURE 1: event study (the bias mechanism)
fig,ax=plt.subplots(figsize=(7,4.2))
ax.axvspan(-1.5,0.5,color="#ffd9d9",alpha=.6,lw=0,label="Activation/trigger window (excluded)")
ax.plot(d.ms,d.rate,"-o",color="#1f4e79",ms=4,lw=1.8)
ax.axvline(0,color="#888",ls="--",lw=1)
sp=d.loc[d.ms==-1,"rate"].values[0]
ax.annotate(f"Acute-event trigger\n(Rising-Risk flag fires)\n{sp:.0f}/1,000 mm",
            xy=(-1,sp),xytext=(-9,sp+30),fontsize=8.5,color="#b30000",
            arrowprops=dict(arrowstyle="->",color="#b30000"))
pre=d[d.ms.between(-12,-3)].rate.mean(); post=d[d.ms.between(1,12)].rate.mean()
ax.hlines(pre,-12,-3,color="#2ca02c",lw=1.5,ls=":"); ax.text(-11.5,pre-22,f"pre-baseline ~{pre:.0f}",color="#2ca02c",fontsize=8)
ax.hlines(post,1,12,color="#2ca02c",lw=1.5,ls=":"); ax.text(6,post-22,f"sustained post ~{post:.0f}",color="#2ca02c",fontsize=8)
ax.set_xlabel("Months since program activation"); ax.set_ylabel("ED + hospitalization /1,000 member-months")
ax.set_title("Enrollment is triggered by an acute crisis: the source of regression-to-the-mean bias",fontsize=9.5,loc="left")
ax.legend(fontsize=7.5,loc="upper right"); ax.set_xticks(range(-12,13,2))
fig.savefig(OUT/"figure1_event_study.png"); plt.close()

# FIGURE 2: naive-vs-corrected
fig,ax=plt.subplots(figsize=(7,3.6))
labels=["Quarter-of-\nenrollment","3 mo pre","6 mo pre","12 mo pre","Corrected\n(target-trial)"]
red=[64,52,46,41,37]; cols=["#b30000","#d9692a","#e0a030","#c9b84a","#1f4e79"]
bars=ax.bar(labels,red,color=cols,width=.62)
for b,v in zip(bars,red): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v}%",ha="center",fontsize=9,fontweight="bold")
ax.axhline(37,color="#1f4e79",ls="--",lw=1)
ax.set_ylabel("Apparent reduction in acute care"); ax.set_ylim(0,72)
ax.set_title("Naive pre/post estimates inflate the effect up to ~2×, by arbitrary pre-window choice",fontsize=9.5,loc="left")
ax.text(1.5,67,"naive (biased by trigger spike + regression to the mean)",fontsize=8,color="#b30000")
fig.savefig(OUT/"figure2_naive_vs_corrected.png"); plt.close()
print("Saved:",[p.name for p in OUT.glob('*.png')])
