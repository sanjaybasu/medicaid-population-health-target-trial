import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"savefig.bbox":"tight"})
OUT="/Users/sanjaybasu/waymark-local/notebooks/waymark-engagement-acute-care/figures/"
fig,ax=plt.subplots(figsize=(7.2,3.6))
items=[("Overlap-weighted DiD (primary)",0.63,0.58,0.69),
       ("Inverse-probability-weighted DiD",0.63,0.58,0.69),
       ("1:5 propensity-score matched",0.60,0.55,0.65)]
y=np.arange(len(items))[::-1]
# shade the reduction region (RR<1) lightly
ax.axvspan(0.45,1.0,color="#eef4fa",lw=0)
for yi,(lab,rr,lo,hi) in zip(y,items):
    ax.plot([lo,hi],[yi,yi],"-",color="#1f4e79",lw=2)
    ax.plot(rr,yi,"s",color="#1f4e79",ms=7)
    ax.text(hi+0.01,yi,f"{rr:.2f} ({lo:.2f}–{hi:.2f})",va="center",fontsize=8.5)
ax.axvline(1.0,color="#666",ls="--",lw=1.2)
ax.text(1.0,len(items)-0.30,"No effect\n(RR = 1.0)",fontsize=8,color="#666",ha="center",va="bottom")
# direction cue
ax.annotate("fewer acute-care events",xy=(0.6,-0.85),xytext=(0.92,-0.85),fontsize=8,color="#1f4e79",va="center",ha="right",
            arrowprops=dict(arrowstyle="->",color="#1f4e79",lw=1.2))
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items],fontsize=9)
ax.set_xlim(0.45,1.12); ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0,1.1])
ax.set_xlabel("Adjusted rate ratio for ED + hospitalization (activation vs not-yet)")
ax.set_title("Corrected effect is consistent across estimators",fontsize=9.5,loc="left")
ax.text(0.455,-1.35,"Falsification (parallel pre-trends) RR 0.97 (0.89–1.06) · E-value 2.5 · Honest-DiD M̄ 1.7 · cost −$454 PMPM",fontsize=7.2,color="#444")
ax.set_ylim(-1.7,len(items)-0.1)
fig.savefig(OUT+"figure3_robustness_forest.png"); plt.close()
print("regenerated figure3")
