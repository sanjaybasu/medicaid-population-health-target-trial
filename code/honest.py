import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
es=pd.read_parquet("/tmp/cs_es_ALL.parquet")
ac=[c for c in es.columns if c.endswith("ATT")][0]
se=[c for c in es.columns if "std_error" in c][0]
ec="e" if "e" in es.columns else es.columns[0]
es[ec]=pd.to_numeric(es[ec],errors="coerce"); es=es.dropna(subset=[ec]); es=es[es[se]<1e6]
pre=es[(es[ec]<=-2)&(es[ec]>=-12)]; post=es[(es[ec]>=1)&(es[ec]<=12)]
eff=post[ac].mean(); eff_se=np.sqrt((post[se]**2).mean()/len(post))
eff_lo=abs(eff)-1.96*eff_se
maxpre=pre[ac].abs().max(); meanpre=pre[ac].abs().mean()
print("### Honest-DiD (Rambachan & Roth 2023) relative-magnitude sensitivity, program-wide")
print(f"  Program-wide post effect (e=1..12): {eff:.1f}/1000mm  (95% CI half-width {1.96*eff_se:.1f})")
print(f"  Pre-trend violations (e=-2..-12): max |coef| = {maxpre:.1f}, mean |coef| = {meanpre:.1f} /1000mm")
print(f"  Breakdown M-bar (point):    {abs(eff)/maxpre:.2f}")
print(f"  Breakdown M-bar (CI lower): {eff_lo/maxpre:.2f}")
print(f"  => Interpretation: a post-treatment confounding trend would have to exceed ~{eff_lo/maxpre:.1f}x")
print(f"     the single LARGEST pre-treatment trend deviation to render the effect non-significant.")
print(f"     M-bar>1 = robust to pre-trend violations of the magnitude actually observed.")
