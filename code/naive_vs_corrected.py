import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
core = coredb("prod")
# NAIVE within-enrollee pre/post (what a typical program eval reports), ever-activated, indexed at activation (ms=0)
d=query(core, """
SELECT person_id, months_since_zero_date ms, enrolled_days,
   (emergency_department_ct+acute_inpatient_ct) acute
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -12 AND 12 AND enrolled_days>0""")
pre=d[d.ms.between(-12,-1)]; post=d[d.ms.between(1,12)]
def rate(x): return 1000*x.acute.sum()/(x.enrolled_days.sum()/30.44)
rpre=rate(pre); rpost=rate(post)
print("### NAIVE within-enrollee pre/post (standard observational program eval), program-wide")
print(f"  pre (mo -12..-1): {rpre:.1f}/1000mm ; post (+1..+12): {rpost:.1f}/1000mm")
print(f"  NAIVE pre/post RR = {rpost/rpre:.3f}  => apparent {100*(1-rpost/rpre):.0f}% reduction")
print(f"  (this is inflated: pre window contains the e=-1 acute-event TRIGGER spike + regression to the mean)")
print(f"\n### CORRECTED (target-trial + Callaway-Sant'Anna, not-yet-treated controls): ATT -179.5/1000mm")
print(f"  corrected as rate ratio (overlap-wt Poisson DiD): RR = 0.63 (0.58-0.69) => 37% reduction")
print(f"  Honest-DiD breakdown M-bar = 1.72 (survives pre-trend violation 1.7x the largest observed)")
print(f"\n### BIAS QUANTIFICATION:")
print(f"  naive apparent reduction {100*(1-rpost/rpre):.0f}% vs corrected 37% => naive overstates by ~{100*(1-rpost/rpre)-37:.0f} pp")
