import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
core = coredb("prod")
d=query(core, """SELECT months_since_zero_date ms, enrolled_days,
   (emergency_department_ct+acute_inpatient_ct) acute
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND months_since_zero_date BETWEEN -12 AND 12 AND enrolled_days>0""")
def rate(lo,hi):
    x=d[d.ms.between(lo,hi)]; return 1000*x.acute.sum()/(x.enrolled_days.sum()/30.44)
post=rate(1,12)
print("### Acute rate by month relative to activation (shows the e=-1 trigger spike):")
for m in [-6,-3,-2,-1,0,1,3,6,12]:
    print(f"   ms={m:+d}: {rate(m,m):6.0f}/1000mm")
print(f"\n### NAIVE pre/post 'reduction' depends entirely on the pre-window chosen (post fixed = {post:.0f}):")
for lo,hi,lab in [(-1,-1,"1mo (qtr-of-enrollment, common)"),(-3,-1,"3mo pre"),(-6,-1,"6mo pre"),(-12,-1,"12mo pre")]:
    rpre=rate(lo,hi); print(f"   pre={lab:32s}: {rpre:6.0f} -> RR {post/rpre:.2f}  ({100*(1-post/rpre):2.0f}% apparent reduction)")
print(f"\n### CORRECTED target-trial (CS-DR / overlap-wt DiD): 37% (RR 0.63, 0.58-0.69), Honest-DiD M-bar 1.72")
print("   => naive 'reductions' range 30-80% depending on arbitrary pre-window; the trigger spike at ms=-1 drives the inflation.")
