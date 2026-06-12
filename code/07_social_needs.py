"""Descriptive characterization of documented health-related social needs at intake
(eTable 9 social section). Social needs are not coded in claims (0 SDOH Z-codes) and the
care platform's structured social-determinants field is unpopulated, so this mines the
free-text "meet the patient" (MEET_THE_PATIENT) intake notes in Lighthouse via a
transparent keyword dictionary (PRAPARE / Accountable Health Communities domains).
EXPLICIT LOWER BOUND: free-text documentation is variable, negation is not parsed, many
intake contacts are administrative, and only ~36% of the cohort has an intake note.
DB-dependent (Lighthouse). Code only; note text (PHI) is processed in aggregate and never
written out. Requires cache/attrs.parquet from 00_pull_data.py for cohort IDs."""
import os, pathlib, warnings, json, re, sys
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT/"cache"; OUT = ROOT/"outputs"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pathlib.Path.home()/".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, query
import pandas as pd

CUT = pd.Period("2025-12", "M").ordinal
attrs = pd.read_parquet(str(CACHE/"attrs.parquet")); attrs["zero_date"] = pd.to_datetime(attrs.zero_date)
attrs["zmon"] = attrs.zero_date.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else None)
ids = tuple(attrs[(attrs.rr == 1) & (attrs.zmon <= CUT)].person_id.tolist()); N = len(ids)
lh = lighthouse("prod")
# Lighthouse Patient.waymarkPatientNumber == coredb person_id; EncounterNote.patientId == Patient.id
df = query(lh, '''SELECT p."waymarkPatientNumber" pid, e.note FROM public."Patient" p
  JOIN public."EncounterNote" e ON e."patientId"=p.id
  WHERE p."waymarkPatientNumber" IN %(ids)s AND e."encounterType"='MEET_THE_PATIENT'
    AND e.deleted IS NOT TRUE AND e.note IS NOT NULL''', ids=ids)

def to_text(s):  # Lighthouse notes are Draft.js JSON ({"blocks":[{"text":...}]}); fall back to HTML strip
    try:
        return " ".join(b.get("text", "") for b in json.loads(s).get("blocks", []))
    except Exception:
        return re.sub(r"<[^>]+>", " ", str(s))

df["txt"] = df.note.map(to_text)
pt = df.groupby("pid").txt.apply(lambda s: " ".join(s).lower())
n_intake = int(pt.shape[0])

DICT = {  # problem-indicating terms per HRSN domain (transparent; imperfect — see caveats above)
 "food_insecurity": r"food insecur|food bank|food pantr|\bno food\b|going hungry|can'?t afford food|\bsnap\b|\bebt\b|\bwic\b|food stamp|meal delivery",
 "housing_instability": r"homeless|evict|shelter|couch.?surf|unstabl\w* hous|housing insecur|behind on rent|sleeping in (her|his|their|the)? ?car|living in (her|his|their|a)? ?car|no(where| place) to (live|stay)|\bmotel\b|transitional housing|unhoused|section 8",
 "utilities": r"utilit|electric\w* (bill|shut)|gas (bill|shut)|water (bill|shut)|shut.?off|liheap|\bno heat\b|heat (is )?(off|out)|behind on (utilit|electric)",
 "interpersonal_safety": r"domestic violence|intimate partner|\bipv\b|\bdv\b|\babuse\b|unsafe (at home|relationship|situation)|feels unsafe|safety concern|fleeing",
 "employment": r"unemploy|lost (her|his|their)? ?job|job loss|out of work|looking for work|can'?t work|unable to work",
 "financial_strain": r"financ\w* (strain|hardship|insecur|stress|difficult)|can'?t afford|cannot afford|no income|low.?income|struggl\w* financ|money problem|in debt|\bpoverty\b|unable to afford",
 "transportation": r"no (transportation|car|ride|vehicle)|lacks? transportation|transportation (barrier|issue|need)|needs? a ride|bus pass|can'?t get to (her|his|the)? ?(appoint|doctor|pharmacy)",
 "legal_immigration": r"immigrat|undocument|legal (aid|issue|status|help)|incarcerat|court date|on probation",
 "language_literacy": r"interpreter|language barrier|limited english|does not speak english|low literacy|health literacy",
 "social_isolation": r"social\w* isolat|\bisolated\b|lonel\w*|no support system|lives alone|no (family|one) (support|nearby|to help)|lacks support",
 "insurance_benefits": r"lapse in (coverage|insurance|medicaid)|uninsured|lost (medicaid|coverage|insurance)|benefits (issue|lapse)|eligibilit\w* (issue|lapse|problem)|recertif|redetermin",
}
flags = pd.DataFrame({d: pt.str.contains(p, regex=True, na=False) for d, p in DICT.items()})
R = {"n_cohort": N, "n_intake": n_intake, "pct_intake": round(100 * n_intake / N),
     "any": {"n": int(flags.any(axis=1).sum()), "den": n_intake, "pct": round(100 * flags.any(axis=1).mean(), 1)},
     "domains": {d: {"n": int(flags[d].sum()), "den": n_intake, "pct": round(100 * flags[d].mean(), 1)} for d in DICT}}
rf = OUT/"results.json"; allR = json.load(open(rf)) if rf.exists() else {}
allR["social_needs"] = R; json.dump(allR, open(rf, "w"), indent=1, default=str)
print(f"social needs -> cohort N={N}; intake notes {n_intake} ({R['pct_intake']}%); any documented {R['any']['pct']}%")
for d in sorted(DICT, key=lambda k: -R["domains"][k]["pct"]):
    print(f"  {d:20s} {R['domains'][d]['n']:4d}/{n_intake}  {R['domains'][d]['pct']}%")
