"""Care-gap characterization at rising-risk identification (eTable 9; eFigure 1).
Operationalizes a safety-net care-cascade model in claims + pharmacy fills + diagnoses,
among the canonical rising-risk activated cohort (rr_flag=1, mature). Merges results
into results.json under "care_gaps". Requires cache/{attrs,panel}.parquet from
00_pull_data.py; pulls pharmacy_claim + condition itself. DB-dependent."""
import os, pathlib, warnings, json, sys
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
CACHE = ROOT/"cache"; OUT = ROOT/"outputs"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pathlib.Path.home()/".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np

core = coredb("prod"); CUT = pd.Period("2025-12", "M").ordinal
attrs = pd.read_parquet(str(CACHE/"attrs.parquet")); attrs["zero_date"] = pd.to_datetime(attrs.zero_date)
attrs["zmon"] = attrs.zero_date.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan)
attrs["age"] = pd.to_numeric(attrs.age, errors="coerce")
C = attrs[(attrs.rr == 1) & (attrs.zmon <= CUT)].copy(); N = C.person_id.nunique()
ids = tuple(C.person_id.tolist())

PH = query(core, """SELECT person_id, lower(ndc_description) nd, dispensing_date dd
 FROM dbt_tuva_core.pharmacy_claim WHERE ndc_description IS NOT NULL AND person_id IN %(ids)s""", ids=ids)
PH = PH.merge(C[["person_id", "zero_date"]], on="person_id", how="inner")
PH["dd"] = pd.to_datetime(PH.dd); PH["days_pre"] = (PH.zero_date - PH.dd).dt.days
CD = query(core, """SELECT person_id pid, normalized_code code, recorded_date rd FROM dbt_tuva_core.condition
 WHERE normalized_code_type='icd-10-cm' AND person_id IN %(ids)s
 AND (normalized_code ~ '^N18' OR normalized_code ~ '^E1[0-4]64' OR normalized_code ~ '^E16' OR normalized_code ~ '^W0'
   OR normalized_code ~ '^W1' OR normalized_code ~ '^R296' OR normalized_code ~ '^Z9181' OR normalized_code ~ '^F11'
   OR normalized_code ~ '^I2[0-5]' OR normalized_code ~ '^I63' OR normalized_code ~ '^I7[0-4]')""", ids=ids)
CD = CD.merge(C[["person_id", "zero_date"]], left_on="pid", right_on="person_id", how="inner")
CD = CD[pd.to_datetime(CD.rd) <= CD.zero_date]
m = lambda p: set(CD[CD.code.str.match(p)].pid)
ckd, hypo, falls = m('N18'), m('E1[0-4]64|E16'), m('W0|W1|R296|Z9181'); oud, ascvd = m('F11'), m('I2[0-5]|I63|I7[0-4]')

CL = {'acei': 'lisinopril|enalapril|ramipril|benazepril|captopril|quinapril|perindopril|fosinopril|trandolapril|moexipril',
'arb': 'losartan|valsartan|irbesartan|candesartan|olmesartan|telmisartan|azilsartan|eprosartan', 'arni': 'sacubitril',
'bb': 'carvedilol|metoprolol|bisoprolol|nebivolol|atenolol|labetalol|propranolol|nadolol|sotalol|betaxolol|pindolol',
'mra': 'spironolactone|eplerenone', 'sglt2': 'gliflozin',
'ccb': 'amlodipine|nifedipine|diltiazem|verapamil|felodipine|nicardipine|isradipine|nisoldipine',
'diur': 'hydrochlorothiazide|chlorthalidone|furosemide|torsemide|bumetanide|indapamide|metolazone|triamterene|hctz',
'insulin': 'insulin|glargine|aspart|lispro|detemir|degludec|isophane|novolog|humalog|lantus|levemir|tresiba|humulin|novolin',
'sulf': 'glipizide|glyburide|glimepiride|glibenclamide', 'metf': 'metformin', 'dpp4': 'gliptin',
'glp1': 'glutide|exenatide|ozempic|trulicity|victoza|mounjaro|rybelsus|wegovy', 'tzd': 'pioglitazone|rosiglitazone',
'ics': 'fluticasone|budesonide|mometasone|beclomethasone|ciclesonide|flunisolide',
'laba': 'salmeterol|formoterol|vilanterol|arformoterol|olodaterol|indacaterol',
'lama': 'tiotropium|umeclidinium|aclidinium|glycopyrrolate|revefenacin',
'anticoag': 'warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|enoxaparin|eliquis|xarelto|pradaxa|coumadin|jantoven',
'statin': 'atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|pitavastatin|fluvastatin',
'antidep': 'fluoxetine|sertraline|paroxetine|citalopram|escitalopram|fluvoxamine|venlafaxine|desvenlafaxine|duloxetine|bupropion|mirtazapine|trazodone|amitriptyline|nortriptyline|imipramine|desipramine|doxepin|vortioxetine|vilazodone',
'moud': 'buprenorphine|suboxone|subutex|sublocade|zubsolv|naltrexone|vivitrol',
'opioid': 'morphine|oxycodone|hydrocodone|hydromorphone|oxymorphone|fentanyl|codeine|tramadol|tapentadol|meperidine',
'benzo': 'alprazolam|lorazepam|clonazepam|diazepam|temazepam|chlordiazepoxide|clorazepate|oxazepam|triazolam|estazolam'}
has = lambda df, p: set(df[df.nd.str.contains(p, regex=True, na=False)].person_id)
pre = PH[PH.days_pre.between(0, 180)]; on = {k: has(pre, v) for k, v in CL.items()}
obs = set(PH[PH.days_pre.between(0, 365)].person_id)
anti = on['acei'] | on['arb'] | on['arni'] | on['bb'] | on['mra'] | on['ccb'] | on['diur']
adm = on['insulin'] | on['sulf'] | on['metf'] | on['dpp4'] | on['glp1'] | on['tzd'] | on['sglt2']
gdmt = on['acei'] | on['arb'] | on['arni'] | on['bb'] | on['mra'] | on['sglt2']; ctrl = on['ics'] | on['laba'] | on['lama']
allid = set(C.person_id); S = lambda c: set(C[C[c] == 1].person_id)
age40dm = set(C[(C.diabetes == 1) & (C.age >= 40)].person_id)
R = {}
def gap(den, g, key, use_obs=True):
    d = (den & obs) if use_obs else den; k = len(d & g); n = len(d)
    R[key] = {"n": k, "den": n, "pct": round(100 * k / n, 1) if n else None}

gap(allid, set(C[C.nopcp == 1].person_id), "no_pcp", use_obs=False)
gap(S('htn'), allid - anti, "htn_norx")
gap(S('htn') & (S('diabetes') | ckd), allid - (on['acei'] | on['arb'] | on['arni']), "htn_cardiorenal")
gap(S('diabetes'), allid - adm, "dm_norx")
gap(ascvd | age40dm, allid - on['statin'], "statin")
gap(S('chf'), allid - gdmt, "hf_nogdmt")
gap(S('asthma') | S('copd'), allid - ctrl, "airway_noctrl")
gap(S('mdd'), allid - on['antidep'], "mdd_noad")
gap(oud, allid - on['moud'], "oud_nomoud")
gap(S('diabetes') & (on['insulin'] | on['sulf']), hypo, "insulin_hypo")
gap(on['anticoag'], falls, "ac_falls")
gap(on['opioid'], on['benzo'], "opioid_benzo")
gap(allid, set(C[C.polypharmacy == 1].person_id), "poly", use_obs=False)
R["sdoh_claims"] = {"n": 0, "den": int(N), "pct": 0.0}

# care transitions: pre-zero hospitalization (-12..-2) not followed by PCP within ~30d (same/next month)
panel = pd.read_parquet(str(CACHE/"panel.parquet"))
panel = panel[panel.person_id.isin(allid)].sort_values(["person_id", "ms"])
panel["pcp_next"] = panel.groupby("person_id").pcp.shift(-1)
hz = panel[(panel.ms.between(-12, -2)) & (panel.ip > 0)].copy()
hz["followed"] = ((hz.pcp > 0) | (hz.pcp_next > 0)).astype(int)
pat = hz.groupby("person_id").followed.apply(lambda s: (s == 0).any())
R["transitions_nofu"] = {"n": int(pat.sum()), "den": int(len(pat)), "pct": round(100 * pat.mean(), 1) if len(pat) else None}
R["_meta"] = {"N": int(N), "observable_rx": len(obs), "dx": {k: len(v) for k, v in
              [("ckd", ckd), ("hypoglycemia", hypo), ("falls", falls), ("oud", oud), ("ascvd", ascvd)]}}

rf = OUT/"results.json"; allR = json.load(open(rf)) if rf.exists() else {}
allR["care_gaps"] = R; json.dump(allR, open(rf, "w"), indent=1, default=str)
print("care gaps -> N =", N, "| observable Rx =", len(obs))
for k in ["no_pcp","airway_noctrl","statin","htn_cardiorenal","hf_nogdmt","oud_nomoud","mdd_noad","insulin_hypo","ac_falls","opioid_benzo","poly","transitions_nofu","sdoh_claims"]:
    print(f"  {k:18s} {R[k]['n']:5d}/{R[k]['den']:5d}  {R[k]['pct']}%")
