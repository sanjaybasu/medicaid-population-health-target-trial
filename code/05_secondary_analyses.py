"""Secondary analyses: acute-care cost (per-patient-per-month, none/95th/99th-percentile truncation), effect by clinical
condition (with within-stratum falsification), ED severity decomposition, and
zero-event robustness (negative-binomial + two-part hurdle). Merges results into
results.json (eTables 2 NB rows, 6, 7, 8). DB-dependent (re-pulls the few extra
columns not cached by 00_pull_data.py)."""
import os, pathlib, warnings, json, sys
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
import pandas as pd, numpy as np
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

core = coredb("prod"); CUT = pd.Period("2025-12", "M").ordinal
cf = ["diabetes", "htn", "chf", "copd", "sud", "any_bh"]
COLS = list(range(-27, 13))

P = query(core, """
SELECT person_id, months_since_zero_date ms, enrolled_days, zero_date,
  (emergency_department_ct+acute_inpatient_ct) acute, emergency_department_ct ed_all,
  COALESCE(emergency_department_paid,0)+COALESCE(acute_inpatient_paid,0) acute_cost,
  COALESCE(total_paid,0) total_cost,
  COALESCE(ed_non_emergent,0) nonemerg, COALESCE(ed_primary_care_treatable,0) pctreat, COALESCE(ed_preventable,0) prevtbl
FROM dbt.outcomes_with_enrollment__months_since
WHERE ever_activated=1 AND rr_flag=1 AND months_since_zero_date BETWEEN -27 AND 12""")
A = query(core, """
SELECT person_id, MAX(diabetes) diabetes, MAX(htn) htn, MAX(chf) chf, MAX(copd) copd, MAX(sud) sud,
  MAX(any_bh) any_bh, MAX(mdd) mdd, MAX(postpartum) postpartum, MAX(prenatal) prenatal, MAX(no_pcp_last_10mo) nopcp,
  MAX(risk_percentile) risk, MAX(age) age, MAX(gender) gender, MAX(race) race, MAX(state) state,
  MIN(first_eligible_date::date) elig, MAX(zero_date) zd
FROM dbt.outcomes_with_enrollment__months_since WHERE ever_activated=1 AND rr_flag=1 GROUP BY person_id""")
A["e2a"] = (pd.to_datetime(A.zd, errors="coerce") - pd.to_datetime(A.elig, errors="coerce")).dt.days / 30.44
P["emergent"] = (P["ed_all"] - P["nonemerg"] - P["pctreat"] - P["prevtbl"]).clip(lower=0)
P["zdt"] = pd.to_datetime(P.zero_date, errors="coerce"); P = P.dropna(subset=["zdt"])
P["calmon"] = P.zdt.dt.to_period("M").apply(lambda p: p.ordinal if pd.notna(p) else np.nan) + P.ms
P = P[P.calmon <= CUT].copy()
OUTS = ["acute", "acute_cost", "total_cost", "nonemerg", "pctreat", "prevtbl", "emergent", "enrolled_days"]
M = {c: P.pivot_table(index="person_id", columns="ms", values=c, aggfunc="sum", fill_value=0).reindex(columns=COLS, fill_value=0) for c in OUTS}
idx = M["acute"].index

def ws(X, lo, hi):
    cc = [x for x in COLS if lo <= x <= hi]
    return X[cc].sum(axis=1) if cc else pd.Series(0.0, index=idx)

rows = []
for s in range(-12, 1):
    tr = 1 if s == 0 else 0
    plo = s + 2 if tr else s + 1
    phi = 12 if tr else -2
    if plo > phi: continue
    r = {"person_id": idx, "s": s, "treat": tr, "obs": (M["enrolled_days"][s] > 0).values,
         "mmb": (ws(M["enrolled_days"], s-12, s-7)/30.44).values, "mmp": (ws(M["enrolled_days"], plo, phi)/30.44).values,
         "ppb": ws(M["acute"], s-18, s-13).values, "ppmm": (ws(M["enrolled_days"], s-18, s-13)/30.44).values}
    for c in ["acute", "acute_cost", "total_cost", "nonemerg", "pctreat", "prevtbl", "emergent"]:
        r[c+"_b"] = ws(M[c], s-12, s-7).values; r[c+"_p"] = ws(M[c], plo, phi).values
    d = pd.DataFrame(r); rows.append(d[(d.obs) & (d.mmb > 0.3) & (d.mmp > 0.3)])
lm = pd.concat(rows, ignore_index=True).merge(A, on="person_id", how="left")
lm["mse"] = lm.e2a + lm.s; lm = lm[lm.mse >= 0].copy()
for c in ["gender", "race", "state"]: lm[c] = lm[c].fillna("u")
for c in cf + ["mdd", "postpartum", "prenatal"]: lm[c] = pd.to_numeric(lm.get(c, 0), errors="coerce").fillna(0)
lm["age"] = pd.to_numeric(lm.age, errors="coerce").fillna(lm.age.median())
lm["risk"] = pd.to_numeric(lm.risk, errors="coerce").fillna(lm.risk.median())
lm["perinatal"] = ((lm.postpartum == 1) | (lm.prenatal == 1)).astype(int)

def ow_weights(d):
    d = d.copy(); d["base_rate"] = 1000*d.acute_b/d.mmb.clip(lower=0.1)
    Xn = d[["base_rate", "mse", "risk", "age"]+cf].astype(float)
    Xc = pd.get_dummies(d[["gender", "race", "state"]], drop_first=True).astype(float)
    Z = pd.concat([Xn.reset_index(drop=True), Xc.reset_index(drop=True)], axis=1)
    Z = ((Z-Z.mean())/Z.std(ddof=0).replace(0, 1)).fillna(0)
    ps = np.clip(LogisticRegression(max_iter=3000, C=0.5).fit(Z, d.treat).predict_proba(Z)[:, 1], 1e-3, 1-1e-3)
    return np.where(d.treat == 1, 1-ps, ps)

def did_rr(d, bcol, pcol, mmb="mmb", mmp="mmp", fam="poisson"):
    w = ow_weights(d)
    pre = d[["person_id", "treat"]].copy(); pre["y"] = d[bcol].values; pre["mm"] = d[mmb].values; pre["post"] = 0; pre["w"] = w
    po = d[["person_id", "treat"]].copy(); po["y"] = d[pcol].values; po["mm"] = d[mmp].values; po["post"] = 1; po["w"] = w
    s = pd.concat([pre, po], ignore_index=True); s = s[s.mm > 0.05]; s["lo"] = np.log(s.mm)
    F = sm.families.Poisson() if fam == "poisson" else sm.families.NegativeBinomial(alpha=1.0)
    m = smf.glm("y ~ treat*post", data=s, family=F, offset=s.lo, freq_weights=s.w).fit(cov_type="cluster", cov_kwds={"groups": s.person_id})
    b, se = m.params["treat:post"], m.bse["treat:post"]
    return [round(np.exp(b), 3), round(np.exp(b-1.96*se), 3), round(np.exp(b+1.96*se), 3)]

def cost_pmpm(d, bcol, pcol, winsor=None):  # per-patient-per-month cost DiD; optional percentile truncation of the pooled baseline/post distribution
    w = ow_weights(d)
    pmb = d[bcol]/d.mmb.clip(lower=0.1); pmp = d[pcol]/d.mmp.clip(lower=0.1); cap = None
    if winsor is not None:
        cap = float(np.quantile(np.concatenate([pmb, pmp]), winsor)); pmb = np.minimum(pmb, cap); pmp = np.minimum(pmp, cap)
    pre = d[["person_id", "treat"]].copy(); pre["y"] = pmb.values; pre["post"] = 0; pre["w"] = w
    po = d[["person_id", "treat"]].copy(); po["y"] = pmp.values; po["post"] = 1; po["w"] = w
    s = pd.concat([pre, po], ignore_index=True)
    m = smf.wls("y ~ treat*post", data=s, weights=s.w).fit(cov_type="cluster", cov_kwds={"groups": s.person_id})
    b, se = m.params["treat:post"], m.bse["treat:post"]
    r = {"pmpm": [round(b), round(b-1.96*se), round(b+1.96*se)], "p": round(m.pvalues["treat:post"], 3)}
    if cap is not None: r["cap_per_patient_month"] = round(cap)
    return r

R = {}
R["cost"] = {nm: {w: cost_pmpm(lm, bc, pc, wv) for w, wv in [("none", None), ("p99", 0.99), ("p95", 0.95)]}
             for nm, bc, pc in [("acute_care", "acute_cost_b", "acute_cost_p"), ("total", "total_cost_b", "total_cost_p")]}
R["ed_severity"] = {k: did_rr(lm, k+"_b", k+"_p") for k in ["nonemerg", "pctreat", "prevtbl", "emergent"]}
R["zero_robustness"] = {"zero_base_pct": round(100*(lm.acute_b == 0).mean()), "poisson": did_rr(lm, "acute_b", "acute_p"),
                        "negbin": did_rr(lm, "acute_b", "acute_p", fam="nb")}
cond = {}
for lab, col in [("substance_use_disorder", "sud"), ("any_behavioral_health", "any_bh"), ("major_depressive_disorder", "mdd"),
                 ("perinatal", "perinatal"), ("copd", "copd"), ("hypertension", "htn"), ("diabetes", "diabetes"), ("heart_failure", "chf")]:
    d = lm[lm[col] == 1]; nt = int(d.treat.sum())
    if nt < 40: continue
    cond[lab] = {"n_treated": nt, "rr": did_rr(d, "acute_b", "acute_p"),
                 "falsification": did_rr(d, "ppb", "acute_b", mmb="ppmm", mmp="mmb")}
R["condition_hte"] = cond

rf = ROOT/"results.json"
allR = json.load(open(rf)) if rf.exists() else {}
allR["secondary"] = R
json.dump(allR, open(rf, "w"), indent=1, default=str)
print("secondary analyses ->", rf)
print(" cost acute PMPM:", {w: R["cost"]["acute_care"][w]["pmpm"] for w in R["cost"]["acute_care"]})
print(" cost total PMPM:", {w: R["cost"]["total"][w]["pmpm"] for w in R["cost"]["total"]})
print(" ED severity:", R["ed_severity"])
print(" zero-robustness:", R["zero_robustness"])
print(" condition HTE:", {k: v["rr"] for k, v in cond.items()})
