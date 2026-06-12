import json, pathlib, os
ROOT=pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))
R=json.load(open(ROOT/"results.json"))
es={int(k):v for k,v in R["event_study"].items()}
assert es[-1]==max(es.values()), "event study must peak at the -1 trigger month"
assert 0.5 < R["primary"]["overlap"][0] < 1.0, "primary RR out of expected range"
assert R["primary"]["pretrend"][1] <= 1.0 <= R["primary"]["pretrend"][2], "pre-trend CI must include 1 (null falsification)"
assert abs(R["validation_all_pathways"][0]-0.79) < 0.03, "ALL-pathway validation should reproduce locked v3 (~0.79)"
print("All invariants hold: trigger peak, RR<1, null pre-trends, validation reproduces 0.79.")
