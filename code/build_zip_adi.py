#!/usr/bin/env python3
"""
Build a 5-digit ZIP (ZCTA5) -> Area Deprivation Index (ADI) crosswalk.

DATA SOURCES (both freely / publicly fetchable, no registration):
  1. Block-group ADI: University of Wisconsin Neighborhood Atlas ADI ranks,
     bundled (as a feather file) in the open-source `geocode-adi` package.
       https://raw.githubusercontent.com/AyushDoshi/geocode-adi/main/geocode-adi/resources/BlockGroupToADI.feather
     Columns: FIPS (12-digit census block group), and National/State ADI rank
     for 2015, 2020, 2021. Non-numeric values are UW suppression codes:
       GQ = >33% group-quarters pop, PH = predominantly public housing,
       QDI = questionable data, (blank/None) = no data.
     ADI national rank = percentile 1-100 (higher = more deprived).
     State rank = decile 1-10.
  2. ZCTA5 <-> census tract relationship: US Census Bureau 2020 relationship file.
       https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt
     Pipe-delimited. Key cols: GEOID_ZCTA5_20 (5-digit ZCTA), GEOID_TRACT_20
     (11-digit tract), AREALAND_PART (land area m^2 of the ZCTA x tract
     intersection -> used as the areal weight).

JOIN LOGIC:
  block group (12-digit FIPS) --[first 11 digits]--> census tract (11-digit)
  Aggregate block-group ADI to tract: unweighted mean of numeric block-group
  ranks within the tract (block groups within a tract are comparable in size).
  Then tract --[Census ZCTA5xTRACT relationship]--> ZCTA5, aggregating tract
  ADI to ZCTA5 as an AREALAND_PART-area-weighted mean.

OUTPUT: /tmp/zip_adi.csv  columns: zip, adi_natrank, adi_staterank
  (5-digit ZCTA5 = the standard 5-digit-ZIP proxy. The Neighborhood Atlas does
   not publish a 5-digit ZIP file by design; ZCTA5 is the accepted public
   equivalent.)
"""
import os, pathlib
_C=pathlib.Path(os.environ.get("ARTIFACTS_DIR", pathlib.Path(__file__).resolve().parents[1]))/"cache"; _C.mkdir(parents=True,exist_ok=True)

import sys
import pandas as pd
import numpy as np

ADI_FEATHER = str(_C/"BlockGroupToADI.feather")
ZCTA_TRACT  = str(_C/"zcta_tract.txt")
OUT_CSV     = str(_C/"zip_adi.csv")

# year of ADI to use (2020 = ACS 2016-2020, 2020 block groups)
NAT_COL   = "2020 National ADI Rank"
STATE_COL = "2020 State ADI Rank"

def to_num(s):
    """Coerce ADI rank to numeric; suppression codes (GQ/PH/QDI/blank) -> NaN."""
    return pd.to_numeric(s, errors="coerce")

def main():
    # --- 1. block-group ADI -> tract ---
    bg = pd.read_feather(ADI_FEATHER)
    bg = bg[["FIPS", NAT_COL, STATE_COL]].copy()
    bg["FIPS"] = bg["FIPS"].astype(str).str.zfill(12)
    bg["tract"] = bg["FIPS"].str[:11]
    bg["nat"]   = to_num(bg[NAT_COL])
    bg["state"] = to_num(bg[STATE_COL])
    bg = bg.dropna(subset=["nat"])              # drop suppressed block groups
    tract = (bg.groupby("tract")
               .agg(nat=("nat", "mean"), state=("state", "mean"))
               .reset_index())

    # --- 2. ZCTA5 <-> tract relationship ---
    cw = pd.read_csv(ZCTA_TRACT, sep="|", dtype=str,
                     usecols=["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_PART"])
    cw = cw[cw["GEOID_ZCTA5_20"].notna() & (cw["GEOID_ZCTA5_20"].str.strip() != "")]
    cw["zip"]   = cw["GEOID_ZCTA5_20"].str.zfill(5)
    cw["tract"] = cw["GEOID_TRACT_20"].str.zfill(11)
    cw["w"]     = pd.to_numeric(cw["AREALAND_PART"], errors="coerce").fillna(0.0)

    # --- 3. join + area-weighted aggregate to ZCTA5 ---
    m = cw.merge(tract, on="tract", how="inner")
    # if a ZCTA has only zero-area (water) overlaps with ADI tracts, fall back
    # to unweighted mean by giving every matched tract weight 1.
    def wmean(g, col):
        w = g["w"].to_numpy(dtype=float)
        x = g[col].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(x)
        return float(np.average(x, weights=w))

    rows = []
    for z, g in m.groupby("zip"):
        rows.append({
            "zip": z,
            "adi_natrank": round(wmean(g, "nat"), 1),
            "adi_staterank": round(wmean(g, "state"), 1),
        })
    out = pd.DataFrame(rows).sort_values("zip")
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}: {len(out)} ZCTA5 rows")
    return out

if __name__ == "__main__":
    main()
