"""
Layer: DEMOGRAPHICS for Chicago (tract). Mirrors London's population density + age structure.

The Census ACS API now requires a key, so we derive the same variables from the ACS
counts already present in the CDC/ATSDR SVI 2022 file (E_TOTPOP, E_AGE65, E_AGE17),
combined with TIGER land area (ALAND) for density. No API key needed.
Features: pop_density (per km2), pct_age65, pct_age17.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVI = PROJECT_ROOT / "data/raw/chicago/svi/svi_2022_illinois.csv"
TRACTS = PROJECT_ROOT / "data/raw/chicago/boundaries/cook_tracts_2020.geojson"

print("=" * 70)
print("CHICAGO DEMOGRAPHICS LAYER — density + age (from ACS-via-SVI counts)")
print("=" * 70)

d = pd.read_csv(SVI, dtype={"FIPS": str})[["FIPS", "E_TOTPOP", "E_AGE65", "E_AGE17"]]
d = d.rename(columns={"FIPS": "GEOID"})
for c in ["E_TOTPOP", "E_AGE65", "E_AGE17"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[d["E_TOTPOP"] > 0]

tr = gpd.read_file(TRACTS)[["GEOID", "ALAND"]]
tr["ALAND"] = pd.to_numeric(tr["ALAND"], errors="coerce")
d = d.merge(tr, on="GEOID", how="inner")
d["pop_density"] = d["E_TOTPOP"] / (d["ALAND"] / 1e6)
d["pct_age65"] = d["E_AGE65"] / d["E_TOTPOP"]
d["pct_age17"] = d["E_AGE17"] / d["E_TOTPOP"]
feat = d[["GEOID", "pop_density", "pct_age65", "pct_age17"]].dropna()
print(f"   Demographics tracts: {len(feat):,}")

run_fusion("Demographics", ["pop_density", "pct_age65", "pct_age17"], static_by_geoid=feat)
