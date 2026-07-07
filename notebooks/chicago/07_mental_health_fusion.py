"""
Layer: MENTAL HEALTH (SAMHI analogue) for Chicago = CDC PLACES, census-tract.

CDC PLACES (data.cdc.gov, cwsq-ngmh) gives model-based small-area health estimates.
Measure MHLTH = "mental health not good for >=14 days among adults". This is a cleaner
direct small-area mental-health measure than the UK SAMHI composite. Static, by tract.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/cdc_places"
RAW.mkdir(parents=True, exist_ok=True)
CSV = RAW / "places_mhlth_il.csv"

print("=" * 70)
print("CHICAGO MENTAL-HEALTH LAYER — CDC PLACES MHLTH (tract)")
print("=" * 70)

if not CSV.exists():
    # crude-prevalence MHLTH at tract level for Illinois, latest release
    url = "https://data.cdc.gov/resource/cwsq-ngmh.csv"
    params = {"measureid": "MHLTH", "stateabbr": "IL",
              "$select": "year,locationname,data_value,data_value_type",
              "$limit": 50000}
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    CSV.write_text(r.text)
    print(f"   downloaded -> {CSV.stat().st_size/1e6:.2f} MB")

p = pd.read_csv(CSV, dtype={"locationname": str})
p = p[p["data_value_type"].str.contains("Crude", na=False)] if "data_value_type" in p.columns else p
latest = p["year"].max()
p = p[p["year"] == latest]
p = p[["locationname", "data_value"]].rename(columns={"locationname": "GEOID", "data_value": "mhlth"})
p = p.dropna()
print(f"   MHLTH tracts (IL, {latest}): {len(p):,}  mean={p['mhlth'].mean():.1f}%")

run_fusion("MentalHealth(MHLTH)", ["mhlth"], static_by_geoid=p)
