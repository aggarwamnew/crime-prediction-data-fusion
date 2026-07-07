"""
Layer: DEPRIVATION (IMD analogue) for Chicago = CDC/ATSDR Social Vulnerability Index.

ADI (the closest IMD analogue) requires an interactive login to download, so we use
the CDC/ATSDR SVI 2022 (census-tract, free, no auth) as the deprivation layer.
Features: overall percentile (RPL_THEMES) + 4 theme rankings (RPL_THEME1-4),
analogous to London's IMD overall score + domain scores. Static, joined by tract GEOID.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/svi"
RAW.mkdir(parents=True, exist_ok=True)
CSV = RAW / "svi_2022_illinois.csv"

CANDIDATES = [
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/Illinois.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_IL.csv",
]

print("=" * 70)
print("CHICAGO DEPRIVATION LAYER — CDC/ATSDR SVI 2022 (tract)")
print("=" * 70)

if not CSV.exists():
    for url in CANDIDATES:
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.text) > 1000:
                CSV.write_text(r.text)
                print(f"   downloaded {url} -> {CSV.stat().st_size/1e6:.1f} MB")
                break
            print(f"   {url} -> HTTP {r.status_code}")
        except Exception as e:
            print(f"   {url} -> {e}")
    else:
        raise SystemExit("Could not download SVI Illinois CSV from known URLs.")

svi = pd.read_csv(CSV, dtype={"FIPS": str})
theme_cols = ["RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
svi = svi[["FIPS"] + theme_cols].rename(columns={"FIPS": "GEOID"})
# SVI uses -999 for missing -> NaN
svi[theme_cols] = svi[theme_cols].where(svi[theme_cols] >= 0)
svi = svi.dropna(subset=["RPL_THEMES"])
print(f"   SVI tracts (IL): {len(svi):,}  | feature cols: {theme_cols}")

run_fusion("SVI", theme_cols, static_by_geoid=svi)
