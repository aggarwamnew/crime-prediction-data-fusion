"""
Vancouver crime download — VPD GeoDASH Open Data (the City portal no longer hosts it).

Direct download: crimedata_csv_all_years.zip (all years, all neighbourhoods).
Schema: TYPE, YEAR, MONTH, DAY, HOUR, MINUTE, HUNDRED_BLOCK, NEIGHBOURHOOD, X, Y
  - X/Y are UTM Zone 10N (EPSG:32610).
  - Person/violent offences are coordinate-suppressed (X=Y=0); only property crime is
    usable at small-area resolution. Baseline build drops X=0 rows.
Filtered to 2023-2025 to mirror the London/Chicago multi-year monthly setup.
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data/raw/vancouver/crime"
OUT.mkdir(parents=True, exist_ok=True)
ZIP = OUT / "crimedata_csv_all_years.zip"
CSV = OUT / "vancouver_crime_2023_2025.csv"
URL = "https://geodash.vpd.ca/opendata/crimedata_download/crimedata_csv_all_years.zip"
H = {"User-Agent": "Mozilla/5.0 thesis-crime-research"}

print("=" * 70)
print("VANCOUVER CRIME DOWNLOAD (VPD GeoDASH)")
print("=" * 70)

if not ZIP.exists():
    r = requests.get(URL, headers=H, timeout=300)
    r.raise_for_status()
    ZIP.write_bytes(r.content)
    print(f"   downloaded -> {ZIP.stat().st_size/1e6:.1f} MB")

z = zipfile.ZipFile(ZIP)
name = [n for n in z.namelist() if n.endswith(".csv")][0]
df = pd.read_csv(z.open(name))
df.columns = [c.lower() for c in df.columns]
df = df[(df["year"] >= 2023) & (df["year"] <= 2025)].copy()
df.to_csv(CSV, index=False)

print(f"   total 2023-2025 rows: {len(df):,}")
print(f"   types: {df['type'].nunique()}")
has_xy = (df["x"].fillna(0) != 0).sum()
print(f"   geocodable (X/Y != 0): {has_xy:,} ({has_xy/len(df)*100:.1f}%)")
print("\n   type breakdown (geocodable share):")
g = df.assign(geo=(df["x"].fillna(0) != 0))
for t, sub in g.groupby("type"):
    print(f"     {t:48s} n={len(sub):>6,}  geo={sub['geo'].mean()*100:4.0f}%")
