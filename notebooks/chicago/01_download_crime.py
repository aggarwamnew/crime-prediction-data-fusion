"""
Chicago crime download — replication of the London baseline for cross-city validation.

Source: City of Chicago Data Portal, "Crimes - 2001 to Present" (Socrata id ijzp-q8t2).
Window: Jan 2023 -> Jan 2026 (37 months), matching the London thesis window.
Pulls only the columns needed for the baseline (date, type, lat/long) via the SODA API,
paginated. Saves a single CSV to data/raw/chicago/crime/.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/raw/chicago/crime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
# Match London: Jan 2023 through Jan 2026 inclusive.
WHERE = "date >= '2023-01-01T00:00:00' AND date < '2026-02-01T00:00:00'"
SELECT = "id,date,primary_type,latitude,longitude,community_area,year"
PAGE = 50000

print("=" * 70)
print("CHICAGO CRIME DOWNLOAD (ijzp-q8t2)")
print("=" * 70)
print(f"Window filter: {WHERE}")

frames = []
offset = 0
while True:
    params = {
        "$select": SELECT,
        "$where": WHERE,
        "$order": "id",          # stable order for pagination
        "$limit": PAGE,
        "$offset": offset,
    }
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=120)
            r.raise_for_status()
            break
        except Exception as e:
            print(f"   retry {attempt+1} (offset {offset}): {e}")
            time.sleep(5)
    else:
        print("   FAILED after retries; aborting.")
        sys.exit(1)

    from io import StringIO
    chunk = pd.read_csv(StringIO(r.text))
    if len(chunk) == 0:
        break
    frames.append(chunk)
    print(f"   fetched {len(chunk):>6,} rows (offset {offset:,}) -> total {sum(len(f) for f in frames):,}")
    offset += PAGE
    if len(chunk) < PAGE:
        break

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows pulled: {len(df):,}")
print(f"With lat/long:     {df['latitude'].notna().sum():,} "
      f"({df['latitude'].notna().mean()*100:.1f}%)")
print(f"Primary types:     {df['primary_type'].nunique()}")
print(f"Date range:        {df['date'].min()} -> {df['date'].max()}")

out = OUT_DIR / "chicago_crime_2023_2026.csv"
df.to_csv(out, index=False)
print(f"\nSaved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
