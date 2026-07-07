"""
Layer: WEATHER for Chicago = NOAA GHCN-Daily, O'Hare (USW00094846), aggregated monthly.

Uses the per-station GHCN-Daily access CSV (no API token required), mirroring London's
single-station monthly weather. City-wide temporal layer: same value broadcast to all
tracts within a month (joined on month only). Features: tmax_mean, tmin_mean,
prcp_total, snow_total.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from _fusion import run_fusion, load_panel_with_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/weather"
RAW.mkdir(parents=True, exist_ok=True)
CSV = RAW / "ohare_ghcnd.csv"
URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094846.csv"

print("=" * 70)
print("CHICAGO WEATHER LAYER — NOAA GHCN-Daily O'Hare USW00094846")
print("=" * 70)

if not CSV.exists():
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    CSV.write_text(r.text)
    print(f"   downloaded -> {CSV.stat().st_size/1e6:.1f} MB")

w = pd.read_csv(CSV, low_memory=False)
w["DATE"] = pd.to_datetime(w["DATE"])
w = w[(w["DATE"] >= "2023-01-01") & (w["DATE"] < "2026-02-01")].copy()
w["month"] = w["DATE"].dt.strftime("%Y-%m")
# GHCN temps are in tenths of degrees C; precip/snow in tenths of mm
for c in ["TMAX", "TMIN", "PRCP", "SNOW"]:
    if c not in w.columns:
        w[c] = pd.NA
agg = w.groupby("month").agg(
    tmax_mean=("TMAX", "mean"),
    tmin_mean=("TMIN", "mean"),
    prcp_total=("PRCP", "sum"),
    snow_total=("SNOW", "sum"),
).reset_index()
agg[["tmax_mean", "tmin_mean"]] /= 10.0  # tenths C -> C
wcols = ["tmax_mean", "tmin_mean", "prcp_total", "snow_total"]
print(f"   Weather months: {len(agg)}  | {agg['month'].min()} -> {agg['month'].max()}")

# broadcast city-wide value to every tract for that month
panel, _ = load_panel_with_features()
tracts = panel[["GEOID"]].drop_duplicates()
dyn = tracts.merge(agg, how="cross") if False else None  # avoid cross; build per (GEOID,month)
dyn = (panel[["GEOID", "month"]].drop_duplicates()
       .merge(agg, on="month", how="inner"))

run_fusion("Weather", wcols, dynamic_panel=dyn[["GEOID", "month"] + wcols])
