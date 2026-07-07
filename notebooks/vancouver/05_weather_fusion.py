"""
Layer: WEATHER for Vancouver = ECCC daily, Vancouver Int'l Airport (YVR, stationID 51442),
aggregated monthly. City-wide temporal layer (join on month). Features: tmax_mean,
tmin_mean, prcp_total, snow_total.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/weather"
RAW.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 thesis-crime-research"}
BULK = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

print("=" * 70)
print("VANCOUVER WEATHER LAYER — ECCC YVR (stationID 51442), monthly")
print("=" * 70)

frames = []
for yr in (2023, 2024, 2025):
    f = RAW / f"yvr_{yr}.csv"
    if not f.exists():
        params = {"format": "csv", "stationID": "51442", "Year": yr, "Month": 1,
                  "Day": 1, "timeframe": 2, "submit": "Download Data"}
        r = requests.get(BULK, params=params, headers=H, timeout=180)
        r.raise_for_status()
        f.write_text(r.text)
    frames.append(pd.read_csv(f))
w = pd.concat(frames, ignore_index=True)
w.columns = [c.strip() for c in w.columns]

# locate columns (names carry units / degree signs)
def col(sub):
    return [c for c in w.columns if sub.lower() in c.lower()][0]

dt = col("Date/Time")
w["month"] = pd.to_datetime(w[dt]).dt.strftime("%Y-%m")
tmax, tmin = col("Max Temp"), col("Min Temp")
prcp = col("Total Precip")
snow = [c for c in w.columns if "Total Snow" in c]
w[tmax] = pd.to_numeric(w[tmax], errors="coerce")
w[tmin] = pd.to_numeric(w[tmin], errors="coerce")
w[prcp] = pd.to_numeric(w[prcp], errors="coerce")
agg = w.groupby("month").agg(tmax_mean=(tmax, "mean"), tmin_mean=(tmin, "mean"),
                             prcp_total=(prcp, "sum")).reset_index()
if snow:
    w[snow[0]] = pd.to_numeric(w[snow[0]], errors="coerce")
    agg = agg.merge(w.groupby("month")[snow[0]].sum().rename("snow_total").reset_index(), on="month")
    wcols = ["tmax_mean", "tmin_mean", "prcp_total", "snow_total"]
else:
    wcols = ["tmax_mean", "tmin_mean", "prcp_total"]
print(f"   weather months: {len(agg)} | {agg['month'].min()}->{agg['month'].max()}")

run_fusion("Weather", wcols, month_only=agg)
