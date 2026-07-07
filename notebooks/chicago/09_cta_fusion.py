"""
Layer: TRANSIT RIDERSHIP for Chicago = CTA 'L' station monthly entries.

Direct analogue to London's TfL station taps (its largest per-type signal).
- Monthly station entries: Socrata t2rn-p8d7 (station_id, month_beginning, monthtotal)
- Station coordinates: Socrata 8pix-ypne ('L' stops, map_id + location)
Aggregate monthly rides per station -> spatial-join station -> tract -> sum per tract-month.
Dynamic layer; only tracts containing an 'L' station match (mirrors London's sparse coverage).
"""
from io import StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/transport"
RAW.mkdir(parents=True, exist_ok=True)
TRACTS = PROJECT_ROOT / "data/raw/chicago/boundaries/cook_tracts_2020.geojson"


def fetch(url, params, path):
    if not path.exists():
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        path.write_text(r.text)
    return pd.read_csv(path)


print("=" * 70)
print("CHICAGO TRANSIT LAYER — CTA 'L' station monthly ridership")
print("=" * 70)

rides = fetch("https://data.cityofchicago.org/resource/t2rn-p8d7.csv",
              {"$select": "station_id,stationame,month_beginning,monthtotal", "$limit": 200000},
              RAW / "cta_monthly_rides.csv")
rides["month"] = pd.to_datetime(rides["month_beginning"]).dt.strftime("%Y-%m")
rides = rides[(rides["month"] >= "2023-01") & (rides["month"] <= "2026-01")]
rides = rides.groupby(["station_id", "month"], as_index=False)["monthtotal"].sum()

# 'L' stops dataset = 8mj8-j3c4; 'location' is a GeoJSON Point -> fetch JSON.
stops_path = RAW / "cta_stops.json"
if not stops_path.exists():
    rr = requests.get("https://data.cityofchicago.org/resource/8mj8-j3c4.json",
                      params={"$select": "map_id,station_name,location", "$limit": 5000}, timeout=120)
    rr.raise_for_status()
    stops_path.write_text(rr.text)
import json
recs = json.loads(stops_path.read_text())
rows = []
for s in recs:
    loc = s.get("location")
    if loc and loc.get("coordinates"):
        lon, lat = loc["coordinates"][0], loc["coordinates"][1]
        rows.append({"map_id": str(s["map_id"]), "lon": lon, "lat": lat})
stops = pd.DataFrame(rows).drop_duplicates("map_id")

# station -> tract
sg = gpd.GeoDataFrame(stops, geometry=[Point(xy) for xy in zip(stops.lon, stops.lat)], crs="EPSG:4326")
tracts = gpd.read_file(TRACTS).to_crs("EPSG:4326")[["GEOID", "geometry"]]
sg = gpd.sjoin(sg, tracts, how="inner", predicate="within")[["map_id", "GEOID"]]
print(f"   'L' stations geocoded to tracts: {len(sg)}")

rides["station_id"] = rides["station_id"].astype(str)
rides = rides.merge(sg, left_on="station_id", right_on="map_id", how="inner")
tract_month = rides.groupby(["GEOID", "month"], as_index=False)["monthtotal"].sum()
tract_month = tract_month.rename(columns={"monthtotal": "cta_rides"})
print(f"   tract-months with CTA ridership: {len(tract_month):,} | tracts: {tract_month['GEOID'].nunique()}")

run_fusion("CTA ridership", ["cta_rides"], dynamic_panel=tract_month)
