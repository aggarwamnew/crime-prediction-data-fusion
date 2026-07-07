"""
Layer: BIKE-SHARE for Chicago = Divvy trips (analogue to London Santander Cycles).

Downloads monthly trip zips from the public S3 bucket (2023-01 .. 2025-12), aggregates
trip starts per station per month, geocodes stations from per-trip start_lat/lng,
spatial-joins to tracts -> dynamic tract-month feature. Only tracts with stations match.
"""
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/divvy"
RAW.mkdir(parents=True, exist_ok=True)
TRACTS = PROJECT_ROOT / "data/raw/chicago/boundaries/cook_tracts_2020.geojson"
AGG = RAW / "divvy_station_month.csv"

print("=" * 70)
print("CHICAGO BIKE-SHARE LAYER — Divvy trips (Santander analogue)")
print("=" * 70)

if not AGG.exists():
    months = [f"{y}{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]
    parts = []
    for ym in months:
        zf = RAW / f"{ym}-divvy-tripdata.zip"
        if not zf.exists():
            url = f"https://divvy-tripdata.s3.amazonaws.com/{ym}-divvy-tripdata.zip"
            try:
                r = requests.get(url, timeout=300)
                if r.status_code != 200:
                    print(f"   {ym}: HTTP {r.status_code} (skip)")
                    continue
                zf.write_bytes(r.content)
            except Exception as e:
                print(f"   {ym}: {str(e)[:60]} (skip)")
                continue
        try:
            z = zipfile.ZipFile(io.BytesIO(zf.read_bytes()))
            name = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__")][0]
            d = pd.read_csv(z.open(name),
                            usecols=["started_at", "start_station_id", "start_lat", "start_lng"],
                            low_memory=False)
        except Exception as e:
            print(f"   {ym}: read error {str(e)[:50]}")
            continue
        d["month"] = pd.to_datetime(d["started_at"], errors="coerce").dt.strftime("%Y-%m")
        d = d.dropna(subset=["start_lat", "start_lng", "month"])
        g = d.groupby(["start_station_id", "month"]).agg(
            trips=("started_at", "size"),
            lat=("start_lat", "median"), lon=("start_lng", "median")).reset_index()
        parts.append(g)
        print(f"   {ym}: {len(d):,} trips")
    allg = pd.concat(parts, ignore_index=True)
    allg.to_csv(AGG, index=False)
    print(f"   aggregated -> {AGG}")

agg = pd.read_csv(AGG)
agg = agg[(agg["month"] >= "2023-01") & (agg["month"] <= "2026-01")]
# station coordinate = median across its months
coords = agg.groupby("start_station_id").agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
sg = gpd.GeoDataFrame(coords, geometry=[Point(xy) for xy in zip(coords.lon, coords.lat)], crs="EPSG:4326")
tracts = gpd.read_file(TRACTS).to_crs("EPSG:4326")[["GEOID", "geometry"]]
sg = gpd.sjoin(sg, tracts, how="inner", predicate="within")[["start_station_id", "GEOID"]]
tm = agg.merge(sg, on="start_station_id").groupby(["GEOID", "month"], as_index=False)["trips"].sum()
tm = tm.rename(columns={"trips": "divvy_trips"})
print(f"   tract-months with Divvy: {len(tm):,} | tracts: {tm['GEOID'].nunique()}")

run_fusion("Divvy bikes", ["divvy_trips"], dynamic_panel=tm)
