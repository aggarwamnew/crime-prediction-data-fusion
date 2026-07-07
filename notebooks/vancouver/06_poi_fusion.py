"""
Layer: POINTs OF INTEREST for Vancouver = OSM via Overpass (same 10 categories as London).
Counts per DA. Static spatial layer. Endpoint rotation + backoff (Overpass load varies).
"""
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/pois"
RAW.mkdir(parents=True, exist_ok=True)
DA = ROOT / "data/raw/vancouver/boundaries/vancouver_da_2021.geojson"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
H = {"User-Agent": "crime-prediction-data-fusion/1.0 (academic research)"}
TAGS = {"pubs": '["amenity"="pub"]', "bars": '["amenity"="bar"]', "nightclubs": '["amenity"="nightclub"]',
        "atms": '["amenity"="atm"]', "fast_food": '["amenity"="fast_food"]', "restaurants": '["amenity"="restaurant"]',
        "schools": '["amenity"="school"]', "bus_stops": '["highway"="bus_stop"]', "parks": '["leisure"="park"]',
        "shops": '["shop"]'}
BBOX = "49.19,-123.27,49.32,-123.02"   # City of Vancouver

print("VANCOUVER POI LAYER — Overpass (10 categories)")
cache = RAW / "vancouver_pois_raw.csv"
if cache.exists():
    pois = pd.read_csv(cache)
else:
    rows = []
    for cat, tag in TAGS.items():
        q = f"[out:json][timeout:180];(node{tag}({BBOX});way{tag}({BBOX}););out center;"
        els = None
        for a in range(8):
            ep = ENDPOINTS[a % len(ENDPOINTS)]
            try:
                r = requests.get(ep, params={"data": q}, headers=H, timeout=180)
                r.raise_for_status()
                els = r.json().get("elements", [])
                break
            except Exception as e:
                print(f"   {cat} try{a+1} {ep.split('//')[1].split('/')[0]}: {str(e)[:50]}", flush=True)
                time.sleep(min(60, 8 * (a + 1)))
        for el in (els or []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                rows.append({"category": cat, "lat": lat, "lon": lon})
        print(f"   {cat}: {sum(1 for x in rows if x['category']==cat):,}", flush=True)
        time.sleep(2)
    pois = pd.DataFrame(rows)
    pois.to_csv(cache, index=False)

gp = gpd.GeoDataFrame(pois, geometry=[Point(xy) for xy in zip(pois.lon, pois.lat)], crs="EPSG:4326")
da = gpd.read_file(DA)[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(gp, da, how="inner", predicate="within")
counts = j.groupby(["DAUID", "category"]).size().unstack(fill_value=0).reset_index()
cat_cols = [c for c in TAGS if c in counts.columns]
counts["poi_total"] = counts[cat_cols].sum(axis=1)
print(f"   DAs with POIs: {len(counts)}")
run_fusion("POIs", cat_cols + ["poi_total"], static_by_da=counts[["DAUID"] + cat_cols + ["poi_total"]])
