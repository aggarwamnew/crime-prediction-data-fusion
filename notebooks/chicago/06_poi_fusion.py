"""
Layer: POINTs OF INTEREST for Chicago = OpenStreetMap via Overpass API.

Replicates London notebooks/eda/12_poi_extraction.py: same 10 crime-relevant categories,
queried over the Chicago bounding box, counted per census tract. Static spatial layer.
Features: 10 category counts + poi_total.
"""
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/chicago/pois"
RAW.mkdir(parents=True, exist_ok=True)
TRACTS = PROJECT_ROOT / "data/raw/chicago/boundaries/cook_tracts_2020.geojson"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "crime-prediction-data-fusion/1.0 (academic research)"}

# identical category set to London script 12
POI_TAGS = {
    "pubs": '["amenity"="pub"]', "bars": '["amenity"="bar"]',
    "nightclubs": '["amenity"="nightclub"]', "atms": '["amenity"="atm"]',
    "fast_food": '["amenity"="fast_food"]', "restaurants": '["amenity"="restaurant"]',
    "schools": '["amenity"="school"]', "bus_stops": '["highway"="bus_stop"]',
    "parks": '["leisure"="park"]', "shops": '["shop"]',
}
# Chicago bounding box (S, W, N, E)
BBOX = "41.62,-87.95,42.05,-87.50"

print("=" * 70)
print("CHICAGO POI LAYER — OpenStreetMap via Overpass (10 categories)")
print("=" * 70)

cache = RAW / "chicago_pois_raw.csv"
if cache.exists():
    pois = pd.read_csv(cache)
    print(f"   loaded cached POIs: {len(pois):,}")
else:
    rows = []
    for cat, tag in POI_TAGS.items():
        q = f"""
        [out:json][timeout:180];
        (
          node{tag}({BBOX});
          way{tag}({BBOX});
        );
        out center;
        """
        els = None
        for attempt in range(8):
            ep = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
            try:
                r = requests.get(ep, params={"data": q}, headers=HEADERS, timeout=180)
                r.raise_for_status()
                els = r.json().get("elements", [])
                break
            except Exception as e:
                print(f"   {cat} attempt {attempt+1} ({ep.split('//')[1].split('/')[0]}): {str(e)[:60]}", flush=True)
                time.sleep(min(60, 8 * (attempt + 1)))
        if els is None:
            print(f"   {cat}: FAILED all endpoints", flush=True)
            els = []
        for el in els:
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                rows.append({"category": cat, "lat": lat, "lon": lon})
        print(f"   {cat}: {sum(1 for x in rows if x['category']==cat):,}")
        time.sleep(2)
    pois = pd.DataFrame(rows)
    pois.to_csv(cache, index=False)
    print(f"   total POIs: {len(pois):,} -> cached")

# spatial join to tracts
gp = gpd.GeoDataFrame(pois, geometry=[Point(xy) for xy in zip(pois.lon, pois.lat)],
                      crs="EPSG:4326")
tracts = gpd.read_file(TRACTS).to_crs("EPSG:4326")[["GEOID", "geometry"]]
j = gpd.sjoin(gp, tracts, how="inner", predicate="within")
counts = j.groupby(["GEOID", "category"]).size().unstack(fill_value=0).reset_index()
cat_cols = [c for c in POI_TAGS if c in counts.columns]
counts["poi_total"] = counts[cat_cols].sum(axis=1)
feat_cols = cat_cols + ["poi_total"]
print(f"   Tracts with POIs: {len(counts):,} | features: {feat_cols}")

run_fusion("POIs", feat_cols, static_by_geoid=counts[["GEOID"] + feat_cols])
