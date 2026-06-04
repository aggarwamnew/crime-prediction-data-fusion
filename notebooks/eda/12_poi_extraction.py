"""
POI Extraction from OpenStreetMap via Overpass API
Downloads POI counts per category for Greater London, then spatial-joins to LSOA 2021 boundaries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.utils.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data/raw/london/pois"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# POI categories relevant to crime (from literature + data_sources.md)
POI_TAGS = {
    'pubs': '["amenity"="pub"]',
    'bars': '["amenity"="bar"]',
    'nightclubs': '["amenity"="nightclub"]',
    'atms': '["amenity"="atm"]',
    'fast_food': '["amenity"="fast_food"]',
    'restaurants': '["amenity"="restaurant"]',
    'schools': '["amenity"="school"]',
    'bus_stops': '["highway"="bus_stop"]',
    'rail_stations': '["railway"="station"]',
    'parks': '["leisure"="park"]',
    'shops': '["shop"]',
}

# London bounding box (approx Greater London)
LONDON_BBOX = "51.28,-0.51,51.69,0.33"  # south,west,north,east

print("=" * 70)
print("POI EXTRACTION FROM OPENSTREETMAP")
print("=" * 70)

all_pois = []
for category, tag in POI_TAGS.items():
    print(f"\n  Fetching {category}...", end=" ", flush=True)
    query = f"""
    [out:json][timeout:120];
    (
      node{tag}({LONDON_BBOX});
      way{tag}({LONDON_BBOX});
    );
    out center;
    """
    try:
        resp = requests.get(OVERPASS_URL, params={'data': query}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get('elements', [])
        for e in elements:
            lat = e.get('lat') or e.get('center', {}).get('lat')
            lon = e.get('lon') or e.get('center', {}).get('lon')
            if lat and lon:
                all_pois.append({'category': category, 'lat': lat, 'lon': lon})
        print(f"{len(elements)} found")
    except Exception as ex:
        print(f"ERROR: {ex}")

print(f"\n  Total POIs extracted: {len(all_pois):,}")

# Save raw POIs
pois_df = pd.DataFrame(all_pois)
pois_df.to_csv(OUT_DIR / "london_pois_raw.csv", index=False)
print(f"  Saved raw POIs to {OUT_DIR / 'london_pois_raw.csv'}")

# Convert to GeoDataFrame
print("\n  Creating GeoDataFrame...")
geometry = [Point(row['lon'], row['lat']) for _, row in pois_df.iterrows()]
pois_gdf = gpd.GeoDataFrame(pois_df, geometry=geometry, crs="EPSG:4326")

# Load LSOA boundaries
print("  Loading LSOA 2021 boundaries...")
lsoa_path = PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson"
lsoa_gdf = gpd.read_file(lsoa_path)
print(f"  LSOAs loaded: {len(lsoa_gdf):,}")
# Check LSOA code column name
print(f"  LSOA columns: {list(lsoa_gdf.columns[:5])}")

# Spatial join: assign each POI to its containing LSOA
print("\n  Spatial joining POIs to LSOAs...")
pois_gdf = pois_gdf.to_crs(lsoa_gdf.crs)
joined = gpd.sjoin(pois_gdf, lsoa_gdf, how='left', predicate='within')

# Find LSOA code column (could be LSOA21CD or similar)
lsoa_code_col = [c for c in joined.columns if 'LSOA' in c.upper() and 'CD' in c.upper()]
if not lsoa_code_col:
    lsoa_code_col = [c for c in joined.columns if 'lsoa' in c.lower() and 'code' in c.lower()]
if not lsoa_code_col:
    print(f"  WARNING: Could not find LSOA code column. Available: {list(joined.columns)}")
    lsoa_code_col = ['LSOA21CD']  # Best guess

lsoa_col = lsoa_code_col[0]
print(f"  Using LSOA column: {lsoa_col}")

# Count POIs per LSOA per category
print("  Counting POIs per LSOA per category...")
poi_counts = joined.groupby([lsoa_col, 'category']).size().unstack(fill_value=0)
poi_counts.columns = [f'poi_{c}' for c in poi_counts.columns]
poi_counts = poi_counts.reset_index().rename(columns={lsoa_col: 'lsoa_code'})

# Add total POI count
poi_feature_cols = [c for c in poi_counts.columns if c.startswith('poi_')]
poi_counts['poi_total'] = poi_counts[poi_feature_cols].sum(axis=1)

print(f"\n  POI features per LSOA:")
for col in poi_feature_cols + ['poi_total']:
    print(f"    {col:25s} mean={poi_counts[col].mean():.1f}  max={poi_counts[col].max()}")

# Save
poi_counts.to_csv(OUT_DIR / "poi_counts_per_lsoa.csv", index=False)
print(f"\n  Saved to {OUT_DIR / 'poi_counts_per_lsoa.csv'}")
print(f"  LSOAs with POI data: {len(poi_counts):,}")
print("\n✅ POI extraction complete!")
