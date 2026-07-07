"""
Download Chicago census-tract boundaries from Census TIGER/Line (2020 vintage).

The City of Chicago lies entirely within Cook County (FIPS 17031), so we pull the
Illinois tract file and filter to COUNTYFP == '031'. Suburban-Cook tracts that carry
no Chicago crime are dropped later by the minimum-crime threshold in the panel build.
2020 vintage aligns with the ACS / ADI / CDC PLACES fusion layers added later.
"""
from pathlib import Path

import geopandas as gpd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/raw/chicago/boundaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_17_tract.zip"
ZIP_PATH = OUT_DIR / "tl_2020_17_tract.zip"

print("=" * 70)
print("CHICAGO/COOK CENSUS-TRACT BOUNDARIES (TIGER 2020, Illinois)")
print("=" * 70)

if not ZIP_PATH.exists():
    print(f"Downloading {URL} ...")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    ZIP_PATH.write_bytes(r.content)
    print(f"   saved {ZIP_PATH.stat().st_size/1e6:.1f} MB")

gdf = gpd.read_file(f"zip://{ZIP_PATH}")
print(f"Illinois tracts: {len(gdf)}  | columns: {list(gdf.columns)[:8]} ...")

cook = gdf[gdf["COUNTYFP"] == "031"].copy()
print(f"Cook County tracts (FIPS 031): {len(cook)}")

out = OUT_DIR / "cook_tracts_2020.geojson"
cook.to_file(out, driver="GeoJSON")
print(f"CRS: {cook.crs}")
print(f"Saved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
