"""
Download Census 2021 Dissemination Area (DA) boundaries from StatCan, filter to the
City of Vancouver (CSDUID = 5915022). DA is the LSOA/tract analogue (~400-700 persons).
Uses the cartographic boundary file (lda_000b21a_e).
"""
from pathlib import Path

import geopandas as gpd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data/raw/vancouver/boundaries"
OUT.mkdir(parents=True, exist_ok=True)
ZIP = OUT / "lda_000b21a_e.zip"
URL = ("https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/"
       "boundary-limites/files-fichiers/lda_000b21a_e.zip")
H = {"User-Agent": "Mozilla/5.0 thesis-crime-research"}

print("=" * 70)
print("VANCOUVER DA BOUNDARIES (StatCan 2021, City CSDUID 5915022)")
print("=" * 70)

if not ZIP.exists():
    print("   downloading national DA file (~large)...")
    r = requests.get(URL, headers=H, timeout=600)
    r.raise_for_status()
    ZIP.write_bytes(r.content)
    print(f"   saved {ZIP.stat().st_size/1e6:.1f} MB")

gdf = gpd.read_file(f"zip://{ZIP}")
print(f"   national DAs: {len(gdf):,} | cols: {list(gdf.columns)[:8]}")
# Cartographic file has no CSDUID; filter to BC (PRUID 59). The VPD crime data is
# City-of-Vancouver only, so the spatial join + MIN_CRIMES filter isolates Vancouver DAs.
van = gdf[gdf["PRUID"] == "59"].copy()
print(f"   BC DAs (PRUID 59): {len(van)}")
out = OUT / "vancouver_da_2021.geojson"
van.to_file(out, driver="GeoJSON")
print(f"   CRS: {van.crs} | saved -> {out} ({out.stat().st_size/1e6:.1f} MB)")
