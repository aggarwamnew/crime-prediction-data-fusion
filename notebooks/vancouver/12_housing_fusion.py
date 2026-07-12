"""12_housing_fusion.py — Vancouver housing-value layer (London price-tertile analogue).

City of Vancouver open data: property tax report (assessed land+improvement value,
keyed by land_coordinate) joined to parcel polygons (tax_coord + geo_point_2d) for
location, then point-in-DA. Median total assessed value per DA -> tertile bands.
Static layer (assessed values are annual).
"""
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/housing"
RAW.mkdir(parents=True, exist_ok=True)
API = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets"


def export_csv(dataset, select, path):
    if path.exists():
        return pd.read_csv(path, sep=";", dtype=str)
    r = requests.get(f"{API}/{dataset}/exports/csv",
                     params={"select": select, "limit": -1}, timeout=600)
    r.raise_for_status()
    path.write_text(r.text)
    return pd.read_csv(StringIO(r.text), sep=";", dtype=str)


print("=" * 70)
print("VANCOUVER HOUSING LAYER — assessed values -> DA tertiles")
print("=" * 70)

tax = export_csv("property-tax-report",
                 "land_coordinate,current_land_value,current_improvement_value",
                 RAW / "tax_report.csv")
print(f"   tax rows: {len(tax):,}  cols: {list(tax.columns)}")
parc = export_csv("property-parcel-polygons", "tax_coord,geo_point_2d",
                  RAW / "parcel_points.csv")
print(f"   parcel rows: {len(parc):,}  cols: {list(parc.columns)}")

for c in ["current_land_value", "current_improvement_value"]:
    tax[c] = pd.to_numeric(tax[c], errors="coerce")
tax["total_value"] = tax["current_land_value"].fillna(0) + tax["current_improvement_value"].fillna(0)
tax = tax[tax["total_value"] > 0]

parc = parc.dropna(subset=["geo_point_2d", "tax_coord"]).drop_duplicates("tax_coord")
ll = parc["geo_point_2d"].str.split(",", expand=True)
parc["lat"] = pd.to_numeric(ll[0], errors="coerce")
parc["lon"] = pd.to_numeric(ll[1], errors="coerce")
parc = parc.dropna(subset=["lat", "lon"])

m = tax.merge(parc[["tax_coord", "lat", "lon"]],
              left_on="land_coordinate", right_on="tax_coord", how="inner")
print(f"   matched parcels with location: {len(m):,}")

g = gpd.GeoDataFrame(m, geometry=[Point(xy) for xy in zip(m.lon, m.lat)], crs="EPSG:4326")
da = gpd.read_file(ROOT / "data/raw/vancouver/boundaries/vancouver_da_2021.geojson")[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(g, da, how="inner", predicate="within")
med = j.groupby("DAUID")["total_value"].median().reset_index().rename(columns={"total_value": "median_value"})
print(f"   DAs with median value: {len(med):,}")

q = med["median_value"].quantile([1 / 3, 2 / 3])
med["price_low"] = (med["median_value"] < q.iloc[0]).astype(int)
med["price_mid"] = ((med["median_value"] >= q.iloc[0]) & (med["median_value"] < q.iloc[1])).astype(int)
med["price_high"] = (med["median_value"] >= q.iloc[1]).astype(int)

run_fusion("Housing (assessed-value tertiles)", ["price_low", "price_mid", "price_high"],
           static_by_da=med[["DAUID", "price_low", "price_mid", "price_high"]])
