"""17_housing_fusion.py — Chicago housing-value layer (London price-tertile analogue).

Cook County Assessor open data: parcel-level assessed values (uzyt-m557, certified_tot)
joined to census tracts via the Parcel Universe (nj4t-kc8j, census_tract_geoid).
Median assessed value per tract -> tertile bands (price_low/mid/high), mirroring the
London construct (script 14). Static layer.
"""
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/chicago/housing"
RAW.mkdir(parents=True, exist_ok=True)


def page(url, params, path, page_size=500000):
    if path.exists():
        return pd.read_csv(path, dtype=str)
    frames, offset = [], 0
    while True:
        p = dict(params, **{"$limit": page_size, "$offset": offset})
        r = requests.get(url, params=p, timeout=300)
        r.raise_for_status()
        chunk = pd.read_csv(StringIO(r.text), dtype=str)
        if chunk.empty:
            break
        frames.append(chunk)
        print(f"   {path.name}: +{len(chunk):,} (offset {offset:,})", flush=True)
        offset += page_size
        if len(chunk) < page_size:
            break
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


print("=" * 70)
print("CHICAGO HOUSING LAYER — Cook County assessed values -> tract tertiles")
print("=" * 70)

# latest complete assessment year
r = requests.get("https://datacatalog.cookcountyil.gov/resource/uzyt-m557.json",
                 params={"$select": "max(year)"}, timeout=60)
yr = int(float(list(r.json()[0].values())[0])) - 1   # use latest fully certified year
print(f"   using year = {yr}")

vals = page("https://datacatalog.cookcountyil.gov/resource/uzyt-m557.csv",
            {"$select": "pin,certified_tot", "$where": f"year={yr}"},
            RAW / f"assessed_{yr}.csv")
uni = page("https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.csv",
           {"$select": "pin,census_tract_geoid"},
           RAW / "parcel_tracts.csv")

vals["certified_tot"] = pd.to_numeric(vals["certified_tot"], errors="coerce")
vals = vals.dropna(subset=["certified_tot"])
vals = vals[vals["certified_tot"] > 0]
uni = uni.dropna(subset=["census_tract_geoid"]).drop_duplicates("pin")

m = vals.merge(uni, on="pin", how="inner")
med = m.groupby("census_tract_geoid")["certified_tot"].median().reset_index()
med = med.rename(columns={"census_tract_geoid": "GEOID", "certified_tot": "median_value"})
print(f"   tracts with median value: {len(med):,}")

q = med["median_value"].quantile([1 / 3, 2 / 3])
med["price_low"] = (med["median_value"] < q.iloc[0]).astype(int)
med["price_mid"] = ((med["median_value"] >= q.iloc[0]) & (med["median_value"] < q.iloc[1])).astype(int)
med["price_high"] = (med["median_value"] >= q.iloc[1]).astype(int)

run_fusion("Housing (assessed-value tertiles)", ["price_low", "price_mid", "price_high"],
           static_by_geoid=med[["GEOID", "price_low", "price_mid", "price_high"]])
