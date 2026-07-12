"""15_mobi_fusion.py — Vancouver bike-share layer (Mobi by Rogers; Santander/Divvy analogue).

Monthly trip files are published as Google Drive links on mobibikes.ca/en/system-data;
the file IDs below were scraped from that page (2026-07-13) for Jan 2023 - Dec 2025.
Trips are aggregated to departures per station-month, stations geocoded via the Mobi
GBFS station_information feed, spatially joined to DAs, and fused as a dynamic layer
(zero-filled for DAs without stations), mirroring the Santander and Divvy constructs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/mobi"
RAW.mkdir(parents=True, exist_ok=True)

FILE_IDS = {  # scraped from mobibikes.ca/en/system-data
    "2023-01": "12aZ-TeKZcPT6w1mLZWfSer5z0V-MIgY6", "2023-02": "106aqqsbwIYHqyJpQuH-7Xj-UJrKotfp6",
    "2023-03": "108-suaYJS1xhdLngXWh02PKUNTrKemd6", "2023-04": "10E5MOzfZPNYhT7UJbKOkYn4_GTyAsJQA",
    "2023-05": "17iCvj1J-VHYUEPOxYfIXmRHhKz1zX8qw", "2023-06": "17iKflbcZDSi3pDXDmXF42Cfkfo-Tkwr2",
    "2023-07": "17iP0AijcRCmpZ0icPYTqhxzztFdtdqfE", "2023-08": "17oWflFFNCWvwwHA_wgPs-umZr5NMBapO",
    "2023-09": "17SowN86MrVJXpI7ou5Y2qrvCgXJSTlAI", "2023-10": "1gKUKesn99zxt76qz8a-1o1AeW8WSIJuK",
    "2023-11": "10medxsRW5v0-hROBfgBIybT7T47Snbcy", "2023-12": "13jW3rph1VyC13EzBxGOMZdFP3xgTb_m-",
    "2024-01": "16wW2fgkyaXoQewe29GxWrrpJFtDUqikJ",
    "2024-02": "SHEET:19hgt6uX63S8NGGJqJtLLhjToapnyz_gy",
    "2024-03": "1FFWoEhYeWz_2-ykei9hJ32oe4DOZxn4x", "2024-04": "1F7GmkgwoWPMh_iaeu619cKzhxLGtM1G1",
    "2024-05": "1IBYLLJIPMmKCZjkYJ735SDKMTQno95RX", "2024-06": "1LZtI4d_38boFZJ1-kZM9KOCpymFwnAU8",
    "2024-07": "1OohrTe-XZphZmaT-s6HUpwvB_JcApfOr", "2024-08": "1-6RhRYMop395N92ZEp-KwRFpXC8Ou1Fs",
    "2024-09": "1-3b7Q8s5Q8-bil9cmPAW2ZxoMwH0JCBC", "2024-10": "12ggXy8RyqV3in2bGT0d_KK8T1ei-69qs",
    "2024-11": "15jvNfaNoPcIwdT2AwisXX5Acdm1AuPDh", "2024-12": "18N6CcYDi_QAycbg75anQIkbp0yGWXWWw",
    "2025-01": "1HZAJUC7jyvkye6oy6ffWf4nHtqwOQ1aa", "2025-02": "17-exUUZttoogAFQxpzkAHGCBIeUiu2GU",
    "2025-03": "1fmz6GWsnmIgYqRFa4LzPW5xPUVolq6px", "2025-04": "1N_Hk_kGCupvO15zs85IemwH7wTwR0vaO",
    "2025-05": "1Cj7NgbCrWJJjuty4E0qtCT8b_fLJJrTy", "2025-06": "1pfOI2AIgS_qiSeHbKqFm_egIGFJGXDxO",
    "2025-07": "1AZUDC3UNb7nXNDcCTGIJyAlVTdakGqBf", "2025-08": "1wH1hGY4u4PFU2tOA2iVB2YqZB2k6mD07",
    "2025-09": "1J7C5G5vNSmDkqfoUTsQ_LnQVaR-X737J", "2025-10": "1yohvHSSBqxxaN-cvTFfK-h5OFCBfmzIi",
    "2025-11": "1up0ysRQylyweDN1SRsaCrZesBs_gaGPc", "2025-12": "1oE3m-KLr_7SyvQopEGCvkMnZ_jlv-9FP",
}

import gdown  # noqa: E402

# ── 1. download + aggregate departures per station-month ──
agg_path = RAW / "mobi_station_month.csv"
if agg_path.exists():
    agg = pd.read_csv(agg_path)
else:
    parts = []
    for month, fid in FILE_IDS.items():
        dest = RAW / f"mobi_{month}.csv"
        if not dest.exists():
            try:
                if fid.startswith("SHEET:"):
                    url = f"https://docs.google.com/spreadsheets/d/{fid[6:]}/export?format=csv"
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                else:
                    gdown.download(id=fid, output=str(dest), quiet=True)
            except Exception as e:
                print(f"   {month}: DOWNLOAD FAILED ({str(e)[:60]})", flush=True)
                continue
        try:
            d = pd.read_csv(dest, low_memory=False)
        except Exception as e:
            print(f"   {month}: READ FAILED ({str(e)[:60]})", flush=True)
            continue
        d.columns = [c.strip().lower() for c in d.columns]
        dep_col = next((c for c in d.columns if "departure station" in c), None)
        if dep_col is None:
            print(f"   {month}: no departure-station column ({list(d.columns)[:6]})", flush=True)
            continue
        g = d.groupby(dep_col).size().reset_index(name="trips")
        g.columns = ["station", "trips"]
        g["month"] = month
        parts.append(g)
        print(f"   {month}: {g['trips'].sum():,} trips, {len(g)} stations", flush=True)
    agg = pd.concat(parts, ignore_index=True)
    agg.to_csv(agg_path, index=False)

print(f"station-months: {len(agg):,} | months: {agg['month'].nunique()}")

# ── 2. station coordinates via Mobi GBFS ──
coords_path = RAW / "mobi_stations.csv"
if coords_path.exists():
    stations = pd.read_csv(coords_path)
else:
    js = requests.get("http://api.citybik.es/v2/networks/mobibikes", timeout=60).json()
    st = js["network"]["stations"]
    stations = pd.DataFrame([{"name": s["name"], "lat": s["latitude"], "lon": s["longitude"]} for s in st])
    print(f"   CityBikes OK ({len(stations)} stations)")
    stations.to_csv(coords_path, index=False)

# Mobi trip files prefix station names with an ID ("0001 10th & Cambie"); GBFS may too.
def norm(s):
    s = str(s).strip().lower()
    return s.split(" ", 1)[1] if s[:4].isdigit() and " " in s else s

stations["key"] = stations["name"].map(norm)
agg["key"] = agg["station"].map(norm)
m = agg.merge(stations.drop_duplicates("key")[["key", "lat", "lon"]], on="key", how="inner")
print(f"matched station-months: {len(m):,} / {len(agg):,}")

# ── 3. DA join + dynamic fusion ──
g = gpd.GeoDataFrame(m, geometry=[Point(xy) for xy in zip(m.lon, m.lat)], crs="EPSG:4326")
da = gpd.read_file(ROOT / "data/raw/vancouver/boundaries/vancouver_da_2021.geojson")[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(g, da, how="inner", predicate="within")
dyn = j.groupby(["DAUID", "month"], as_index=False)["trips"].sum().rename(columns={"trips": "mobi_trips"})
print(f"DA-months with Mobi: {len(dyn):,} | DAs: {dyn['DAUID'].nunique()}")

# zero-fill: full panel DAs x months (mirrors Santander/Divvy corrected construct)
from _fusion import load_panel_with_features
panel, months = load_panel_with_features()
base = panel[["DAUID", "month"]].drop_duplicates()
dyn_full = base.merge(dyn, on=["DAUID", "month"], how="left").fillna({"mobi_trips": 0})
run_fusion("Mobi bike-share", ["mobi_trips"], dynamic_panel=dyn_full)
