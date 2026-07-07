"""
Shared loaders for Chicago fusion layers — reconstruct each layer's feature table from
the cached raw downloads, so full-fusion and per-type analyses reuse identical inputs.
Returns ("static", df[GEOID,...]) or ("dynamic", df[GEOID,month,...]).
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/chicago"
TRACTS = RAW / "boundaries/cook_tracts_2020.geojson"


def _tracts():
    return gpd.read_file(TRACTS).to_crs("EPSG:4326")[["GEOID", "geometry"]]


def svi():
    cols = ["RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
    d = pd.read_csv(RAW / "svi/svi_2022_illinois.csv", dtype={"FIPS": str})[["FIPS"] + cols]
    d = d.rename(columns={"FIPS": "GEOID"})
    d[cols] = d[cols].where(d[cols] >= 0)
    return "static", d.dropna(subset=["RPL_THEMES"]), cols


def demographics():
    d = pd.read_csv(RAW / "svi/svi_2022_illinois.csv", dtype={"FIPS": str})[
        ["FIPS", "E_TOTPOP", "E_AGE65", "E_AGE17"]].rename(columns={"FIPS": "GEOID"})
    for c in ["E_TOTPOP", "E_AGE65", "E_AGE17"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["E_TOTPOP"] > 0]
    tr = gpd.read_file(TRACTS)[["GEOID", "ALAND"]]
    tr["ALAND"] = pd.to_numeric(tr["ALAND"], errors="coerce")
    d = d.merge(tr, on="GEOID")
    d["pop_density"] = d["E_TOTPOP"] / (d["ALAND"] / 1e6)
    d["pct_age65"] = d["E_AGE65"] / d["E_TOTPOP"]
    d["pct_age17"] = d["E_AGE17"] / d["E_TOTPOP"]
    cols = ["pop_density", "pct_age65", "pct_age17"]
    return "static", d[["GEOID"] + cols].dropna(), cols


def mental_health():
    p = pd.read_csv(RAW / "cdc_places/places_mhlth_il.csv", dtype={"locationname": str})
    if "data_value_type" in p.columns:
        p = p[p["data_value_type"].str.contains("Crude", na=False)]
    p = p[p["year"] == p["year"].max()][["locationname", "data_value"]]
    p = p.rename(columns={"locationname": "GEOID", "data_value": "mhlth"}).dropna()
    return "static", p, ["mhlth"]


def weather():
    w = pd.read_csv(RAW / "weather/ohare_ghcnd.csv", low_memory=False)
    w["DATE"] = pd.to_datetime(w["DATE"])
    w = w[(w["DATE"] >= "2023-01-01") & (w["DATE"] < "2026-02-01")].copy()
    w["month"] = w["DATE"].dt.strftime("%Y-%m")
    for c in ["TMAX", "TMIN", "PRCP", "SNOW"]:
        if c not in w.columns:
            w[c] = pd.NA
    agg = w.groupby("month").agg(tmax_mean=("TMAX", "mean"), tmin_mean=("TMIN", "mean"),
                                 prcp_total=("PRCP", "sum"), snow_total=("SNOW", "sum")).reset_index()
    agg[["tmax_mean", "tmin_mean"]] /= 10.0
    cols = ["tmax_mean", "tmin_mean", "prcp_total", "snow_total"]
    return "weather_monthly", agg, cols  # special: join on month only


def cta():
    rides = pd.read_csv(RAW / "transport/cta_monthly_rides.csv")
    rides["month"] = pd.to_datetime(rides["month_beginning"]).dt.strftime("%Y-%m")
    rides = rides[(rides["month"] >= "2023-01") & (rides["month"] <= "2026-01")]
    rides = rides.groupby(["station_id", "month"], as_index=False)["monthtotal"].sum()
    recs = json.loads((RAW / "transport/cta_stops.json").read_text())
    rows = [{"map_id": str(s["map_id"]), "lon": s["location"]["coordinates"][0],
             "lat": s["location"]["coordinates"][1]} for s in recs if s.get("location")]
    stops = pd.DataFrame(rows).drop_duplicates("map_id")
    sg = gpd.GeoDataFrame(stops, geometry=[Point(xy) for xy in zip(stops.lon, stops.lat)], crs="EPSG:4326")
    sg = gpd.sjoin(sg, _tracts(), how="inner", predicate="within")[["map_id", "GEOID"]]
    rides["station_id"] = rides["station_id"].astype(str)
    tm = rides.merge(sg, left_on="station_id", right_on="map_id").groupby(
        ["GEOID", "month"], as_index=False)["monthtotal"].sum().rename(columns={"monthtotal": "cta_rides"})
    return "dynamic", tm, ["cta_rides"]


def divvy():
    agg = pd.read_csv(RAW / "divvy/divvy_station_month.csv")
    agg = agg[(agg["month"] >= "2023-01") & (agg["month"] <= "2026-01")]
    coords = agg.groupby("start_station_id").agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    sg = gpd.GeoDataFrame(coords, geometry=[Point(xy) for xy in zip(coords.lon, coords.lat)], crs="EPSG:4326")
    sg = gpd.sjoin(sg, _tracts(), how="inner", predicate="within")[["start_station_id", "GEOID"]]
    tm = agg.merge(sg, on="start_station_id").groupby(["GEOID", "month"], as_index=False)["trips"].sum()
    return "dynamic", tm.rename(columns={"trips": "divvy_trips"}), ["divvy_trips"]


def pois():
    """POI counts per tract (only if the Overpass cache exists)."""
    cache = RAW / "pois/chicago_pois_raw.csv"
    if not cache.exists():
        return None
    p = pd.read_csv(cache)
    gp = gpd.GeoDataFrame(p, geometry=[Point(xy) for xy in zip(p.lon, p.lat)], crs="EPSG:4326")
    j = gpd.sjoin(gp, _tracts(), how="inner", predicate="within")
    counts = j.groupby(["GEOID", "category"]).size().unstack(fill_value=0).reset_index()
    cat_cols = [c for c in counts.columns if c != "GEOID"]
    counts["poi_total"] = counts[cat_cols].sum(axis=1)
    return "static", counts, cat_cols + ["poi_total"]
