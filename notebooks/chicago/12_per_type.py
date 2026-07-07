"""
Chicago PER-CRIME-TYPE fusion — mirror of London script 11 (Table 5.6).

For each major crime type, builds that type's per-(tract,month) panel on the active
tracts, then reports baseline R2 and Delta R2 for each supplementary layer. Reveals
per-type heterogeneity (London: weapons<->transit/POI, drugs<->deprivation).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import layers as L

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/chicago"
PROC = ROOT / "data/processed/chicago"
BASE = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month_sin", "month_cos", "time_idx"]

TYPES = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "ROBBERY",
         "MOTOR VEHICLE THEFT", "BURGLARY", "NARCOTICS", "WEAPONS VIOLATION", "DECEPTIVE PRACTICE"]

print("=" * 70)
print("CHICAGO PER-CRIME-TYPE FUSION  [mirror London 11]")
print("=" * 70)

# 1. per (tract, month, type) counts  (cache)
cache = PROC / "tract_month_type.parquet"
if cache.exists():
    cnt = pd.read_parquet(cache)
else:
    df = pd.read_csv(RAW / "crime/chicago_crime_2023_2026.csv",
                     usecols=["date", "primary_type", "latitude", "longitude"]).dropna(subset=["latitude", "longitude"])
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)], crs="EPSG:4326")
    tr = gpd.read_file(RAW / "boundaries/cook_tracts_2020.geojson").to_crs("EPSG:4326")[["GEOID", "geometry"]]
    j = gpd.sjoin(pts, tr, how="inner", predicate="within")
    cnt = j.groupby(["GEOID", "month", "primary_type"]).size().reset_index(name="n")
    cnt.to_parquet(cache, index=False)
print(f"   (tract,month,type) rows: {len(cnt):,}")

active = pd.read_parquet(PROC / "tract_month_panel.parquet")["GEOID"].unique()
all_months = sorted(cnt["month"].unique())
test_months = all_months[-6:]
month_idx = {m: i for i, m in enumerate(all_months)}


def build_features(sub):
    sub = sub.sort_values(["GEOID", "month"])
    for lag in [1, 2, 3, 6, 12]:
        sub[f"lag_{lag}"] = sub.groupby("GEOID")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        sub[f"rolling_mean_{w}"] = sub.groupby("GEOID")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    mn = sub["month"].str[-2:].astype(int)
    sub["month_sin"] = np.sin(2 * np.pi * mn / 12)
    sub["month_cos"] = np.cos(2 * np.pi * mn / 12)
    sub["time_idx"] = sub["month"].map(month_idx)
    return sub


def evals(df, feats):
    d = df.dropna(subset=feats + ["crime_count"])
    tr = d[~d["month"].isin(test_months)]
    te = d[d["month"].isin(test_months)]
    if len(te) < 50 or tr["crime_count"].sum() == 0:
        return None
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(tr[feats], tr["crime_count"])
    return r2_score(te["crime_count"], rf.predict(te[feats]))


# preload layers
LSPECS = {"SVI": L.svi(), "Weather": L.weather(), "Demo": L.demographics(),
          "MentHlth": L.mental_health(), "CTA": L.cta(), "Divvy": L.divvy()}
poi = L.pois()
if poi:
    LSPECS["POI"] = poi
    print("   POI included")

results = []
for t in TYPES:
    sub = cnt[cnt["primary_type"] == t]
    grid = pd.MultiIndex.from_product([active, all_months], names=["GEOID", "month"])
    panel = (sub.groupby(["GEOID", "month"])["n"].sum().reindex(grid, fill_value=0)
             .reset_index(name="crime_count"))
    panel = build_features(panel)
    base = evals(panel, BASE)
    if base is None:
        continue
    row = {"type": t, "base_R2": base}
    for name, (kind, tbl, cols) in LSPECS.items():
        if kind == "weather_monthly":
            m = panel.merge(tbl, on="month", how="inner")
        elif kind == "static":
            m = panel.merge(tbl, on="GEOID", how="inner")
        else:  # dynamic
            m = panel.merge(tbl, on=["GEOID", "month"], how="inner")
        m = m.dropna(subset=cols)
        b = evals(m, BASE)
        f = evals(m, BASE + cols)
        row[name] = (f - b) if (b is not None and f is not None) else np.nan
    results.append(row)
    print(f"   {t}: base R2={base:.3f}")

res = pd.DataFrame(results).set_index("type")
res.to_csv(PROC / "per_type_fusion.csv")
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\nPER-TYPE Delta R2 BY LAYER:")
print(res.round(4).to_string())
print("\nBest layer per type:")
layer_cols = [c for c in res.columns if c != "base_R2"]
for t in res.index:
    best = res.loc[t, layer_cols].idxmax()
    print(f"   {t:22s} -> {best} ({res.loc[t, best]:+.4f})")
