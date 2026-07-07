"""
Chicago baseline model — faithful replication of London notebooks/eda/03_baseline_model.py.

Same recipe, different city:
  - aggregate all crime types to per-(tract, month) counts
  - MIN_CRIMES = 36 filter (avg >=1/month); this also drops suburban-Cook tracts
  - 11 features: lag_1/2/3/6/12, rolling_mean_3/6/12, month_sin, month_cos, time_idx
  - RF(200, max_depth=15, min_samples_leaf=5, random_state=42)
  - temporal split: last 6 months = test
Saves the monthly tract panel for reuse by later fusion experiments.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRIME = PROJECT_ROOT / "data/raw/chicago/crime/chicago_crime_2023_2026.csv"
TRACTS = PROJECT_ROOT / "data/raw/chicago/boundaries/cook_tracts_2020.geojson"
PROC = PROJECT_ROOT / "data/processed/chicago"
PROC.mkdir(parents=True, exist_ok=True)
MIN_CRIMES = 36

print("=" * 70)
print("CHICAGO CRIME-ONLY BASELINE  (replicates London script 03)")
print("=" * 70)

# 1. Load + clean crime
print("\n1. Loading crime + spatial-joining to tracts...")
df = pd.read_csv(CRIME, usecols=["date", "primary_type", "latitude", "longitude"])
df = df.dropna(subset=["latitude", "longitude"])
df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
print(f"   Geocoded incidents: {len(df):,}")

pts = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
    crs="EPSG:4326",
)
tracts = gpd.read_file(TRACTS).to_crs("EPSG:4326")[["GEOID", "geometry"]]
joined = gpd.sjoin(pts, tracts, how="inner", predicate="within")
print(f"   Incidents matched to a Cook tract: {len(joined):,} "
      f"({len(joined)/len(df)*100:.1f}%)")

# 2. Monthly counts per tract
monthly = (joined.groupby(["GEOID", "month"]).size()
           .reset_index(name="crime_count"))
print(f"   (tract, month) pairs: {len(monthly):,} | tracts: {monthly['GEOID'].nunique():,}")

# 3. Filter sparse tracts (drops suburban-Cook tracts with ~0 Chicago crime)
totals = monthly.groupby("GEOID")["crime_count"].sum()
active = totals[totals >= MIN_CRIMES].index
n_before = monthly["GEOID"].nunique()
monthly = monthly[monthly["GEOID"].isin(active)]
print(f"   Tracts >= {MIN_CRIMES} crimes: {len(active):,} (dropped {n_before - len(active):,})")

# 4. Complete tract x month grid (fill missing months with 0)
all_months = sorted(monthly["month"].unique())
grid = pd.MultiIndex.from_product([active, all_months], names=["GEOID", "month"])
panel = (monthly.set_index(["GEOID", "month"]).reindex(grid, fill_value=0)
         .reset_index())
print(f"   Grid: {len(panel):,} rows ({len(active):,} tracts x {len(all_months)} months)")
panel.to_parquet(PROC / "tract_month_panel.parquet", index=False)

# 5. Features (identical to London)
panel = panel.sort_values(["GEOID", "month"])
for lag in [1, 2, 3, 6, 12]:
    panel[f"lag_{lag}"] = panel.groupby("GEOID")["crime_count"].shift(lag)
for w in [3, 6, 12]:
    panel[f"rolling_mean_{w}"] = panel.groupby("GEOID")["crime_count"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
panel["month_num"] = panel["month"].str[-2:].astype(int)
panel["month_sin"] = np.sin(2 * np.pi * panel["month_num"] / 12)
panel["month_cos"] = np.cos(2 * np.pi * panel["month_num"] / 12)
month_to_idx = {m: i for i, m in enumerate(all_months)}
panel["time_idx"] = panel["month"].map(month_to_idx)

# 6. Temporal split: last 6 months = test
model_df = panel.dropna().copy()
test_months = all_months[-6:]
train_df = model_df[~model_df["month"].isin(test_months)]
test_df = model_df[model_df["month"].isin(test_months)]
feature_cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
                "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
                "month_sin", "month_cos", "time_idx"]
print(f"\n   Train: {len(train_df):,} rows ({train_df['month'].min()} -> {train_df['month'].max()})")
print(f"   Test:  {len(test_df):,} rows ({test_df['month'].min()} -> {test_df['month'].max()})")
print(f"   Features ({len(feature_cols)}): {feature_cols}")

# 7. Train RF
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                           n_jobs=-1, random_state=42)
rf.fit(train_df[feature_cols], train_df["crime_count"])
pred_tr = rf.predict(train_df[feature_cols])
pred_te = rf.predict(test_df[feature_cols])

tr_r2 = r2_score(train_df["crime_count"], pred_tr)
te_r2 = r2_score(test_df["crime_count"], pred_te)
mae = mean_absolute_error(test_df["crime_count"], pred_te)
rmse = mean_squared_error(test_df["crime_count"], pred_te) ** 0.5
mean_y = test_df["crime_count"].mean()

print("\n" + "=" * 70)
print("RESULTS  (Chicago baseline vs London R2=0.943, MAE=4.38)")
print("=" * 70)
print(f"  Train R2:          {tr_r2:.4f}")
print(f"  Test  R2:          {te_r2:.4f}")
print(f"  Test  MAE:         {mae:.4f}")
print(f"  Test  RMSE:        {rmse:.4f}")
print(f"  MAE as % of mean:  {mae/mean_y*100:.1f}%  (test mean={mean_y:.2f})")

imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda t: -t[1])
print("\n  Top 3 features:")
for f, v in imp[:3]:
    print(f"    {f}: {v:.4f}")
print("\nDone.")
