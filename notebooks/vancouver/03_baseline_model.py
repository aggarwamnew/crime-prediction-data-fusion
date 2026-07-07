"""
Vancouver baseline — replication of London/Chicago baseline on Dissemination Areas.

Property crime only (violent/person offences are coordinate-suppressed in VPD open data).
Same recipe: 11 features, RF(200, depth=15, leaf=5, seed=42), last-6-months test split,
MIN_CRIMES=36. Window 2023-2025 (36 months). Saves the DA-month panel for fusion reuse.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
CRIME = ROOT / "data/raw/vancouver/crime/vancouver_crime_2023_2025.csv"
DA = ROOT / "data/raw/vancouver/boundaries/vancouver_da_2021.geojson"
PROC = ROOT / "data/processed/vancouver"
PROC.mkdir(parents=True, exist_ok=True)
MIN_CRIMES = 36

print("=" * 70)
print("VANCOUVER BASELINE (property crime, Dissemination Areas)")
print("=" * 70)

df = pd.read_csv(CRIME)
df.columns = [c.lower() for c in df.columns]
df = df[df["x"].fillna(0) != 0].copy()          # drop coordinate-suppressed (violent) crime
df["month"] = df["year"].astype(str) + "-" + df["month"].astype(int).map(lambda m: f"{m:02d}")
print(f"   geocodable property-crime incidents: {len(df):,}")

# crime X/Y are UTM Zone 10N (EPSG:32610)
pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["x"], df["y"])], crs="EPSG:32610")
da = gpd.read_file(DA)[["DAUID", "geometry"]]
pts = pts.to_crs(da.crs)
joined = gpd.sjoin(pts, da, how="inner", predicate="within")
print(f"   matched to a Vancouver DA: {len(joined):,} ({len(joined)/len(df)*100:.1f}%)")

monthly = joined.groupby(["DAUID", "month"]).size().reset_index(name="crime_count")
totals = monthly.groupby("DAUID")["crime_count"].sum()
active = totals[totals >= MIN_CRIMES].index
monthly = monthly[monthly["DAUID"].isin(active)]
print(f"   DAs >= {MIN_CRIMES} crimes: {len(active)} (of {da['DAUID'].nunique()} City DAs)")

all_months = sorted(monthly["month"].unique())
grid = pd.MultiIndex.from_product([active, all_months], names=["DAUID", "month"])
panel = monthly.set_index(["DAUID", "month"]).reindex(grid, fill_value=0).reset_index()
panel.to_parquet(PROC / "da_month_panel.parquet", index=False)
print(f"   Grid: {len(panel):,} rows ({len(active)} DAs x {len(all_months)} months)")

panel = panel.sort_values(["DAUID", "month"])
for lag in [1, 2, 3, 6, 12]:
    panel[f"lag_{lag}"] = panel.groupby("DAUID")["crime_count"].shift(lag)
for w in [3, 6, 12]:
    panel[f"rolling_mean_{w}"] = panel.groupby("DAUID")["crime_count"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
mn = panel["month"].str[-2:].astype(int)
panel["month_sin"] = np.sin(2 * np.pi * mn / 12)
panel["month_cos"] = np.cos(2 * np.pi * mn / 12)
panel["time_idx"] = panel["month"].map({m: i for i, m in enumerate(all_months)})

feats = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
         "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month_sin", "month_cos", "time_idx"]
model_df = panel.dropna().copy()
test_months = all_months[-6:]
tr = model_df[~model_df["month"].isin(test_months)]
te = model_df[model_df["month"].isin(test_months)]
print(f"   Train {len(tr):,} ({tr['month'].min()}->{tr['month'].max()}) | Test {len(te):,} ({te['month'].min()}->{te['month'].max()})")

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(tr[feats], tr["crime_count"])
pred = rf.predict(te[feats])
r2 = r2_score(te["crime_count"], pred)
mae = mean_absolute_error(te["crime_count"], pred)
rmse = mean_squared_error(te["crime_count"], pred) ** 0.5
mean_y = te["crime_count"].mean()

print("\n" + "=" * 70)
print("RESULTS  (vs London R2=0.943, Chicago R2=0.902)")
print("=" * 70)
print(f"  Train R2:  {r2_score(tr['crime_count'], rf.predict(tr[feats])):.4f}")
print(f"  Test  R2:  {r2:.4f}")
print(f"  Test  MAE: {mae:.4f}   RMSE: {rmse:.4f}   MAE/mean: {mae/mean_y*100:.1f}% (mean {mean_y:.2f})")
imp = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])[:3]
print("  Top 3 features: " + ", ".join(f"{f} ({v:.3f})" for f, v in imp))
