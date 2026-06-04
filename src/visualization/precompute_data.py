"""
precompute_data.py — Pre-compute per-LSOA per-month predictions for all crime types.

Produces:
  - data/processed/london/predictions.parquet  (lsoa × month × crime_type → actual, predicted)
  - data/processed/london/lsoa_features.parquet (lsoa → static features + geometry)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

db = ThesisDB()

print("=" * 60)
print("PRE-COMPUTING PREDICTION DATA")
print("=" * 60)

# ── 1. ALL CRIMES BASELINE ──
print("\n  [1/3] Building all-crimes predictions...")
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]

all_months = sorted(crime['month'].unique())
all_lsoas = sorted(active)
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = crime.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rm_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year - ts.dt.year.min()) * 12 + ts.dt.month

features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rm_3', 'rm_6', 'rm_12', 'month_sin', 'month_cos', 'time_idx']

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(train[features], train['crime_count'])

test_out = test[['lsoa_code', 'month', 'crime_count']].copy()
test_out['predicted'] = rf.predict(test[features])
test_out['crime_type'] = 'All crimes'
all_predictions = [test_out]
print(f"    All crimes: {len(test_out):,} predictions")

# ── 2. PER-CRIME-TYPE PREDICTIONS ──
print("\n  [2/3] Building per-type predictions...")
crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()

for ct in crime_types:
    ct_safe = ct.replace("'", "''")
    ct_data = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct_safe}' GROUP BY lsoa_code, month")

    ct_totals = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_totals[ct_totals >= 12].index
    if len(ct_active) < 50:
        print(f"    {ct}: SKIP (too few LSOAs)")
        continue

    ct_lsoas = sorted(ct_active)
    ct_months = sorted(ct_data['month'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])
    ct_df = ct_data.set_index(['lsoa_code', 'month']).reindex(ct_grid, fill_value=0).reset_index()
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rm_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rm_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rm_3', 'rm_12', 'month_sin', 'month_cos']
    ct_model = ct_df.dropna()
    ct_test_months = ct_months[-6:]
    ct_train = ct_model[~ct_model['month'].isin(ct_test_months)]
    ct_test = ct_model[ct_model['month'].isin(ct_test_months)]

    if len(ct_test) < 50:
        print(f"    {ct}: SKIP (too few test rows)")
        continue

    ct_rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    ct_rf.fit(ct_train[ct_feats], ct_train['crime_count'])

    ct_out = ct_test[['lsoa_code', 'month', 'crime_count']].copy()
    ct_out['predicted'] = ct_rf.predict(ct_test[ct_feats])
    ct_out['crime_type'] = ct
    all_predictions.append(ct_out)
    print(f"    {ct}: {len(ct_out):,} predictions")

predictions = pd.concat(all_predictions, ignore_index=True)
predictions['residual'] = predictions['crime_count'] - predictions['predicted']

# ── 3. STATIC FEATURES + GEOMETRY ──
print("\n  [3/3] Loading static features & geometry...")
lsoa_features = pd.DataFrame({'lsoa_code': all_lsoas})

# IMD
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
imd_slim = imd[['LSOA code (2021)', 'Index of Multiple Deprivation (IMD) Score']].copy()
imd_slim.columns = ['lsoa_code', 'imd_score']
lsoa_features = lsoa_features.merge(imd_slim, on='lsoa_code', how='left')

# Demographics
ts006 = pd.read_csv(PROJECT_ROOT / "data/raw/london/census/ts006/census2021-ts006-lsoa.csv")
ts006 = ts006.rename(columns={'geography code': 'lsoa_code',
                               'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})
lsoa_features = lsoa_features.merge(ts006[['lsoa_code', 'pop_density']], on='lsoa_code', how='left')

# POIs
poi = pd.read_csv(PROJECT_ROOT / "data/raw/london/pois/poi_counts_per_lsoa.csv")
lsoa_features = lsoa_features.merge(poi, on='lsoa_code', how='left')
for c in poi.columns:
    if c != 'lsoa_code':
        lsoa_features[c] = lsoa_features[c].fillna(0)

# Housing
housing = pd.read_csv(PROJECT_ROOT / "data/raw/london/housing/housing_per_lsoa.csv")
lsoa_features = lsoa_features.merge(housing[['lsoa_code', 'median_house_price']], on='lsoa_code', how='left')

# SAMHI
samhi = pd.read_csv(PROJECT_ROOT / "data/raw/london/mental_health/samhi_lsoa.csv")
samhi = samhi[['lsoa11', 'samhi_index.2022']].copy()
samhi.columns = ['lsoa_code', 'samhi_index']
lsoa_features = lsoa_features.merge(samhi, on='lsoa_code', how='left')

# Geometry
boundaries = gpd.read_file(PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson")
if 'LSOA21CD' in boundaries.columns:
    boundaries = boundaries.rename(columns={'LSOA21CD': 'lsoa_code'})
else:
    boundaries = boundaries.rename(columns={boundaries.columns[0]: 'lsoa_code'})
boundaries['geometry'] = boundaries['geometry'].simplify(tolerance=0.0005, preserve_topology=True)

gdf = boundaries[['lsoa_code', 'geometry']].merge(lsoa_features, on='lsoa_code', how='inner')
gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

# ── SAVE ──
pred_path = PROJECT_ROOT / "data/processed/london/predictions.parquet"
feat_path = PROJECT_ROOT / "data/processed/london/lsoa_features.parquet"

predictions.to_parquet(pred_path, index=False)
gdf.to_parquet(feat_path)

print(f"\n  ✅ Predictions: {pred_path} ({len(predictions):,} rows)")
print(f"  ✅ Features: {feat_path} ({len(gdf):,} LSOAs × {len(gdf.columns)} cols)")

db.close()
print("\n✅ Pre-computation complete!")
