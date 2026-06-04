"""
Experiment: POI Fusion - aggregate + per-crime-type
Tests whether Points of Interest density improves crime prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

db = ThesisDB()
print("=" * 70)
print("EXPERIMENT: POI FUSION")
print("=" * 70)

# Load POI counts per LSOA
poi = pd.read_csv(PROJECT_ROOT / "data/raw/london/pois/poi_counts_per_lsoa.csv")
poi_features = [c for c in poi.columns if c.startswith('poi_')]
print(f"\n1. POI data: {len(poi):,} LSOAs, {len(poi_features)} features")
print(f"   Features: {poi_features}")

# Load crime
monthly = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")
crime_lsoas = set(monthly['lsoa_code'].unique())
poi_lsoas = set(poi['lsoa_code'].unique())
match = len(crime_lsoas & poi_lsoas)
print(f"\n2. Match: {match:,}/{len(crime_lsoas):,} crime LSOAs have POI data ({match/len(crime_lsoas)*100:.1f}%)")
# Note: LSOAs with 0 POIs are NOT in the CSV (they were never joined).
# We should treat those as 0 for all POI features.
# Expand poi to include all crime LSOAs with 0 fills
all_crime_lsoas = pd.DataFrame({'lsoa_code': list(crime_lsoas)})
poi_full = all_crime_lsoas.merge(poi, on='lsoa_code', how='left').fillna(0)
print(f"   After filling missing with 0: {len(poi_full):,} LSOAs")

# ── AGGREGATE EXPERIMENT ──
print(f"\n{'='*70}")
print("AGGREGATE FUSION")
print(f"{'='*70}")

merged = monthly.merge(poi_full, on='lsoa_code', how='left')
merged[poi_features] = merged[poi_features].fillna(0)

MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
n_lsoas = merged['lsoa_code'].nunique()
print(f"   Active LSOAs: {n_lsoas:,}")

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

poi_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + poi_features].set_index('lsoa_code')
crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()
df = df.merge(poi_per, on='lsoa_code', how='left')
df[poi_features] = df[poi_features].fillna(0)

df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
month_idx = {m: i for i, m in enumerate(all_months)}
df['time_idx'] = df['month'].map(month_idx)

df_model = df.dropna().copy()
test_months = all_months[-6:]
train_df = df_model[~df_model['month'].isin(test_months)]
test_df = df_model[df_model['month'].isin(test_months)]

lag_features = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
temporal_features = ['month_sin', 'month_cos', 'time_idx']
crime_only_cols = lag_features + temporal_features
fused_cols = lag_features + temporal_features + poi_features

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")

rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])

results = {}
for name, model, cols in [("Crime-only", rf_crime, crime_only_cols), ("Fused (+POI)", rf_fused, fused_cols)]:
    y_pred = model.predict(test_df[cols])
    r2 = r2_score(test_df['crime_count'], y_pred)
    mae = mean_absolute_error(test_df['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"\n   {name}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

delta_r2 = results["Fused (+POI)"]['R²'] - results["Crime-only"]['R²']
print(f"\n   Δ R² = {delta_r2:+.4f}")

# Feature importance
print("\n   Feature importance (POI features):")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
poi_imp = imp[imp['feature'].isin(poi_features)]
for _, row in poi_imp.iterrows():
    print(f"     {row['feature']:25s} {row['importance']:.4f}")

# ── PER-CRIME-TYPE ──
print(f"\n{'='*70}")
print("PER-CRIME-TYPE POI ANALYSIS")
print(f"{'='*70}")

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
per_type_results = []

for ct in crime_types:
    print(f"\n  {ct}...", end=" ", flush=True)
    ct_monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")
    ct_merged = ct_monthly.merge(poi_full, on='lsoa_code', how='left')
    ct_merged[poi_features] = ct_merged[poi_features].fillna(0)
    
    lsoa_totals = ct_merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= 12].index
    if len(active) < 50: print("SKIP"); continue
    ct_merged = ct_merged[ct_merged['lsoa_code'].isin(active)]

    all_months = sorted(ct_merged['month'].unique())
    all_lsoas = sorted(ct_merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
    poi_per = ct_merged.drop_duplicates('lsoa_code')[['lsoa_code'] + poi_features].set_index('lsoa_code')
    ct_df = ct_merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
    ct_df = ct_df.merge(poi_per, on='lsoa_code', how='left')
    ct_df[poi_features] = ct_df[poi_features].fillna(0)
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rolling_mean_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rolling_mean_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_model = ct_df.dropna()
    if len(ct_model) < 100: print("SKIP"); continue
    test_months = all_months[-6:]
    train = ct_model[~ct_model['month'].isin(test_months)]
    test = ct_model[ct_model['month'].isin(test_months)]
    if len(test) < 50: print("SKIP"); continue

    lag_cols = ['lag_1','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_12','month_sin','month_cos']
    
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_base.fit(train[lag_cols], train['crime_count'])
    r2_base = r2_score(test['crime_count'], rf_base.predict(test[lag_cols]))
    
    rf_poi = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_poi.fit(train[lag_cols + poi_features], train['crime_count'])
    r2_poi = r2_score(test['crime_count'], rf_poi.predict(test[lag_cols + poi_features]))
    
    delta = r2_poi - r2_base
    
    # Top POI feature
    ct_imp = pd.DataFrame({'feature': lag_cols + poi_features, 'importance': rf_poi.feature_importances_})
    top_poi = ct_imp[ct_imp['feature'].isin(poi_features)].sort_values('importance', ascending=False).iloc[0]
    
    print(f"Δ={delta:+.4f}  top_poi={top_poi['feature']} ({top_poi['importance']:.4f})")
    per_type_results.append({'crime_type': ct, 'r2_base': r2_base, 'r2_poi': r2_poi, 'delta': delta, 'top_poi': top_poi['feature'], 'n_lsoas': len(active)})

db.close()

res = pd.DataFrame(per_type_results).sort_values('delta', ascending=False)
print(f"\n{'='*90}")
print("SUMMARY: POI IMPACT BY CRIME TYPE")
print(f"{'='*90}")
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ POI':>8} {'Top POI Feature':>20}")
print("-" * 75)
for _, r in res.iterrows():
    print(f"{r['crime_type']:<35} {r['r2_base']:>8.4f} {r['delta']:>+8.4f} {r['top_poi']:>20}")

print(f"\n   Aggregate Δ R² = {delta_r2:+.4f}")

# Compare all layers
print(f"\n{'='*70}")
print("ALL DATA LAYERS COMPARISON (aggregate Δ R²)")
print(f"{'='*70}")
print(f"   IMD 2019:       +0.0007")
print(f"   IMD 2025:       +0.0008")
print(f"   Demographics:   +0.0007")
print(f"   Weather:        +0.0025")
print(f"   POIs:           {delta_r2:+.4f}  ← NEW")
print("\n✅ POI fusion experiment complete!")
