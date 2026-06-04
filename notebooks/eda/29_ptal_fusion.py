"""
Script 29: PTAL Fusion — Static Transport Accessibility
Tests whether Public Transport Accessibility Level improves crime prediction.
PTAL is already at LSOA level, so no spatial engineering needed.
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
print("EXPERIMENT: PTAL FUSION (Static Transport Accessibility)")
print("=" * 70)

# 1. Load PTAL
print("\n1. Loading PTAL 2023...")
ptal = pd.read_csv(PROJECT_ROOT / "data/raw/london/transport/ptal/ptal_lsoa_2023.csv")
print(f"   PTAL rows: {len(ptal):,}")
print(f"   Columns: {list(ptal.columns)}")

# Select useful features: mean_AI (continuous accessibility index), MEAN_PTAL_ (categorical 1a-6b)
# Convert PTAL category to numeric for the model
ptal_cat_map = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
ptal['ptal_category_num'] = ptal['MEAN_PTAL_'].map(ptal_cat_map)

ptal_features_df = ptal[['LSOA21CD', 'mean_AI', 'MEDIAN_AI', 'MIN_AI', 'MAX_AI', 'ptal_category_num']].copy()
ptal_features_df = ptal_features_df.rename(columns={
    'LSOA21CD': 'lsoa_code',
    'mean_AI': 'ptal_mean_ai',
    'MEDIAN_AI': 'ptal_median_ai',
    'MIN_AI': 'ptal_min_ai',
    'MAX_AI': 'ptal_max_ai',
})
ptal_feature_cols = ['ptal_mean_ai', 'ptal_median_ai', 'ptal_min_ai', 'ptal_max_ai', 'ptal_category_num']
print(f"   PTAL features: {ptal_feature_cols}")
print(f"   PTAL category distribution:")
print(ptal['MEAN_PTAL_'].value_counts().sort_index().to_string())

# 2. Load crime
print("\n2. Loading crime data...")
monthly = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")
crime_lsoas = set(monthly['lsoa_code'].unique())
ptal_lsoas = set(ptal_features_df['lsoa_code'].unique())
match = len(crime_lsoas & ptal_lsoas)
print(f"   Crime LSOAs: {len(crime_lsoas):,}")
print(f"   PTAL LSOAs: {len(ptal_lsoas):,}")
print(f"   Match: {match:,} ({match/len(crime_lsoas)*100:.1f}%)")

# LSOAs without PTAL get 0 (they are outside Greater London TfL coverage)
all_crime_lsoas = pd.DataFrame({'lsoa_code': list(crime_lsoas)})
ptal_full = all_crime_lsoas.merge(ptal_features_df, on='lsoa_code', how='left').fillna(0)

# ── AGGREGATE EXPERIMENT ──
print(f"\n{'='*70}")
print("AGGREGATE FUSION")
print(f"{'='*70}")

merged = monthly.merge(ptal_full, on='lsoa_code', how='left')
merged[ptal_feature_cols] = merged[ptal_feature_cols].fillna(0)

MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
n_lsoas = merged['lsoa_code'].nunique()
print(f"   Active LSOAs (>={MIN_CRIMES} crimes): {n_lsoas:,}")

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

ptal_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + ptal_feature_cols].set_index('lsoa_code')
crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()
df = df.merge(ptal_per, on='lsoa_code', how='left')
df[ptal_feature_cols] = df[ptal_feature_cols].fillna(0)

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
fused_cols = lag_features + temporal_features + ptal_feature_cols

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"   Crime-only features: {len(crime_only_cols)}")
print(f"   Fused features: {len(fused_cols)} (+{len(ptal_feature_cols)} PTAL)")

rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])

results = {}
for name, model, cols in [("Crime-only", rf_crime, crime_only_cols), ("Fused (+PTAL)", rf_fused, fused_cols)]:
    y_pred = model.predict(test_df[cols])
    r2 = r2_score(test_df['crime_count'], y_pred)
    mae = mean_absolute_error(test_df['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"\n   {name}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

delta_r2 = results["Fused (+PTAL)"]['R²'] - results["Crime-only"]['R²']
print(f"\n   Δ R² = {delta_r2:+.4f}")

# Feature importance
print("\n   Feature importance (PTAL features):")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
ptal_imp = imp[imp['feature'].isin(ptal_feature_cols)]
for _, row in ptal_imp.iterrows():
    print(f"     {row['feature']:25s} {row['importance']:.4f}")

# ── PER-CRIME-TYPE ──
print(f"\n{'='*70}")
print("PER-CRIME-TYPE PTAL ANALYSIS")
print(f"{'='*70}")

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
per_type_results = []

for ct in crime_types:
    print(f"\n  {ct}...", end=" ", flush=True)
    ct_monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")
    ct_merged = ct_monthly.merge(ptal_full, on='lsoa_code', how='left')
    ct_merged[ptal_feature_cols] = ct_merged[ptal_feature_cols].fillna(0)

    lsoa_totals = ct_merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= 12].index
    if len(active) < 50: print("SKIP"); continue
    ct_merged = ct_merged[ct_merged['lsoa_code'].isin(active)]

    all_months = sorted(ct_merged['month'].unique())
    all_lsoas = sorted(ct_merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
    ptal_per = ct_merged.drop_duplicates('lsoa_code')[['lsoa_code'] + ptal_feature_cols].set_index('lsoa_code')
    ct_df = ct_merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
    ct_df = ct_df.merge(ptal_per, on='lsoa_code', how='left')
    ct_df[ptal_feature_cols] = ct_df[ptal_feature_cols].fillna(0)
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

    rf_ptal = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_ptal.fit(train[lag_cols + ptal_feature_cols], train['crime_count'])
    r2_ptal = r2_score(test['crime_count'], rf_ptal.predict(test[lag_cols + ptal_feature_cols]))

    delta = r2_ptal - r2_base

    ct_imp = pd.DataFrame({'feature': lag_cols + ptal_feature_cols, 'importance': rf_ptal.feature_importances_})
    top_ptal = ct_imp[ct_imp['feature'].isin(ptal_feature_cols)].sort_values('importance', ascending=False).iloc[0]

    print(f"Δ={delta:+.4f}  top={top_ptal['feature']} ({top_ptal['importance']:.4f})")
    per_type_results.append({'crime_type': ct, 'r2_base': r2_base, 'r2_ptal': r2_ptal, 'delta': delta, 'top_feature': top_ptal['feature'], 'n_lsoas': len(active)})

db.close()

res = pd.DataFrame(per_type_results).sort_values('delta', ascending=False)
print(f"\n{'='*90}")
print("SUMMARY: PTAL IMPACT BY CRIME TYPE")
print(f"{'='*90}")
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ PTAL':>8} {'Top Feature':>20}")
print("-" * 75)
for _, r in res.iterrows():
    print(f"{r['crime_type']:<35} {r['r2_base']:>8.4f} {r['delta']:>+8.4f} {r['top_feature']:>20}")

print(f"\n   Aggregate Δ R² = {delta_r2:+.4f}")
print("\n✅ PTAL fusion experiment complete!")
