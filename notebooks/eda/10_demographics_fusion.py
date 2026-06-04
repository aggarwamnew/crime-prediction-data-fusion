"""
Experiment 8: Demographics Fusion
Tests whether Census 2021 demographics (population density + age structure)
improve crime prediction beyond the crime-only baseline.
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

CENSUS_DIR = PROJECT_ROOT / "data/raw/london/census"
db = ThesisDB()
print("=" * 70)
print("EXPERIMENT: DEMOGRAPHICS FUSION (Census 2021)")
print("=" * 70)

# ── 1. Load population density (TS006) ──
print("\n1. Loading Census 2021 TS006 (population density)...")
ts006 = pd.read_csv(CENSUS_DIR / "ts006/census2021-ts006-lsoa.csv")
ts006 = ts006.rename(columns={
    'geography code': 'lsoa_code',
    'Population Density: Persons per square kilometre; measures: Value': 'pop_density'
})[['lsoa_code', 'pop_density']]
print(f"   TS006: {len(ts006):,} LSOAs, pop_density range: {ts006['pop_density'].min():.0f} – {ts006['pop_density'].max():.0f}")

# ── 2. Load age structure (TS007a) ──
print("\n2. Loading Census 2021 TS007a (age bands)...")
ts007 = pd.read_csv(CENSUS_DIR / "ts007a/census2021-ts007a-lsoa.csv")
ts007 = ts007.rename(columns={'geography code': 'lsoa_code'})

# Derive meaningful proportions rather than raw counts
total_col = 'Age: Total'
ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007[total_col]
ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007[total_col]
ts007['pct_25_44'] = (ts007['Age: Aged 25 to 29 years'] + ts007['Age: Aged 30 to 34 years'] + ts007['Age: Aged 35 to 39 years'] + ts007['Age: Aged 40 to 44 years']) / ts007[total_col]
ts007['pct_45_64'] = (ts007['Age: Aged 45 to 49 years'] + ts007['Age: Aged 50 to 54 years'] + ts007['Age: Aged 55 to 59 years'] + ts007['Age: Aged 60 to 64 years']) / ts007[total_col]
ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007[total_col]
ts007['total_pop'] = ts007[total_col]

demo_cols_from_ts007 = ['lsoa_code', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']
ts007_slim = ts007[demo_cols_from_ts007]
print(f"   TS007a: {len(ts007_slim):,} LSOAs, derived 5 age-band proportions + total_pop")

# ── 3. Merge demographics ──
print("\n3. Merging demographics...")
demo = ts006.merge(ts007_slim, on='lsoa_code', how='inner')
demo_features = ['pop_density', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']
print(f"   Combined: {len(demo):,} LSOAs with {len(demo_features)} demographic features")
print(f"   Features: {demo_features}")

# ── 4. Load crime data ──
print("\n4. Loading crime data...")
monthly = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")
crime_lsoas = set(monthly['lsoa_code'].unique())
demo_lsoas = set(demo['lsoa_code'].unique())
match = len(crime_lsoas & demo_lsoas)
print(f"   Crime LSOAs: {len(crime_lsoas):,}")
print(f"   Demo LSOAs:  {len(demo_lsoas):,}")
print(f"   Matched:     {match:,}/{len(crime_lsoas):,} ({match/len(crime_lsoas)*100:.1f}%)")

# ── 5. Join and build features ──
print("\n5. Building feature matrix...")
merged = monthly.merge(demo, on='lsoa_code', how='left')
merged = merged.dropna(subset=['pop_density'])

MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
n_lsoas = merged['lsoa_code'].nunique()
print(f"   Active LSOAs (≥{MIN_CRIMES} crimes): {n_lsoas:,}")

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

demo_per_lsoa = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + demo_features].set_index('lsoa_code')

crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()
df = df.merge(demo_per_lsoa, on='lsoa_code', how='left')

# Lag features
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

# ── 6. Train/test split ──
print("\n6. Train/test split...")
df_model = df.dropna().copy()
test_months = all_months[-6:]
train_df = df_model[~df_model['month'].isin(test_months)]
test_df = df_model[df_model['month'].isin(test_months)]

lag_features = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
temporal_features = ['month_sin', 'month_cos', 'time_idx']

crime_only_cols = lag_features + temporal_features
fused_cols = lag_features + temporal_features + demo_features

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"   Crime-only features: {len(crime_only_cols)}")
print(f"   Fused features: {len(fused_cols)} (+{len(demo_features)} demographics)")

# ── 7. Train models ──
print("\n7. Training models...")
rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])
print("   ✅ Crime-only RF trained")

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])
print("   ✅ Fused RF (crime + demographics) trained")

# ── 8. Evaluate ──
print("\n" + "=" * 70)
print("8. RESULTS")
print("=" * 70)

results = {}
for name, model, cols in [("Crime-only", rf_crime, crime_only_cols),
                           ("Fused (+Demo)", rf_fused, fused_cols)]:
    y_pred = model.predict(test_df[cols])
    r2 = r2_score(test_df['crime_count'], y_pred)
    mae = mean_absolute_error(test_df['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"\n   {name}:")
    print(f"     R²:   {r2:.4f}")
    print(f"     MAE:  {mae:.4f}")
    print(f"     RMSE: {rmse:.4f}")

delta_r2 = results["Fused (+Demo)"]['R²'] - results["Crime-only"]['R²']
delta_mae = results["Crime-only"]['MAE'] - results["Fused (+Demo)"]['MAE']
print(f"\n   Δ R² = {delta_r2:+.4f}")
print(f"   Δ MAE = {delta_mae:+.4f}")

# ── 9. Feature importance ──
print("\n9. Feature importance (fused model):")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
for _, row in imp.iterrows():
    marker = " ← DEMO" if row['feature'] in demo_features else ""
    print(f"   {row['feature']:30s} {row['importance']:.4f}{marker}")

# ── 10. Compare with IMD and Weather ──
print("\n" + "=" * 70)
print("10. COMPARISON WITH OTHER DATA LAYERS")
print("=" * 70)
print(f"\n   Data Layer        Δ R² (from previous experiments)")
print(f"   {'─'*50}")
print(f"   IMD 2019          +0.0007")
print(f"   IMD 2025          +0.0008")
print(f"   Weather           +0.0025")
print(f"   Demographics      {delta_r2:+.4f}  ← NEW")

db.close()
print("\n✅ Demographics fusion experiment complete!")
