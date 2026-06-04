"""
16_mental_health_fusion.py — SAMHI mental health index fusion experiment.

Source: PLDR Small Area Mental Health Index (SAMHI) v5.00
        LSOA-level composite mental health score, 2011–2022
URL:    https://pldr.org/dataset/small-area-mental-health-index-samhi-2noyv

Features:
  - samhi_index: composite mental health index (higher = worse mental health)
  - samhi_decile: decile rank (1=best, 10=worst)

Experiment: Aggregate + per-crime-type impact of mental health data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

db = ThesisDB()

# ── 1. EXTRACT & CLEAN ──
print("=" * 60)
print("MENTAL HEALTH (SAMHI) DATA EXTRACTION")
print("=" * 60)

csv_path = PROJECT_ROOT / "data/raw/london/mental_health/samhi_lsoa.csv"
raw = pd.read_csv(csv_path)

# Use latest year (2022) — columns: lsoa11, samhi_index.2022, samhi_dec.2022
raw = raw[['lsoa11', 'samhi_index.2022', 'samhi_dec.2022']].copy()
raw.columns = ['lsoa_code', 'samhi_index', 'samhi_decile']

# Filter to London LSOAs (E01 codes that appear in our crime data)
london_lsoas = db.query("SELECT DISTINCT lsoa_code FROM crime_clean")['lsoa_code'].tolist()

# SAMHI uses LSOA 2011 codes; our crime data uses LSOA 2011 codes too
samhi = raw[raw['lsoa_code'].isin(london_lsoas)].dropna().copy()

print(f"  All England LSOAs: {len(raw):,}")
print(f"  London LSOAs matched: {len(samhi):,}")
print(f"  SAMHI index range: {samhi['samhi_index'].min():.3f} to {samhi['samhi_index'].max():.3f}")
print(f"  SAMHI index mean: {samhi['samhi_index'].mean():.3f}")
print(f"  Decile distribution:\n{samhi['samhi_decile'].value_counts().sort_index().to_string()}")

# ── 2. AGGREGATE FUSION ──
print("\n" + "=" * 60)
print("AGGREGATE FUSION EXPERIMENT")
print("=" * 60)

crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]

all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = crime.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year - ts.dt.year.min()) * 12 + ts.dt.month

base_features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                 'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                 'month_sin', 'month_cos', 'time_idx']

df = df.merge(samhi, on='lsoa_code', how='left')
df_model = df.dropna()
print(f"  LSOAs with SAMHI data: {df_model['lsoa_code'].nunique():,} / {len(all_lsoas):,}")

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

# Baseline (crime-only)
rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base_features], train['crime_count'])
r2_base = r2_score(test['crime_count'], rf_base.predict(test[base_features]))
mae_base = mean_absolute_error(test['crime_count'], rf_base.predict(test[base_features]))

# Fused (crime + SAMHI index)
samhi_features = ['samhi_index', 'samhi_decile']
fused_features = base_features + samhi_features
rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused_features], train['crime_count'])
r2_fused = r2_score(test['crime_count'], rf_fused.predict(test[fused_features]))
mae_fused = mean_absolute_error(test['crime_count'], rf_fused.predict(test[fused_features]))

delta = r2_fused - r2_base
print(f"\n  Baseline R²:  {r2_base:.4f}  (MAE: {mae_base:.2f})")
print(f"  + SAMHI R²:   {r2_fused:.4f}  (MAE: {mae_fused:.2f})")
print(f"  Δ R²:         {delta:+.4f}")
print(f"  Δ MAE:        {mae_fused - mae_base:+.2f}")

# Feature importance
imp = pd.Series(rf_fused.feature_importances_, index=fused_features).sort_values(ascending=False)
print(f"\n  Feature importances (SAMHI features):")
for f in samhi_features:
    print(f"    {f}: {imp[f]:.4f} (rank {list(imp.index).index(f)+1}/{len(imp)})")

# ── 3. PER-CRIME-TYPE ──
print("\n" + "=" * 60)
print("PER-CRIME-TYPE FUSION")
print("=" * 60)

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
results = []

for ct in crime_types:
    ct_data = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct}' GROUP BY lsoa_code, month")
    ct_totals = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_totals[ct_totals >= 12].index
    if len(ct_active) < 50:
        results.append({'crime_type': ct, 'r2_base': None, 'r2_fused': None, 'delta': None})
        continue

    ct_lsoas = sorted(ct_active)
    ct_months = sorted(ct_data['month'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])
    ct_df = ct_data.set_index(['lsoa_code', 'month']).reindex(ct_grid, fill_value=0).reset_index()
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rolling_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rolling_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_base_feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3', 'rolling_12', 'month_sin', 'month_cos']
    ct_df = ct_df.merge(samhi, on='lsoa_code', how='left')
    ct_model = ct_df.dropna()

    ct_test_months = ct_months[-6:]
    ct_train = ct_model[~ct_model['month'].isin(ct_test_months)]
    ct_test = ct_model[ct_model['month'].isin(ct_test_months)]

    if len(ct_test) < 50:
        results.append({'crime_type': ct, 'r2_base': None, 'r2_fused': None, 'delta': None})
        continue

    rf_b = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_b.fit(ct_train[ct_base_feats], ct_train['crime_count'])
    r2_b = r2_score(ct_test['crime_count'], rf_b.predict(ct_test[ct_base_feats]))

    ct_fused_feats = ct_base_feats + samhi_features
    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[ct_fused_feats], ct_train['crime_count'])
    r2_f = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[ct_fused_feats]))

    d = r2_f - r2_b
    results.append({'crime_type': ct, 'r2_base': r2_b, 'r2_fused': r2_f, 'delta': d})
    print(f"  {ct:30s} | Base: {r2_b:.4f} | +SAMHI: {r2_f:.4f} | Δ: {d:+.4f}")

print("\n" + "=" * 60)
print("SUMMARY (sorted by Δ)")
print("=" * 60)
res_df = pd.DataFrame(results).dropna().sort_values('delta', ascending=False)
for _, r in res_df.iterrows():
    print(f"  {r['crime_type']:30s} | Δ R² = {r['delta']:+.4f}")

db.close()
print("\n✅ Mental health (SAMHI) fusion experiment complete!")
