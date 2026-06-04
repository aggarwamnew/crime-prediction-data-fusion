"""
18_household_fusion.py — Household composition fusion experiment.

Source: ONS Census 2021 — TS003 Household Composition
        LSOA-level household type breakdown
URL:    https://www.nomisweb.co.uk/output/census/2021/census2021-ts003.zip

Features (converted to percentages of total households):
  - pct_one_person: % single-person households (proxy for social isolation)
  - pct_one_person_66plus: % single elderly living alone
  - pct_lone_parent_dep: % lone parents with dependent children
  - pct_married_dep: % married couples with dependent children
  - pct_cohabiting: % cohabiting couples (all)
  - pct_other_household: % other household types (HMOs, students, etc.)

Experiment: Aggregate + per-crime-type impact of household composition.
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
print("HOUSEHOLD COMPOSITION (TS003) DATA EXTRACTION")
print("=" * 60)

csv_path = PROJECT_ROOT / "data/raw/london/census/ts003/census2021-ts003-lsoa.csv"
raw = pd.read_csv(csv_path)

# Build percentage features from key categories
hh = pd.DataFrame()
hh['lsoa_code'] = raw['geography code']
total_col = [c for c in raw.columns if 'Total' in c and 'measures' in c][0]
hh['total_hh'] = raw[total_col]

one_person_col = [c for c in raw.columns if 'One person household; measures' in c][0]
one_66_col = [c for c in raw.columns if 'One person household: Aged 66' in c][0]
lone_parent_dep_col = [c for c in raw.columns if 'Lone parent family: With dependent' in c][0]
married_dep_col = [c for c in raw.columns if 'Married or civil partnership couple: Dependent' in c][0]
cohabiting_col = [c for c in raw.columns if 'Cohabiting couple family; measures' in c][0]
other_hh_col = [c for c in raw.columns if 'Other household types; measures' in c][0]

hh['pct_one_person'] = raw[one_person_col] / hh['total_hh'] * 100
hh['pct_one_person_66plus'] = raw[one_66_col] / hh['total_hh'] * 100
hh['pct_lone_parent_dep'] = raw[lone_parent_dep_col] / hh['total_hh'] * 100
hh['pct_married_dep'] = raw[married_dep_col] / hh['total_hh'] * 100
hh['pct_cohabiting'] = raw[cohabiting_col] / hh['total_hh'] * 100
hh['pct_other_household'] = raw[other_hh_col] / hh['total_hh'] * 100

# Filter to London LSOAs
london_lsoas = db.query("SELECT DISTINCT lsoa_code FROM crime_clean")['lsoa_code'].tolist()
hh_london = hh[hh['lsoa_code'].isin(london_lsoas)].copy()

hh_features_cols = ['pct_one_person', 'pct_one_person_66plus', 'pct_lone_parent_dep',
                    'pct_married_dep', 'pct_cohabiting', 'pct_other_household']

print(f"  All England LSOAs: {len(hh):,}")
print(f"  London LSOAs matched: {len(hh_london):,}")
for f in hh_features_cols:
    print(f"  {f:30s}: mean {hh_london[f].mean():.1f}% (range {hh_london[f].min():.1f}-{hh_london[f].max():.1f}%)")

hh_for_merge = hh_london[['lsoa_code'] + hh_features_cols].copy()

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

df = df.merge(hh_for_merge, on='lsoa_code', how='left')
df_model = df.dropna()
print(f"  LSOAs with household data: {df_model['lsoa_code'].nunique():,} / {len(all_lsoas):,}")

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base_features], train['crime_count'])
r2_base = r2_score(test['crime_count'], rf_base.predict(test[base_features]))
mae_base = mean_absolute_error(test['crime_count'], rf_base.predict(test[base_features]))

fused_features = base_features + hh_features_cols
rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused_features], train['crime_count'])
r2_fused = r2_score(test['crime_count'], rf_fused.predict(test[fused_features]))
mae_fused = mean_absolute_error(test['crime_count'], rf_fused.predict(test[fused_features]))

delta = r2_fused - r2_base
print(f"\n  Baseline R²:      {r2_base:.4f}  (MAE: {mae_base:.2f})")
print(f"  + Household R²:   {r2_fused:.4f}  (MAE: {mae_fused:.2f})")
print(f"  Δ R²:             {delta:+.4f}")
print(f"  Δ MAE:            {mae_fused - mae_base:+.2f}")

imp = pd.Series(rf_fused.feature_importances_, index=fused_features).sort_values(ascending=False)
print(f"\n  Feature importances (household features):")
for f in hh_features_cols:
    print(f"    {f:30s}: {imp[f]:.4f} (rank {list(imp.index).index(f)+1}/{len(imp)})")

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
    ct_df = ct_df.merge(hh_for_merge, on='lsoa_code', how='left')
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

    ct_fused_feats = ct_base_feats + hh_features_cols
    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[ct_fused_feats], ct_train['crime_count'])
    r2_f = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[ct_fused_feats]))

    d = r2_f - r2_b
    results.append({'crime_type': ct, 'r2_base': r2_b, 'r2_fused': r2_f, 'delta': d})
    print(f"  {ct:30s} | Base: {r2_b:.4f} | +HH: {r2_f:.4f} | Δ: {d:+.4f}")

print("\n" + "=" * 60)
print("SUMMARY (sorted by Δ)")
print("=" * 60)
res_df = pd.DataFrame(results).dropna().sort_values('delta', ascending=False)
for _, r in res_df.iterrows():
    print(f"  {r['crime_type']:30s} | Δ R² = {r['delta']:+.4f}")

db.close()
print("\n✅ Household composition fusion experiment complete!")
