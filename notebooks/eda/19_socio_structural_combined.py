"""
19_socio_structural_combined.py — Combined socio-structural indicators experiment.

Tests: SAMHI + Education (TS067) + Household (TS003) together.
Question: Do these layers add to each other or overlap?

Compares:
  - Baseline (crime-only)
  - + SAMHI only
  - + Education only
  - + Household only
  - + All 3 combined
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

# ── 1. LOAD ALL 3 DATASETS ──
print("=" * 60)
print("LOADING SOCIO-STRUCTURAL DATASETS")
print("=" * 60)

# SAMHI
samhi_raw = pd.read_csv(PROJECT_ROOT / "data/raw/london/mental_health/samhi_lsoa.csv")
samhi = samhi_raw[['lsoa11', 'samhi_index.2022', 'samhi_dec.2022']].copy()
samhi.columns = ['lsoa_code', 'samhi_index', 'samhi_decile']
print(f"  SAMHI: {len(samhi):,} LSOAs")

# Education (TS067)
edu_raw = pd.read_csv(PROJECT_ROOT / "data/raw/london/census/ts067/census2021-ts067-lsoa.csv")
total_col = [c for c in edu_raw.columns if 'Total' in c][0]
edu = pd.DataFrame()
edu['lsoa_code'] = edu_raw['geography code']
edu['total_16plus'] = edu_raw[total_col]
for orig, short in [('No qualifications', 'pct_no_qual'),
                     ('Level 1 and entry level', 'pct_level1'),
                     ('Level 2 qualifications', 'pct_level2'),
                     ('Apprenticeship', 'pct_apprentice'),
                     ('Level 3 qualifications', 'pct_level3'),
                     ('Level 4 qualifications and above', 'pct_level4_plus'),
                     ('Other qualifications', 'pct_other_qual')]:
    col = [c for c in edu_raw.columns if orig in c][0]
    edu[short] = edu_raw[col] / edu['total_16plus'] * 100
print(f"  Education: {len(edu):,} LSOAs")

# Household (TS003)
hh_raw = pd.read_csv(PROJECT_ROOT / "data/raw/london/census/ts003/census2021-ts003-lsoa.csv")
hh = pd.DataFrame()
hh['lsoa_code'] = hh_raw['geography code']
hh_total_col = [c for c in hh_raw.columns if 'Total' in c][0]
hh['total_hh'] = hh_raw[hh_total_col]
for pattern, name in [('One person household; measures', 'pct_one_person'),
                       ('One person household: Aged 66', 'pct_one_person_66plus'),
                       ('Lone parent family: With dependent', 'pct_lone_parent_dep'),
                       ('Married or civil partnership couple: Dependent', 'pct_married_dep'),
                       ('Cohabiting couple family; measures', 'pct_cohabiting'),
                       ('Other household types; measures', 'pct_other_household')]:
    col = [c for c in hh_raw.columns if pattern in c][0]
    hh[name] = hh_raw[col] / hh['total_hh'] * 100
print(f"  Household: {len(hh):,} LSOAs")

# Feature lists
samhi_feats = ['samhi_index', 'samhi_decile']
edu_feats = ['pct_no_qual', 'pct_level1', 'pct_level2', 'pct_apprentice', 'pct_level3', 'pct_level4_plus', 'pct_other_qual']
hh_feats = ['pct_one_person', 'pct_one_person_66plus', 'pct_lone_parent_dep', 'pct_married_dep', 'pct_cohabiting', 'pct_other_household']
all_ss_feats = samhi_feats + edu_feats + hh_feats

# ── 2. BUILD BASE DATAFRAME ──
print("\n" + "=" * 60)
print("AGGREGATE FUSION — COMBINED vs INDIVIDUAL")
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

# Merge all
df = df.merge(samhi[['lsoa_code'] + samhi_feats], on='lsoa_code', how='left')
df = df.merge(edu[['lsoa_code'] + edu_feats], on='lsoa_code', how='left')
df = df.merge(hh[['lsoa_code'] + hh_feats], on='lsoa_code', how='left')
df_model = df.dropna()

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]
print(f"  LSOAs with all 3 datasets: {df_model['lsoa_code'].nunique():,} / {len(all_lsoas):,}")

# Run all configurations
configs = [
    ('Baseline', base_features),
    ('+ SAMHI only', base_features + samhi_feats),
    ('+ Education only', base_features + edu_feats),
    ('+ Household only', base_features + hh_feats),
    ('+ All 3 combined', base_features + all_ss_feats),
]

agg_results = []
for name, feats in configs:
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(train[feats], train['crime_count'])
    r2 = r2_score(test['crime_count'], rf.predict(test[feats]))
    mae = mean_absolute_error(test['crime_count'], rf.predict(test[feats]))
    agg_results.append({'config': name, 'r2': r2, 'mae': mae})

r2_base = agg_results[0]['r2']
print(f"\n  {'Configuration':25s} | R²      | Δ R²     | MAE")
print(f"  {'-'*25} | {'-'*7} | {'-'*8} | {'-'*5}")
for r in agg_results:
    delta = r['r2'] - r2_base
    print(f"  {r['config']:25s} | {r['r2']:.4f}  | {delta:+.4f}  | {r['mae']:.2f}")

# Feature importance for combined model
rf_combined = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_combined.fit(train[base_features + all_ss_feats], train['crime_count'])
imp = pd.Series(rf_combined.feature_importances_, index=base_features + all_ss_feats).sort_values(ascending=False)
print(f"\n  Top 10 features (combined model):")
for i, (f, v) in enumerate(imp.head(10).items()):
    print(f"    {i+1}. {f:25s}: {v:.4f}")

# ── 3. PER-CRIME-TYPE (combined only) ──
print("\n" + "=" * 60)
print("PER-CRIME-TYPE — ALL 3 COMBINED")
print("=" * 60)

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
results = []

for ct in crime_types:
    ct_data = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct}' GROUP BY lsoa_code, month")

    # Inner join with all SS datasets first (consistent with script 11 pattern)
    ct_data = ct_data.merge(samhi[['lsoa_code'] + samhi_feats], on='lsoa_code', how='inner')
    ct_data = ct_data.merge(edu[['lsoa_code'] + edu_feats], on='lsoa_code', how='inner')
    ct_data = ct_data.merge(hh[['lsoa_code'] + hh_feats], on='lsoa_code', how='inner')
    ct_totals = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_totals[ct_totals >= 12].index
    if len(ct_active) < 50:
        results.append({'crime_type': ct, 'r2_base': None, 'r2_fused': None, 'delta': None})
        continue

    ct_lsoas = sorted(ct_active)
    ct_months = sorted(ct_data['month'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])

    ss_per = ct_data.drop_duplicates('lsoa_code')[['lsoa_code'] + all_ss_feats].set_index('lsoa_code')

    ct_df = ct_data[['lsoa_code','month','crime_count']].set_index(['lsoa_code', 'month']).reindex(ct_grid, fill_value=0).reset_index()
    ct_df = ct_df.merge(ss_per, on='lsoa_code', how='left')
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rolling_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rolling_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_base_feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3', 'rolling_12', 'month_sin', 'month_cos']
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

    ct_combined = ct_base_feats + all_ss_feats
    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[ct_combined], ct_train['crime_count'])
    r2_f = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[ct_combined]))

    d = r2_f - r2_b
    results.append({'crime_type': ct, 'r2_base': r2_b, 'r2_fused': r2_f, 'delta': d})
    print(f"  {ct:30s} | Base: {r2_b:.4f} | +SS: {r2_f:.4f} | Δ: {d:+.4f}")

print("\n" + "=" * 60)
print("SUMMARY (sorted by Δ)")
print("=" * 60)
res_df = pd.DataFrame(results).dropna().sort_values('delta', ascending=False)
for _, r in res_df.iterrows():
    print(f"  {r['crime_type']:30s} | Δ R² = {r['delta']:+.4f}")

# Additivity check
print("\n" + "=" * 60)
print("ADDITIVITY CHECK")
print("=" * 60)
sum_individual = (agg_results[1]['r2'] - r2_base) + (agg_results[2]['r2'] - r2_base) + (agg_results[3]['r2'] - r2_base)
combined_delta = agg_results[4]['r2'] - r2_base
print(f"  Sum of individual Δ R²:  {sum_individual:+.4f}")
print(f"  Combined Δ R²:           {combined_delta:+.4f}")
if combined_delta < sum_individual:
    print(f"  → SUBADDITIVE (overlap): combined is {sum_individual - combined_delta:.4f} less than sum")
else:
    print(f"  → SUPERADDITIVE (synergy): combined is {combined_delta - sum_individual:.4f} more than sum")

db.close()
print("\n✅ Combined socio-structural experiment complete!")
