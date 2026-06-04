"""
15_temporal_activity_fusion.py — Temporal Activity Indicators experiment.

Tests whether richer temporal encoding (beyond month_sin/cos) improves
crime prediction. Two indicators:
  - pct_holiday_days: fraction of days in school holidays (0.0–1.0)
  - pct_bank_holiday_days: fraction of days that are UK bank holidays

Source: London state school term dates + gov.uk bank holiday schedule

Key question: Do population-dynamic temporal features add predictive lift
beyond the existing sinusoidal month encoding?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import numpy as np
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

db = ThesisDB()

# ── 1. BUILD TEMPORAL ACTIVITY FEATURES ──
print("=" * 60)
print("TEMPORAL ACTIVITY INDICATORS — FEATURE ENGINEERING")
print("=" * 60)

# 1a. School holidays
holidays = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/london_school_holidays.csv",
                        parse_dates=['start_date', 'end_date'])
print(f"\n  School holiday periods: {len(holidays)}")

# 1b. Bank holidays
bank_hols = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/uk_bank_holidays.csv",
                          parse_dates=['date'])
print(f"  Bank holidays: {len(bank_hols)}")

# Compute features for each month
study_months = pd.date_range('2023-02-01', '2026-01-01', freq='MS')
temporal_features = []

for month_start in study_months:
    year, month = month_start.year, month_start.month
    days_in_month = monthrange(year, month)[1]
    month_end = pd.Timestamp(year, month, days_in_month)

    # School holiday days
    school_days = 0
    for _, h in holidays.iterrows():
        overlap_start = max(month_start, h['start_date'])
        overlap_end = min(month_end, h['end_date'])
        if overlap_start <= overlap_end:
            school_days += (overlap_end - overlap_start).days + 1

    # Bank holiday days
    bank_days = bank_hols[(bank_hols['date'] >= month_start) & 
                           (bank_hols['date'] <= month_end)].shape[0]

    year_month = f"{year}-{month:02d}"
    temporal_features.append({
        'year_month': year_month,
        'pct_holiday_days': round(school_days / days_in_month, 3),
        'pct_bank_holiday_days': round(bank_days / days_in_month, 3),
    })

tf_df = pd.DataFrame(temporal_features)
print(f"\n  {'Month':>8} | {'School':>6} | {'Bank':>5}")
print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*5}")
for _, row in tf_df.iterrows():
    s_bar = '█' * int(row['pct_holiday_days'] * 15)
    b_bar = '█' * int(row['pct_bank_holiday_days'] * 15)
    print(f"  {row['year_month']:>8} | {row['pct_holiday_days']:.3f} {s_bar:15s} | {row['pct_bank_holiday_days']:.3f} {b_bar}")

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

# Join temporal activity features
df = df.merge(tf_df, left_on='month', right_on='year_month', how='left').drop(columns=['year_month'])

base_features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                 'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                 'month_sin', 'month_cos', 'time_idx']

temporal_feat_names = ['pct_holiday_days', 'pct_bank_holiday_days']
fused_features = base_features + temporal_feat_names

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

# Baseline
rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base_features], train['crime_count'])
pred_base = rf_base.predict(test[base_features])
r2_base = r2_score(test['crime_count'], pred_base)
mae_base = mean_absolute_error(test['crime_count'], pred_base)

# Fused (baseline + temporal activity)
rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused_features], train['crime_count'])
pred_fused = rf_fused.predict(test[fused_features])
r2_fused = r2_score(test['crime_count'], pred_fused)
mae_fused = mean_absolute_error(test['crime_count'], pred_fused)

delta_r2 = r2_fused - r2_base
delta_mae = mae_fused - mae_base

print(f"\n  Baseline R²:            {r2_base:.4f}  (MAE: {mae_base:.3f})")
print(f"  + Temporal Activity R²: {r2_fused:.4f}  (MAE: {mae_fused:.3f})")
print(f"  Δ R²:                   {delta_r2:+.4f}")
print(f"  Δ MAE:                  {delta_mae:+.3f}")

imp = pd.Series(rf_fused.feature_importances_, index=fused_features).sort_values(ascending=False)
print(f"\n  Feature importance ranking:")
for feat, val in imp.items():
    marker = " ← TEMPORAL" if feat in temporal_feat_names else ""
    print(f"    {feat:25s} {val:.4f}{marker}")

# ── 3. ABLATION: school holidays only vs bank holidays only vs both ──
print("\n" + "=" * 60)
print("ABLATION: SCHOOL vs BANK vs BOTH")
print("=" * 60)

for label, feats in [
    ("School holidays only", base_features + ['pct_holiday_days']),
    ("Bank holidays only",   base_features + ['pct_bank_holiday_days']),
    ("Both combined",        fused_features),
]:
    rf_abl = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_abl.fit(train[feats], train['crime_count'])
    r2_abl = r2_score(test['crime_count'], rf_abl.predict(test[feats]))
    print(f"  {label:25s} | R²: {r2_abl:.4f} | Δ: {r2_abl - r2_base:+.4f}")

# ── 4. PER-CRIME-TYPE ──
print("\n" + "=" * 60)
print("PER-CRIME-TYPE FUSION (both temporal features)")
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

    ct_df = ct_df.merge(tf_df, left_on='month', right_on='year_month', how='left').drop(columns=['year_month'])

    ct_base_feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3', 'rolling_12', 'month_sin', 'month_cos']
    ct_fused_feats = ct_base_feats + temporal_feat_names

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

    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[ct_fused_feats], ct_train['crime_count'])
    r2_f = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[ct_fused_feats]))

    d = r2_f - r2_b
    results.append({'crime_type': ct, 'r2_base': r2_b, 'r2_fused': r2_f, 'delta': d})
    print(f"  {ct:30s} | Base: {r2_b:.4f} | +Temporal: {r2_f:.4f} | Δ: {d:+.4f}")

# ── 5. SUMMARY ──
print("\n" + "=" * 60)
print("SUMMARY (sorted by Δ R²)")
print("=" * 60)
res_df = pd.DataFrame(results).dropna().sort_values('delta', ascending=False)
for _, r in res_df.iterrows():
    print(f"  {r['crime_type']:30s} | Δ R² = {r['delta']:+.4f}")

print(f"\n  Top 3 beneficiaries:")
for _, r in res_df.head(3).iterrows():
    print(f"    {r['crime_type']:30s} Δ R² = {r['delta']:+.4f}")

# Noise floor
print(f"\n{'='*60}")
print("NOISE FLOOR IMPACT")
print(f"{'='*60}")
noise_floor = 3.79
print(f"  Original MAE:     {mae_base:.3f}")
print(f"  With temporal:    {mae_fused:.3f}")
print(f"  Gap before:       {mae_base - noise_floor:.3f} ({(mae_base/noise_floor - 1)*100:.1f}% above floor)")
print(f"  Gap after:        {mae_fused - noise_floor:.3f} ({(mae_fused/noise_floor - 1)*100:.1f}% above floor)")

db.close()
print("\n✅ Temporal activity indicators experiment complete!")
