"""
14_housing_fusion.py — Housing data extraction + fusion experiment.

Source: ONS HPSSA Dataset 9 — Median price paid for all dwellings by LSOA
        Year ending December 2020
URL:    https://www.ons.gov.uk/visualisations/dvc1415/fig1/datadownload.xlsx

Features:
  - price_low: 1 if median price in bottom tercile (< ~£422.5K)
  - price_mid: 1 if median price in middle tercile
  - price_high: 1 if median price in top tercile (> ~£587.5K)
   
Experiment: Aggregate + per-crime-type impact of housing data.
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
print("HOUSING DATA EXTRACTION")
print("=" * 60)

xlsx_path = PROJECT_ROOT / "data/raw/london/housing/ons_house_prices_lsoa.xlsx"
raw = pd.read_excel(xlsx_path, skiprows=4, header=None,
                    names=['la_code', 'la_name', 'lsoa_code', 'lsoa_name', 'median_house_price'])

# Skip header row, filter London (E09 = London boroughs)
raw = raw[raw['la_code'].astype(str).str.startswith('E09')].copy()
raw['median_house_price'] = pd.to_numeric(raw['median_house_price'], errors='coerce')
housing = raw[['lsoa_code', 'median_house_price']].dropna().copy()

# Create tertile bands
thresholds = housing['median_house_price'].quantile([1/3, 2/3])
t_low, t_high = thresholds.iloc[0], thresholds.iloc[1]
housing['price_low'] = (housing['median_house_price'] < t_low).astype(int)
housing['price_mid'] = ((housing['median_house_price'] >= t_low) & (housing['median_house_price'] < t_high)).astype(int)
housing['price_high'] = (housing['median_house_price'] >= t_high).astype(int)

# Save cleaned CSV
out_path = PROJECT_ROOT / "data/raw/london/housing/housing_per_lsoa.csv"
housing.to_csv(out_path, index=False)
print(f"  London LSOAs with price data: {len(housing):,}")
print(f"  Price range: £{housing['median_house_price'].min():,.0f} – £{housing['median_house_price'].max():,.0f}")
print(f"  Tertile thresholds: Low < £{t_low:,.0f} | Mid £{t_low:,.0f}–£{t_high:,.0f} | High > £{t_high:,.0f}")
print(f"  Band counts: Low={housing['price_low'].sum()}, Mid={housing['price_mid'].sum()}, High={housing['price_high'].sum()}")
print(f"  Saved: {out_path}")

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

df = df.merge(housing, on='lsoa_code', how='left')
df_model = df.dropna()
print(f"  LSOAs with housing data: {df_model['lsoa_code'].nunique():,} / {len(all_lsoas):,}")

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

# Baseline (crime-only)
rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base_features], train['crime_count'])
r2_base = r2_score(test['crime_count'], rf_base.predict(test[base_features]))

# Fused (crime + housing tertile bands)
housing_features = ['price_low', 'price_mid', 'price_high']
fused_features = base_features + housing_features
rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused_features], train['crime_count'])
r2_fused = r2_score(test['crime_count'], rf_fused.predict(test[fused_features]))

delta = r2_fused - r2_base
print(f"\n  Baseline R²:  {r2_base:.4f}")
print(f"  + Housing R²: {r2_fused:.4f}")
print(f"  Δ R²:         {delta:+.4f}")

# Feature importance
imp = pd.Series(rf_fused.feature_importances_, index=fused_features).sort_values(ascending=False)
for f in housing_features:
    print(f"  {f} importance: {imp[f]:.4f} (rank {list(imp.index).index(f)+1}/{len(imp)})")

# ── 3. PER-CRIME-TYPE ──
print("\n" + "=" * 60)
print("PER-CRIME-TYPE FUSION")
print("=" * 60)

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
results = []

for ct in crime_types:
    ct_data = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct}' GROUP BY lsoa_code, month")

    # Inner join with housing first (consistent with script 11 pattern)
    ct_data = ct_data.merge(housing[['lsoa_code'] + housing_features], on='lsoa_code', how='inner')
    ct_totals = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_totals[ct_totals >= 12].index
    if len(ct_active) < 50:
        results.append({'crime_type': ct, 'r2_base': None, 'r2_fused': None, 'delta': None})
        continue

    ct_lsoas = sorted(ct_active)
    ct_months = sorted(ct_data['month'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])

    housing_per = ct_data.drop_duplicates('lsoa_code')[['lsoa_code'] + housing_features].set_index('lsoa_code')

    ct_df = ct_data[['lsoa_code','month','crime_count']].set_index(['lsoa_code', 'month']).reindex(ct_grid, fill_value=0).reset_index()
    ct_df = ct_df.merge(housing_per, on='lsoa_code', how='left')
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

    ct_fused_feats = ct_base_feats + housing_features
    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[ct_fused_feats], ct_train['crime_count'])
    r2_f = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[ct_fused_feats]))

    d = r2_f - r2_b
    results.append({'crime_type': ct, 'r2_base': r2_b, 'r2_fused': r2_f, 'delta': d})
    print(f"  {ct:30s} | Base: {r2_b:.4f} | +Housing: {r2_f:.4f} | Δ: {d:+.4f}")

print("\n" + "=" * 60)
print("SUMMARY (sorted by Δ)")
print("=" * 60)
res_df = pd.DataFrame(results).dropna().sort_values('delta', ascending=False)
for _, r in res_df.iterrows():
    print(f"  {r['crime_type']:30s} | Δ R² = {r['delta']:+.4f}")

db.close()
print("\n✅ Housing fusion experiment complete!")
