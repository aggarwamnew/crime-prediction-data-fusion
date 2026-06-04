"""
Experiment 7: IMD 2019 vs 2025 Robustness Check
Runs the same fusion experiment with both IMD editions to check consistency.
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
print("IMD 2019 vs 2025 ROBUSTNESS CHECK")
print("=" * 70)

# ── Load IMD 2019 ──
print("\n1. Loading IMD 2019...")
imd19 = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
imd19_cols = {
    'LSOA code (2011)': 'lsoa_code',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Income Score (rate)': 'income_score',
    'Employment Score (rate)': 'employment_score',
    'Education, Skills and Training Score': 'education_score',
    'Health Deprivation and Disability Score': 'health_score',
    'Crime Score': 'crime_deprivation_score',
    'Barriers to Housing and Services Score': 'housing_score',
    'Living Environment Score': 'living_environment_score',
    'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': 'imd_decile',
}
imd19_slim = imd19[list(imd19_cols.keys())].rename(columns=imd19_cols)
print(f"   IMD 2019: {len(imd19_slim):,} LSOAs (LSOA 2011 codes)")

# ── Load IMD 2025 ──
print("\n2. Loading IMD 2025...")
imd25 = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
# Map equivalent columns (names slightly different in 2025)
print(f"   IMD 2025 columns: {list(imd25.columns[:10])}")

imd25_cols = {
    'LSOA code (2021)': 'lsoa_code',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Income Score (rate)': 'income_score',
    'Employment Score (rate)': 'employment_score',
    'Health Deprivation and Disability Score': 'health_score',
    'Crime Score': 'crime_deprivation_score',
    'Barriers to Housing and Services Score': 'housing_score',
    'Living Environment Score': 'living_environment_score',
    'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': 'imd_decile',
}

# Education column name has linebreak in 2025 CSV - find it
edu_col = [c for c in imd25.columns if 'Education' in c and 'Score' in c]
if edu_col:
    imd25_cols[edu_col[0]] = 'education_score'
    print(f"   Education column found: '{edu_col[0]}'")

available_cols = {k: v for k, v in imd25_cols.items() if k in imd25.columns}
print(f"   Matched {len(available_cols)}/{len(imd25_cols)} columns")

imd25_slim = imd25[list(available_cols.keys())].rename(columns=available_cols)
print(f"   IMD 2025: {len(imd25_slim):,} LSOAs (LSOA 2021 codes)")

# ── Crime data ──
print("\n3. Loading crime data...")
monthly = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")
crime_lsoas = monthly['lsoa_code'].nunique()
print(f"   Crime LSOAs: {crime_lsoas:,}")

# ── Test match rates ──
print("\n4. Match rate comparison:")
crime_codes = set(monthly['lsoa_code'].unique())
imd19_codes = set(imd19_slim['lsoa_code'].unique())
imd25_codes = set(imd25_slim['lsoa_code'].unique())

match_19 = len(crime_codes & imd19_codes)
match_25 = len(crime_codes & imd25_codes)
print(f"   IMD 2019: {match_19:,}/{len(crime_codes):,} LSOAs matched ({match_19/len(crime_codes)*100:.1f}%)")
print(f"   IMD 2025: {match_25:,}/{len(crime_codes):,} LSOAs matched ({match_25/len(crime_codes)*100:.1f}%)")
print(f"   Improvement: +{match_25 - match_19:,} LSOAs recovered")

# ── Run fusion experiment for both ──
imd_feature_names = ['imd_score', 'income_score', 'employment_score', 'education_score',
                     'health_score', 'crime_deprivation_score', 'housing_score',
                     'living_environment_score', 'imd_decile']

# Common features available in both
imd25_available_features = [f for f in imd_feature_names if f in imd25_slim.columns]
# Use only features present in BOTH for fair comparison
common_features = [f for f in imd_feature_names if f in imd25_slim.columns and f in imd19_slim.columns]
print(f"\n   Common IMD features for comparison: {common_features}")

results_all = {}

for version, imd_df, join_col_name in [("IMD 2019", imd19_slim, "lsoa_code"),
                                        ("IMD 2025", imd25_slim, "lsoa_code")]:
    print(f"\n{'='*70}")
    print(f"Running experiment: {version}")
    print(f"{'='*70}")

    # Join
    merged = monthly.merge(imd_df, on='lsoa_code', how='left')
    merged = merged.dropna(subset=['imd_score'])

    # Filter active LSOAs
    MIN_CRIMES = 36
    lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
    merged = merged[merged['lsoa_code'].isin(active)]
    n_lsoas = merged['lsoa_code'].nunique()
    print(f"   Active LSOAs (≥{MIN_CRIMES} crimes): {n_lsoas:,}")

    # Build grid
    all_months = sorted(merged['month'].unique())
    all_lsoas = sorted(merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

    imd_per_lsoa = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + common_features].set_index('lsoa_code')

    crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
    df = crime_grid.reindex(grid, fill_value=0).reset_index()
    df = df.merge(imd_per_lsoa, on='lsoa_code', how='left')

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

    # Split
    df_model = df.dropna().copy()
    test_months = all_months[-6:]
    train_df = df_model[~df_model['month'].isin(test_months)]
    test_df = df_model[df_model['month'].isin(test_months)]

    lag_features = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
    temporal_features = ['month_sin', 'month_cos', 'time_idx']
    crime_only_cols = lag_features + temporal_features
    fused_cols = lag_features + temporal_features + common_features

    # Train
    rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])

    rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_fused.fit(train_df[fused_cols], train_df['crime_count'])

    # Evaluate
    for name, model, cols in [("Crime-only", rf_crime, crime_only_cols), ("Fused", rf_fused, fused_cols)]:
        y_pred = model.predict(test_df[cols])
        r2 = r2_score(test_df['crime_count'], y_pred)
        mae = mean_absolute_error(test_df['crime_count'], y_pred)
        rmse = np.sqrt(mean_squared_error(test_df['crime_count'], y_pred))
        results_all[f"{version} {name}"] = {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'LSOAs': n_lsoas}
        print(f"   {name}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    delta = results_all[f"{version} Fused"]['R²'] - results_all[f"{version} Crime-only"]['R²']
    print(f"   Δ R² ({version}) = {delta:+.4f}")

# ── Summary comparison ──
print("\n" + "=" * 70)
print("SUMMARY: IMD 2019 vs 2025")
print("=" * 70)

d19 = results_all["IMD 2019 Fused"]['R²'] - results_all["IMD 2019 Crime-only"]['R²']
d25 = results_all["IMD 2025 Fused"]['R²'] - results_all["IMD 2025 Crime-only"]['R²']

print(f"\n   {'Metric':<25} {'IMD 2019':>12} {'IMD 2025':>12}")
print(f"   {'-'*49}")
print(f"   {'LSOAs matched':<25} {results_all['IMD 2019 Fused']['LSOAs']:>12,} {results_all['IMD 2025 Fused']['LSOAs']:>12,}")
print(f"   {'Crime-only R²':<25} {results_all['IMD 2019 Crime-only']['R²']:>12.4f} {results_all['IMD 2025 Crime-only']['R²']:>12.4f}")
print(f"   {'Fused R²':<25} {results_all['IMD 2019 Fused']['R²']:>12.4f} {results_all['IMD 2025 Fused']['R²']:>12.4f}")
print(f"   {'Δ R² (fusion lift)':<25} {d19:>+12.4f} {d25:>+12.4f}")
print(f"   {'Fused MAE':<25} {results_all['IMD 2019 Fused']['MAE']:>12.4f} {results_all['IMD 2025 Fused']['MAE']:>12.4f}")

if abs(d19 - d25) < 0.005:
    print(f"\n   ✅ CONSISTENT: Both IMD editions show similar Δ R² (difference = {abs(d19-d25):.4f})")
    print(f"   → Confirms that static socioeconomic data provides negligible predictive lift")
    print(f"      regardless of edition or LSOA code matching quality.")
else:
    print(f"\n   ⚠️  DIVERGENT: Δ R² differs by {abs(d19-d25):.4f}")
    print(f"   → Further investigation needed.")

db.close()
print("\n✅ IMD robustness check complete!")
