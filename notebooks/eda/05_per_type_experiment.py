"""
Experiment 4: Per-Crime-Type Analysis
Run baseline vs fused model for each crime type individually.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "per_type"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

db = ThesisDB()

# Load IMD
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
imd_cols_map = {
    'LSOA code (2011)': 'lsoa_code_2011',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Income Score (rate)': 'income_score',
    'Employment Score (rate)': 'employment_score',
    'Education, Skills and Training Score': 'education_score',
    'Health Deprivation and Disability Score': 'health_score',
    'Crime Score': 'crime_deprivation_score',
    'Barriers to Housing and Services Score': 'housing_score',
    'Living Environment Score': 'living_environment_score',
}
imd_slim = imd[list(imd_cols_map.keys())].rename(columns=imd_cols_map)
imd_features = list(imd_cols_map.values())[1:]

# Get all crime types
crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
print(f"Running experiment for {len(crime_types)} crime types...\n")

results = []

for ct in crime_types:
    print(f"  {ct}...", end=" ", flush=True)
    
    # Aggregate for this crime type
    monthly = db.query(f"""
        SELECT lsoa_code, month, COUNT(*) as crime_count
        FROM crime_clean WHERE crime_type = '{ct.replace("'", "''")}'
        GROUP BY lsoa_code, month
    """)
    
    # Join IMD
    merged = monthly.merge(imd_slim, left_on='lsoa_code', right_on='lsoa_code_2011', how='inner')
    
    # Filter: need enough data per LSOA (at least 12 crimes for this type)
    lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= 12].index
    if len(active) < 50:
        print(f"SKIP (only {len(active)} LSOAs with ≥12 crimes)")
        results.append({'crime_type': ct, 'n_lsoas': len(active), 'skipped': True})
        continue
    
    merged = merged[merged['lsoa_code'].isin(active)]
    
    # Complete grid
    all_months = sorted(merged['month'].unique())
    all_lsoas = sorted(merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
    
    imd_per_lsoa = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + imd_features].set_index('lsoa_code')
    df = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
    df = df.merge(imd_per_lsoa, on='lsoa_code', how='left')
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    
    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    
    df_model = df.dropna()
    if len(df_model) < 100:
        print(f"SKIP (only {len(df_model)} rows after lag)")
        results.append({'crime_type': ct, 'n_lsoas': len(active), 'skipped': True})
        continue
    
    test_months = all_months[-6:]
    train = df_model[~df_model['month'].isin(test_months)]
    test = df_model[df_model['month'].isin(test_months)]
    
    if len(test) < 50:
        print(f"SKIP (only {len(test)} test rows)")
        results.append({'crime_type': ct, 'n_lsoas': len(active), 'skipped': True})
        continue
    
    lag_cols = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_12', 'month_sin', 'month_cos']
    fused_cols = lag_cols + imd_features
    
    # Train both
    rf_crime = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_fused = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    
    rf_crime.fit(train[lag_cols], train['crime_count'])
    rf_fused.fit(train[fused_cols], train['crime_count'])
    
    r2_crime = r2_score(test['crime_count'], rf_crime.predict(test[lag_cols]))
    r2_fused = r2_score(test['crime_count'], rf_fused.predict(test[fused_cols]))
    mae_crime = mean_absolute_error(test['crime_count'], rf_crime.predict(test[lag_cols]))
    mae_fused = mean_absolute_error(test['crime_count'], rf_fused.predict(test[fused_cols]))
    
    delta = r2_fused - r2_crime
    print(f"R²: {r2_crime:.4f} → {r2_fused:.4f} (Δ={delta:+.4f}) | LSOAs={len(active)}")
    
    results.append({
        'crime_type': ct, 'n_lsoas': len(active), 'skipped': False,
        'r2_crime': r2_crime, 'r2_fused': r2_fused, 'delta_r2': delta,
        'mae_crime': mae_crime, 'mae_fused': mae_fused,
        'mean_count': test['crime_count'].mean()
    })

db.close()

# Results table
print("\n" + "="*90)
print("PER-CRIME-TYPE RESULTS")
print("="*90)

res_df = pd.DataFrame(results)
active_res = res_df[~res_df.get('skipped', False)].sort_values('delta_r2', ascending=False)

print(f"\n{'Crime Type':<35} {'LSOAs':>6} {'R²(crime)':>10} {'R²(fused)':>10} {'Δ R²':>8} {'Mean':>6}")
print("-"*80)
for _, r in active_res.iterrows():
    marker = "✅" if r['delta_r2'] > 0.005 else "➖" if r['delta_r2'] > -0.005 else "❌"
    print(f"{marker} {r['crime_type']:<33} {r['n_lsoas']:>6} {r['r2_crime']:>10.4f} {r['r2_fused']:>10.4f} {r['delta_r2']:>+8.4f} {r['mean_count']:>6.1f}")

skipped = res_df[res_df.get('skipped', False)]
if len(skipped) > 0:
    print(f"\nSkipped ({len(skipped)}): {', '.join(skipped['crime_type'])}")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#22c55e' if d > 0.005 else '#ef4444' if d < -0.005 else '#6b7280' for d in active_res['delta_r2']]
ax.barh(range(len(active_res)), active_res['delta_r2'], color=colors)
ax.set_yticks(range(len(active_res)))
ax.set_yticklabels(active_res['crime_type'])
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Δ R² (Fused - Crime-Only)')
ax.set_title('IMD Fusion Impact by Crime Type', fontweight='bold', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_delta_r2_by_type.png")
plt.close()
print(f"\n✅ Figure saved: {FIGURES_DIR / '01_delta_r2_by_type.png'}")
