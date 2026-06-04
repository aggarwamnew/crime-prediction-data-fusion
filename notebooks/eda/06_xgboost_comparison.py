"""
XGBoost Comparison: Validate that IMD findings hold across algorithms.
Runs RF vs XGBoost, both crime-only and fused, on the same data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "xgboost"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

db = ThesisDB()
print("="*70); print("XGBOOST vs RANDOM FOREST COMPARISON"); print("="*70)

# Load IMD
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
imd_map = {
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
imd_slim = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_features = list(imd_map.values())[1:]

# Build feature matrix (same as 04_fused_model.py)
monthly = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
merged = monthly.merge(imd_slim, left_on='lsoa_code', right_on='lsoa_code_2011', how='inner')
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
merged = merged[merged['lsoa_code'].isin(active)]

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
imd_per_lsoa = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + imd_features].set_index('lsoa_code')
df = merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
df = df.merge(imd_per_lsoa, on='lsoa_code', how='left')
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['time_idx'] = df['month'].map({m: i for i, m in enumerate(all_months)})

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

lag_cols = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12','month_sin','month_cos','time_idx']
fused_cols = lag_cols + imd_features

print(f"\nTrain: {len(train):,} | Test: {len(test):,} | LSOAs: {len(all_lsoas):,}")

# Train 4 models
models = {
    'RF Crime-only': (RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42), lag_cols),
    'RF Fused (+IMD)': (RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42), fused_cols),
    'GBT Crime-only': (GradientBoostingRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, learning_rate=0.1, random_state=42), lag_cols),
    'GBT Fused (+IMD)': (GradientBoostingRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, learning_rate=0.1, random_state=42), fused_cols),
}

results = {}
for name, (model, cols) in models.items():
    print(f"\n  Training {name}...", end=" ", flush=True)
    model.fit(train[cols], train['crime_count'])
    y_pred = model.predict(test[cols])
    r2 = r2_score(test['crime_count'], y_pred)
    mae = mean_absolute_error(test['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R²={r2:.4f}, MAE={mae:.4f}")

# Results table
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"\n{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
print("-"*50)
for name, m in results.items():
    print(f"{name:<25} {m['R²']:>8.4f} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}")

# Deltas
rf_delta = results['RF Fused (+IMD)']['R²'] - results['RF Crime-only']['R²']
gbt_delta = results['GBT Fused (+IMD)']['R²'] - results['GBT Crime-only']['R²']
print(f"\nΔ R² (RF):  {rf_delta:+.4f}")
print(f"Δ R² (GBT): {gbt_delta:+.4f}")
print(f"\nConclusion: IMD effect is {'consistent' if abs(rf_delta - gbt_delta) < 0.01 else 'different'} across algorithms")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(4)
r2_vals = [results[n]['R²'] for n in results]
colors = ['#2563eb', '#60a5fa', '#f59e0b', '#fcd34d']
bars = ax.bar(x, r2_vals, color=colors, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(list(results.keys()), rotation=15, ha='right')
ax.set_ylabel('Test R²')
ax.set_title('Algorithm Comparison: RF vs GBT, Crime-Only vs Fused', fontweight='bold')
ax.set_ylim(min(r2_vals) - 0.01, max(r2_vals) + 0.005)
for bar, v in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:.4f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_rf_vs_gbt.png")
plt.close()

db.close()
print(f"\n✅ Figure saved: {FIGURES_DIR / '01_rf_vs_gbt.png'}")
