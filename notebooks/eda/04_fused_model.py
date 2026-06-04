"""
Step 5: IMD Fusion Model
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "fusion"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150; plt.rcParams["savefig.dpi"] = 150

db = ThesisDB()
print("="*70); print("STEP 5: IMD FUSION MODEL"); print("="*70)

# 1. Load IMD
print("\n1. Loading IMD 2019...")
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
print(f"   IMD rows: {len(imd):,}")

# Select score columns only (not ranks/deciles — they're redundant)
imd_cols = {
    'LSOA code (2011)': 'lsoa_code_2011',
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
imd_slim = imd[list(imd_cols.keys())].rename(columns=imd_cols)
print(f"   IMD features: {list(imd_slim.columns[1:])}")

# 2. Crime data uses 2021 LSOA codes, IMD uses 2011 codes
# For most LSOAs, the code is unchanged between 2011 and 2021
# Let's try direct join first and see coverage
print("\n2. Joining IMD to crime data...")
monthly = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")

# Direct join on lsoa_code = lsoa_code_2011
merged = monthly.merge(imd_slim, left_on='lsoa_code', right_on='lsoa_code_2011', how='left')
matched = merged['imd_score'].notna().sum()
total = len(merged)
print(f"   Direct join match: {matched:,}/{total:,} ({matched/total*100:.1f}%)")
print(f"   LSOAs matched: {merged[merged['imd_score'].notna()]['lsoa_code'].nunique():,}")
print(f"   LSOAs unmatched: {merged[merged['imd_score'].isna()]['lsoa_code'].nunique():,}")

# Drop unmatched (missing IMD data)
merged = merged.dropna(subset=['imd_score'])
print(f"   After dropping unmatched: {len(merged):,} rows")

# 3. Filter to active LSOAs (same threshold as baseline)
MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
print(f"\n3. After filtering ≥{MIN_CRIMES} crimes: {merged['lsoa_code'].nunique():,} LSOAs")

# 4. Create complete grid and lag features
print("\n4. Creating features...")
all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

# Get IMD per LSOA (static)
imd_per_lsoa = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + list(imd_cols.values())[1:]].set_index('lsoa_code')

# Build grid with crime counts
crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()

# Join IMD features
df = df.merge(imd_per_lsoa, on='lsoa_code', how='left')

# Lag features
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())

# Temporal
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
month_idx = {m: i for i, m in enumerate(all_months)}
df['time_idx'] = df['month'].map(month_idx)

# 5. Train/test split
print("\n5. Train/test split...")
df_model = df.dropna().copy()
test_months = all_months[-6:]
train_df = df_model[~df_model['month'].isin(test_months)]
test_df = df_model[df_model['month'].isin(test_months)]

imd_features = list(imd_cols.values())[1:]  # IMD score columns
lag_features = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
temporal_features = ['month_sin', 'month_cos', 'time_idx']

# Model A: crime-only (same as baseline, for fair comparison on same LSOAs)
crime_only_cols = lag_features + temporal_features
# Model B: crime + IMD (fused)
fused_cols = lag_features + temporal_features + imd_features

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"   Crime-only features: {len(crime_only_cols)}")
print(f"   Fused features: {len(fused_cols)} (+{len(imd_features)} IMD)")

# 6. Train both models
print("\n6. Training models...")

rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])
print("   ✅ Crime-only RF trained")

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])
print("   ✅ Fused RF trained")

# 7. Evaluate
print("\n" + "="*70); print("7. RESULTS"); print("="*70)

results = {}
for name, model, cols in [("Crime-only", rf_crime, crime_only_cols), ("Fused (+ IMD)", rf_fused, fused_cols)]:
    y_pred = model.predict(test_df[cols])
    r2 = r2_score(test_df['crime_count'], y_pred)
    mae = mean_absolute_error(test_df['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"\n   {name}:")
    print(f"     R²:   {r2:.4f}")
    print(f"     MAE:  {mae:.4f}")
    print(f"     RMSE: {rmse:.4f}")

delta_r2 = results["Fused (+ IMD)"]["R²"] - results["Crime-only"]["R²"]
delta_mae = results["Crime-only"]["MAE"] - results["Fused (+ IMD)"]["MAE"]
print(f"\n   Δ R² = {delta_r2:+.4f}")
print(f"   Δ MAE = {delta_mae:+.4f} (negative = fused is better)")

# 8. Feature importance for fused model
print("\n8. Fused model feature importance:")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
for _, row in imp.iterrows():
    marker = " ← IMD" if row['feature'] in imd_features else ""
    print(f"   {row['feature']:30s} {row['importance']:.4f}{marker}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#f59e0b' if f in imd_features else '#2563eb' for f in imp['feature']]
ax.barh(range(len(imp)), imp['importance'], color=colors)
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Fused Model — Feature Importance (Blue=Crime, Orange=IMD)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout(); plt.savefig(FIGURES_DIR / "01_fused_feature_importance.png"); plt.close()

# 9. Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ['R²', 'MAE', 'RMSE']):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(list(results.keys()), vals, color=['#2563eb', '#f59e0b'])
    ax.set_title(metric, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:.4f}', ha='center', va='bottom', fontsize=10)
plt.suptitle('Crime-Only vs Fused Model Comparison', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "02_model_comparison.png"); plt.close()

db.close()
print(f"\n✅ Figures saved to {FIGURES_DIR}")
print("✅ Step 5 complete!")
