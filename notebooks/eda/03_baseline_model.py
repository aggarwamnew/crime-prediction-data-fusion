"""
Step 4: Crime-Only Baseline Model
===================================

Builds a Random Forest regressor that predicts total crime count per LSOA
per month using ONLY lagged crime counts and temporal features.

This is the baseline that data fusion (Step 5) must beat.

Pipeline:
1. Aggregate crime counts per (LSOA, month)
2. Filter to LSOAs with sufficient data (≥36 total crimes = avg 1/month)
3. Create lag features (t-1, t-2, t-3, t-6, t-12)
4. Add temporal features (month-of-year for seasonality)
5. Temporal train/test split (last 6 months = test)
6. Train Random Forest
7. Evaluate: MAE, RMSE, R², and spatial analysis of errors

Usage:
    python notebooks/eda/03_baseline_model.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')

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

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "baseline"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

db = ThesisDB()

print("=" * 70)
print("STEP 4: CRIME-ONLY BASELINE MODEL")
print("=" * 70)

# =====================================================================
# 1. Aggregate crime counts per (LSOA, month)
# =====================================================================
print("\n1. Aggregating crime counts per (LSOA, month)...")

monthly_counts = db.query("""
    SELECT 
        lsoa_code,
        month,
        COUNT(*) as crime_count
    FROM crime_clean
    GROUP BY lsoa_code, month
    ORDER BY lsoa_code, month
""")

print(f"   Raw aggregation: {len(monthly_counts):,} (LSOA, month) pairs")
print(f"   Unique LSOAs: {monthly_counts['lsoa_code'].nunique():,}")
print(f"   Months: {monthly_counts['month'].nunique()}")

# =====================================================================
# 2. Filter to LSOAs with sufficient data
# =====================================================================
print("\n2. Filtering LSOAs...")

# Calculate total crimes per LSOA
lsoa_totals = monthly_counts.groupby('lsoa_code')['crime_count'].sum()

# Threshold: at least 36 total crimes (avg ≥1 per month)
MIN_CRIMES = 36
active_lsoas = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
n_before = monthly_counts['lsoa_code'].nunique()
monthly_counts = monthly_counts[monthly_counts['lsoa_code'].isin(active_lsoas)]
n_after = monthly_counts['lsoa_code'].nunique()

print(f"   Threshold: ≥{MIN_CRIMES} total crimes (avg ≥1/month)")
print(f"   LSOAs before: {n_before:,}")
print(f"   LSOAs after:  {n_after:,} (dropped {n_before - n_after:,} sparse LSOAs)")
print(f"   Crimes retained: {monthly_counts['crime_count'].sum():,} "
      f"({monthly_counts['crime_count'].sum()/db.query('SELECT COUNT(*) as n FROM crime_clean')['n'].iloc[0]*100:.1f}% of cleaned data)")

# Create a complete grid (every LSOA × every month) to handle months with 0 crimes
all_months = sorted(monthly_counts['month'].unique())
all_lsoas = sorted(monthly_counts['lsoa_code'].unique())

print(f"\n   Creating complete LSOA × month grid...")
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = monthly_counts.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
print(f"   Grid size: {len(df):,} rows ({len(all_lsoas):,} LSOAs × {len(all_months)} months)")

# =====================================================================
# 3. Create lag features
# =====================================================================
print("\n3. Creating lag features...")

# Sort by LSOA and month for correct lag computation
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

# Create lags within each LSOA
lag_periods = [1, 2, 3, 6, 12]
for lag in lag_periods:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)

# Rolling features
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(
    lambda x: x.shift(1).rolling(12, min_periods=1).mean()
)

# =====================================================================
# 4. Temporal features
# =====================================================================
print("4. Adding temporal features...")

df['month_num'] = pd.to_datetime(df['month']).dt.month  # 1-12
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Year trend (linear)
month_to_idx = {m: i for i, m in enumerate(all_months)}
df['time_idx'] = df['month'].map(month_to_idx)

print(f"   Features created: {[c for c in df.columns if c not in ['lsoa_code', 'month', 'crime_count']]}")

# =====================================================================
# 5. Drop rows with NaN (from lag computation) and prepare splits
# =====================================================================
print("\n5. Preparing train/test split...")

# Drop rows where lag_12 is NaN (first 12 months don't have all lags)
df_model = df.dropna().copy()
print(f"   Rows after dropping NaN lags: {len(df_model):,} "
      f"(dropped {len(df) - len(df_model):,} rows from first 12 months)")

# Temporal split: last 6 months = test
test_months = all_months[-6:]  # Aug 2025 - Jan 2026
train_months = [m for m in all_months if m not in test_months]

# Only use train months that have all lags available
train_df = df_model[df_model['month'].isin(train_months)]
test_df = df_model[df_model['month'].isin(test_months)]

print(f"   Train: {len(train_df):,} rows ({train_df['month'].nunique()} months: "
      f"{train_df['month'].min()} → {train_df['month'].max()})")
print(f"   Test:  {len(test_df):,} rows ({test_df['month'].nunique()} months: "
      f"{test_df['month'].min()} → {test_df['month'].max()})")

# Feature columns
feature_cols = [c for c in df_model.columns 
                if c not in ['lsoa_code', 'month', 'crime_count', 'month_num']]

X_train = train_df[feature_cols]
y_train = train_df['crime_count']
X_test = test_df[feature_cols]
y_test = test_df['crime_count']

print(f"\n   Features ({len(feature_cols)}):")
for f in feature_cols:
    print(f"     {f}")

# =====================================================================
# 6. Train Random Forest
# =====================================================================
print("\n6. Training Random Forest...")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
print(f"   ✅ RF trained: {rf.n_estimators} trees, max_depth={rf.max_depth}")

# =====================================================================
# 7. Evaluation
# =====================================================================
print("\n" + "=" * 70)
print("7. EVALUATION")
print("=" * 70)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Metrics
metrics = {
    'Train MAE': mean_absolute_error(y_train, y_pred_train),
    'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'Train R²': r2_score(y_train, y_pred_train),
    'Test MAE': mean_absolute_error(y_test, y_pred_test),
    'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'Test R²': r2_score(y_test, y_pred_test),
}

print(f"\n   {'Metric':<15} {'Value':>10}")
print(f"   {'-'*26}")
for name, val in metrics.items():
    print(f"   {name:<15} {val:>10.4f}")

# Mean crime count for context
print(f"\n   Context:")
print(f"     Mean crime count (test): {y_test.mean():.2f}")
print(f"     Median crime count (test): {y_test.median():.2f}")
print(f"     MAE as % of mean: {metrics['Test MAE']/y_test.mean()*100:.1f}%")

# =====================================================================
# 8. Feature Importance
# =====================================================================
print("\n8. Feature Importance:")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   {row['feature']:20s} {row['importance']:.4f} {bar}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(importance)), importance['importance'], color='#2563eb')
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest — Feature Importance (Crime-Only Baseline)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_feature_importance.png")
plt.close()

# =====================================================================
# 9. Predicted vs Actual scatter
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_pred_train, 'Train'),
    (axes[1], y_test, y_pred_test, 'Test')
]:
    ax.scatter(y_true, y_pred, alpha=0.1, s=5, color='#2563eb')
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Actual Crime Count')
    ax.set_ylabel('Predicted Crime Count')
    ax.set_title(f'{title} — Predicted vs Actual', fontweight='bold')
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.2f}', 
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_predicted_vs_actual.png")
plt.close()

# =====================================================================
# 10. Error by month (temporal pattern in residuals)
# =====================================================================
test_df = test_df.copy()
test_df['y_pred'] = y_pred_test
test_df['error'] = test_df['crime_count'] - test_df['y_pred']
test_df['abs_error'] = test_df['error'].abs()

monthly_error = test_df.groupby('month').agg(
    mean_actual=('crime_count', 'mean'),
    mean_predicted=('y_pred', 'mean'),
    mae=('abs_error', 'mean'),
    count=('crime_count', 'count')
).reset_index()

print("\n9. Error by test month:")
print(monthly_error.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(monthly_error))
ax.bar(x, monthly_error['mean_actual'], width=0.4, label='Actual', color='#2563eb', alpha=0.8)
ax.bar([i+0.4 for i in x], monthly_error['mean_predicted'], width=0.4, label='Predicted', color='#f59e0b', alpha=0.8)
ax.set_xticks([i+0.2 for i in x])
ax.set_xticklabels(monthly_error['month'], rotation=45, ha='right')
ax.set_ylabel('Mean Crime Count per LSOA')
ax.set_title('Baseline: Actual vs Predicted by Month', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_monthly_prediction.png")
plt.close()

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("BASELINE MODEL SUMMARY")
print("=" * 70)
print(f"""
Model:              Random Forest (200 trees, max_depth=15)
Target:             Total crime count per LSOA per month
Features:           {len(feature_cols)} (lagged counts + temporal)
Training data:      {len(train_df):,} rows ({train_df['month'].min()} → {train_df['month'].max()})
Test data:          {len(test_df):,} rows ({test_df['month'].min()} → {test_df['month'].max()})
LSOAs:              {n_after:,} (filtered ≥{MIN_CRIMES} crimes)
  
RESULTS:
  Train R²:         {metrics['Train R²']:.4f}
  Test R²:          {metrics['Test R²']:.4f}
  Test MAE:         {metrics['Test MAE']:.4f}
  Test RMSE:        {metrics['Test RMSE']:.4f}
  MAE as % of mean: {metrics['Test MAE']/y_test.mean()*100:.1f}%

Top 3 features:
  1. {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']:.4f})
  2. {importance.iloc[1]['feature']} ({importance.iloc[1]['importance']:.4f})
  3. {importance.iloc[2]['feature']} ({importance.iloc[2]['importance']:.4f})

Figures saved to: {FIGURES_DIR}
""")

db.close()
print("✅ Baseline model complete!")
