"""
26_noise_floor_analysis.py — Binned residual analysis for Poisson noise floor.

Validates that the baseline model's prediction error tracks the theoretical
Poisson noise floor across all crime levels, using binned residual plots
(Gelman & Hill, 2007).

Outputs:
    reports/figures/noise_floor_bins.csv  — per-bin statistics
    Console: verification numbers for thesis text
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("NOISE FLOOR ANALYSIS — Binned Residual Validation")
print("=" * 70)

db = duckdb.connect(str(PROJECT_ROOT / "data/processed/london/thesis.duckdb"), read_only=True)
crime = db.sql("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean
    GROUP BY lsoa_code, month
""").df()
db.close()

# Filter active LSOAs (same threshold as all experiments)
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
df = crime[crime['lsoa_code'].isin(active)].copy()
df = df.sort_values(['lsoa_code', 'month'])

print(f"\nActive LSOAs: {len(active):,}")
print(f"Total rows:   {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (identical to 03_baseline_model.py)
# ══════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['time_index'] = pd.to_datetime(df['month']).dt.year * 12 + pd.to_datetime(df['month']).dt.month
df = df.dropna()

features = [f'lag_{l}' for l in [1, 2, 3, 6, 12]] + \
           [f'rolling_mean_{w}' for w in [3, 6, 12]] + \
           ['month_sin', 'month_cos', 'time_index']

# ══════════════════════════════════════════════════════════════════════════
# 3. TRAIN/TEST SPLIT + MODEL (identical to baseline)
# ══════════════════════════════════════════════════════════════════════════
train = df[df['month'] < '2025-08']
test = df[df['month'] >= '2025-08'].copy()

print(f"Train: {len(train):,} rows")
print(f"Test:  {len(test):,} rows")

rf = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
rf.fit(train[features], train['crime_count'])

test['pred'] = rf.predict(test[features])
test['residual'] = test['crime_count'] - test['pred']

r2 = r2_score(test['crime_count'], test['pred'])
mae = mean_absolute_error(test['crime_count'], test['pred'])
print(f"\nTest R²:  {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. BINNED RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BINNED RESIDUAL ANALYSIS")
print("=" * 70)

# Bin predictions into deciles
test['pred_bin'] = pd.qcut(test['pred'], 10, duplicates='drop')

bin_stats = test.groupby('pred_bin', observed=False).agg(
    mean_pred=('pred', 'mean'),
    var_residual=('residual', 'var'),
    actual_mae=('residual', lambda x: np.abs(x).mean()),
    count=('residual', 'count')
).reset_index()

# Compute per-bin overdispersion and theoretical MAE
bin_stats['phi_residual'] = bin_stats['var_residual'] / bin_stats['mean_pred']
bin_stats['theoretical_mae'] = np.sqrt(2 * bin_stats['mean_pred'] * bin_stats['phi_residual'] / np.pi)
bin_stats['ratio'] = bin_stats['actual_mae'] / bin_stats['theoretical_mae']

# Create a clean label for each bin
bin_stats['crime_range'] = bin_stats['pred_bin'].astype(str)

print(f"\n{'Crime Range':22s} | {'Mean':>6s} | {'φ':>5s} | {'Actual':>7s} | {'Theory':>7s} | {'Ratio':>5s} | {'n':>6s}")
print("-" * 75)
for _, row in bin_stats.iterrows():
    print(f"{row['crime_range']:22s} | {row['mean_pred']:6.1f} | {row['phi_residual']:5.2f} | "
          f"{row['actual_mae']:7.2f} | {row['theoretical_mae']:7.2f} | {row['ratio']:5.2f} | {row['count']:6,}")

# Save to CSV
out_csv = FIGURES_DIR / "noise_floor_bins.csv"
bin_stats.to_csv(out_csv, index=False, float_format='%.4f')
print(f"\nSaved: {out_csv}")

# ══════════════════════════════════════════════════════════════════════════
# 5. SUMMARY STATISTICS FOR THESIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("THESIS-READY NUMBERS")
print("=" * 70)

# Median ratio across bins
median_ratio = bin_stats['ratio'].median()
mean_ratio = bin_stats['ratio'].mean()
print(f"\nActual/Theoretical MAE ratio:")
print(f"  Median: {median_ratio:.2f}")
print(f"  Mean:   {mean_ratio:.2f}")

# Pure Poisson floor (φ=1)
y_bar = test['pred'].mean()
pure_floor = math.sqrt(2 * y_bar / math.pi)
pure_gap = (mae - pure_floor) / pure_floor * 100
print(f"\nPure Poisson floor (φ=1, ȳ={y_bar:.1f}): {pure_floor:.2f}")
print(f"Gap above pure Poisson: {pure_gap:+.1f}%")

# Per-LSOA averaged
overall_var = test['residual'].var()
overall_phi = overall_var / y_bar
print(f"\nOverall residual φ: {overall_phi:.2f}")
print(f"Overall residual σ: {math.sqrt(overall_var):.2f}")

# Key claim: bins 1-9 (excluding extreme high-crime outliers)
bins_1_9 = bin_stats.iloc[:9]
print(f"\nBins 1-9 (excl. extreme high-crime):")
print(f"  Mean ratio:   {bins_1_9['ratio'].mean():.2f}")
print(f"  Min ratio:    {bins_1_9['ratio'].min():.2f}")
print(f"  Max ratio:    {bins_1_9['ratio'].max():.2f}")

print(f"\n✅ Noise floor analysis complete!")
