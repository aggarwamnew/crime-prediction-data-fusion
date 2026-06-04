"""
27_r2_by_crime_level.py — R² decomposition by LSOA mean crime level.

Bins LSOAs by their mean crime count per month and computes baseline
model R², MAE, RMSE, and MASE within each bin. Shows how much of the
headline R²=0.94 is driven by between-group variance vs genuine
temporal prediction power.

Bins:
    ≤2     (very low crime — dominated by Poisson noise)
    >2–10  (low crime)
    >10–50 (mid crime — the "honest" prediction band)
    >50–150 (high crime)
    >150    (extreme — Westminster, Camden, etc.)

Usage:
    python notebooks/eda/27_r2_by_crime_level.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA (identical to 03_baseline_model.py)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SCRIPT 27: R² BY LSOA MEAN CRIME LEVEL")
print("=" * 80)

db = ThesisDB()
crime = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean
    GROUP BY lsoa_code, month
    ORDER BY lsoa_code, month
""")
db.close()

# Filter active LSOAs (≥36 total crimes)
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
df = crime[crime['lsoa_code'].isin(active)].copy()

# Complete grid (every LSOA × every month)
all_months = sorted(df['month'].unique())
all_lsoas = sorted(df['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = df.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

print(f"\nActive LSOAs: {len(all_lsoas):,}")
print(f"Months: {len(all_months)}")
print(f"Total rows: {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
month_to_idx = {m: i for i, m in enumerate(all_months)}
df['time_idx'] = df['month'].map(month_to_idx)

features = [f'lag_{l}' for l in [1, 2, 3, 6, 12]] + \
           [f'rolling_mean_{w}' for w in [3, 6, 12]] + \
           ['month_sin', 'month_cos', 'time_idx']

df_model = df.dropna().copy()

# ══════════════════════════════════════════════════════════════════════════
# 3. COMPUTE MEAN CRIME PER LSOA
# ══════════════════════════════════════════════════════════════════════════
print("\nComputing mean crime per LSOA...")

lsoa_means = df.groupby('lsoa_code')['crime_count'].mean()

# Define bins
bins = [
    ("≤2",      0,   2),
    (">2–10",   2,  10),
    (">10–50", 10,  50),
    (">50–150", 50, 150),
    (">150",  150, 9999),
]

print(f"\n{'Bin':12s} | {'LSOAs':>7s} | {'Mean crime':>10s} | {'Min':>5s} | {'Max':>5s}")
print("-" * 55)
for label, lo, hi in bins:
    mask = (lsoa_means > lo) if lo > 0 else (lsoa_means >= 0)
    mask = mask & (lsoa_means <= hi)
    group_means = lsoa_means[mask]
    if len(group_means) > 0:
        print(f"{label:12s} | {len(group_means):7,} | {group_means.mean():10.1f} | "
              f"{group_means.min():5.1f} | {group_means.max():5.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)].copy()

print(f"\nTrain: {len(train):,} rows ({train['month'].min()} → {train['month'].max()})")
print(f"Test:  {len(test):,} rows ({test['month'].min()} → {test['month'].max()})")

# ══════════════════════════════════════════════════════════════════════════
# 5. TRAIN SINGLE MODEL ON ALL DATA (same as baseline)
# ══════════════════════════════════════════════════════════════════════════
print("\nTraining baseline RF on ALL LSOAs...")

rf = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
rf.fit(train[features], train['crime_count'])
test['pred'] = rf.predict(test[features])

# Overall metrics
overall_r2 = r2_score(test['crime_count'], test['pred'])
overall_mae = mean_absolute_error(test['crime_count'], test['pred'])
overall_rmse = np.sqrt(mean_squared_error(test['crime_count'], test['pred']))

# MASE: scale by naive forecast (lag_1)
naive_errors = np.abs(test['crime_count'] - test['lag_1'])
overall_mase = overall_mae / naive_errors.mean()

print(f"\nOverall (all LSOAs): R²={overall_r2:.4f}  MAE={overall_mae:.2f}  "
      f"RMSE={overall_rmse:.2f}  MASE={overall_mase:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. R² BY CRIME-LEVEL BIN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("R² BY LSOA MEAN CRIME LEVEL")
print("=" * 80)

# Add LSOA mean to test set
test['lsoa_mean'] = test['lsoa_code'].map(lsoa_means)

results = []

print(f"\n{'Bin':12s} | {'LSOAs':>6s} | {'Rows':>7s} | {'Mean':>5s} | "
      f"{'R²':>7s} | {'MAE':>6s} | {'RMSE':>6s} | {'MASE':>6s} | "
      f"{'MAE/ȳ':>6s} | {'SS_tot':>10s} | {'SS_res':>10s}")
print("-" * 110)

for label, lo, hi in bins:
    # Get LSOAs in this bin
    if lo == 0:
        bin_lsoas = lsoa_means[lsoa_means <= hi].index
    else:
        bin_lsoas = lsoa_means[(lsoa_means > lo) & (lsoa_means <= hi)].index

    bin_test = test[test['lsoa_code'].isin(bin_lsoas)]

    if len(bin_test) < 10:
        print(f"{label:12s} | {'(too few)':>6s}")
        continue

    y_true = bin_test['crime_count'].values
    y_pred = bin_test['pred'].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MASE
    naive_err = np.abs(bin_test['crime_count'].values - bin_test['lag_1'].values)
    mase = mae / naive_err.mean() if naive_err.mean() > 0 else float('nan')

    y_bar = y_true.mean()
    ss_tot = np.sum((y_true - y_bar) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    mae_pct = mae / y_bar * 100 if y_bar > 0 else float('nan')

    n_lsoas = len(bin_lsoas)

    print(f"{label:12s} | {n_lsoas:6,} | {len(bin_test):7,} | {y_bar:5.1f} | "
          f"{r2:7.4f} | {mae:6.2f} | {rmse:6.2f} | {mase:6.4f} | "
          f"{mae_pct:5.1f}% | {ss_tot:10,.0f} | {ss_res:10,.0f}")

    results.append({
        'bin': label, 'n_lsoas': n_lsoas, 'n_rows': len(bin_test),
        'mean_crime': y_bar, 'r2': r2, 'mae': mae, 'rmse': rmse,
        'mase': mase, 'mae_pct': mae_pct, 'ss_tot': ss_tot, 'ss_res': ss_res
    })

# ══════════════════════════════════════════════════════════════════════════
# 7. KEY INSIGHT: SS_res IS SIMILAR, SS_tot CHANGES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("KEY INSIGHT: WHY R² VARIES")
print("=" * 80)

if len(results) >= 2:
    # Per-row SS_res (normalised)
    print(f"\n{'Bin':12s} | {'SS_res/row':>10s} | {'SS_tot/row':>10s} | {'Ratio':>6s}")
    print("-" * 50)
    for r in results:
        ss_res_per = r['ss_res'] / r['n_rows']
        ss_tot_per = r['ss_tot'] / r['n_rows']
        ratio = ss_res_per / ss_tot_per if ss_tot_per > 0 else float('nan')
        print(f"{r['bin']:12s} | {ss_res_per:10.2f} | {ss_tot_per:10.2f} | {ratio:6.4f}")

    print(f"\n→ SS_res/row reflects prediction error (should scale with crime level)")
    print(f"→ SS_tot/row reflects total variance (MUCH higher when range is wider)")
    print(f"→ R² = 1 - ratio: higher when SS_tot dominates (i.e. wide crime range)")

# ══════════════════════════════════════════════════════════════════════════
# 8. MASE INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MASE INTERPRETATION (Scale-Independent)")
print("=" * 80)
print("\nMASE < 1.0 means the model beats the naive (lag-1) forecast.")
print("MASE is not affected by between-LSOA variance inflation.\n")

for r in results:
    beats = "✅ beats naive" if r['mase'] < 1.0 else "❌ worse than naive"
    print(f"  {r['bin']:12s}  MASE = {r['mase']:.4f}  {beats}")

# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)
out_csv = PROJECT_ROOT / "reports" / "figures" / "r2_by_crime_level.csv"
results_df.to_csv(out_csv, index=False, float_format='%.4f')
print(f"\n✅ Results saved: {out_csv}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
