"""
28_error_analysis_map.py — Per-LSOA prediction error choropleth map.

Produces:
  1. Absolute MAE per LSOA (test set)
  2. Relative MAE (MAE / mean crime) per LSOA
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "error_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

db = ThesisDB()

# ── 1. BUILD BASELINE MODEL (same as script 03) ──
print("=" * 60)
print("PER-LSOA ERROR ANALYSIS")
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

features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'month_sin', 'month_cos', 'time_idx']

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

print(f"  LSOAs: {len(all_lsoas):,}")
print(f"  Train: {len(train):,}  Test: {len(test):,}")

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(train[features], train['crime_count'])
test = test.copy()
test['predicted'] = rf.predict(test[features])
test['abs_error'] = (test['crime_count'] - test['predicted']).abs()

# ── 2. COMPUTE PER-LSOA METRICS ──
per_lsoa = test.groupby('lsoa_code').agg(
    mae=('abs_error', 'mean'),
    mean_crime=('crime_count', 'mean'),
    std_crime=('crime_count', 'std'),
).reset_index()
per_lsoa['relative_mae'] = per_lsoa['mae'] / per_lsoa['mean_crime'].clip(lower=0.1)

print(f"\n  Per-LSOA MAE: median={per_lsoa['mae'].median():.2f}, "
      f"mean={per_lsoa['mae'].mean():.2f}, "
      f"max={per_lsoa['mae'].max():.1f}")
print(f"  Per-LSOA relative MAE: median={per_lsoa['relative_mae'].median():.2f}, "
      f"mean={per_lsoa['relative_mae'].mean():.2f}")

# ── 3. LOAD BOUNDARIES ──
boundaries = gpd.read_file(PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson")
boundaries = boundaries.rename(columns={boundaries.columns[0]: 'lsoa_code'} if 'LSOA21CD' not in boundaries.columns else {})
if 'LSOA21CD' in boundaries.columns:
    boundaries = boundaries.rename(columns={'LSOA21CD': 'lsoa_code'})

merged = boundaries.merge(per_lsoa, on='lsoa_code', how='inner')
print(f"  Matched: {len(merged):,} / {len(per_lsoa):,} LSOAs")

db.close()

# ── 4. PLOT: ABSOLUTE MAE ──
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
merged.plot(column='mae', ax=ax, legend=True, cmap='YlOrRd',
            legend_kwds={'label': 'Mean Absolute Error (crimes/month)', 'shrink': 0.6},
            edgecolor='face', linewidth=0.1,
            vmin=0, vmax=per_lsoa['mae'].quantile(0.95))
ax.set_title('Per-LSOA Prediction Error (Absolute MAE)\nBaseline Model, 6-Month Test Set',
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_mae_map.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {FIGURES_DIR / '01_mae_map.png'}")

# ── 5. PLOT: RELATIVE MAE (MAE / mean) ──
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
merged.plot(column='relative_mae', ax=ax, legend=True, cmap='RdYlGn_r',
            legend_kwds={'label': 'Relative MAE (MAE / Mean Crime)', 'shrink': 0.6},
            edgecolor='face', linewidth=0.1,
            vmin=0, vmax=per_lsoa['relative_mae'].quantile(0.95))
ax.set_title('Per-LSOA Prediction Error (Relative MAE)\nHigher = Worse Relative Performance',
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_relative_mae_map.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {FIGURES_DIR / '02_relative_mae_map.png'}")

# ── 6. SUMMARY STATS ──
print(f"\n{'='*60}")
print("TOP 10 HIGHEST ABSOLUTE MAE LSOAs")
print("="*60)
for _, r in per_lsoa.nlargest(10, 'mae').iterrows():
    print(f"  {r['lsoa_code']}  MAE={r['mae']:.1f}  mean={r['mean_crime']:.1f}  rel={r['relative_mae']:.2f}")

print(f"\n{'='*60}")
print("TOP 10 HIGHEST RELATIVE MAE LSOAs")
print("="*60)
for _, r in per_lsoa.nlargest(10, 'relative_mae').iterrows():
    print(f"  {r['lsoa_code']}  MAE={r['mae']:.2f}  mean={r['mean_crime']:.2f}  rel={r['relative_mae']:.2f}")

print(f"\n✅ Error analysis complete!")
