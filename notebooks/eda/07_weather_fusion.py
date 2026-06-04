"""
Weather Fusion Experiment: Add monthly weather features to the model.
Tests whether DYNAMIC features (weather) add more value than STATIC features (IMD).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "weather"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

db = ThesisDB()
print("="*70); print("WEATHER FUSION EXPERIMENT"); print("="*70)

# 1. Parse Met Office data
print("\n1. Parsing Heathrow weather data...")
weather_path = PROJECT_ROOT / "data/raw/london/weather/heathrow_monthly.txt"
lines = weather_path.read_text().strip().split('\n')

records = []
for line in lines:
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        try:
            year, month = int(parts[0]), int(parts[1])
            tmax = float(parts[2].replace('*','')) if '---' not in parts[2] else None
            tmin = float(parts[3].replace('*','')) if '---' not in parts[3] else None
            rain = float(parts[5].replace('*','')) if '---' not in parts[5] else None
            sun = float(parts[6].replace('*','').replace('#','')) if '---' not in parts[6] else None
            records.append({'year': year, 'month_num': month, 'tmax': tmax, 'tmin': tmin, 'rain_mm': rain, 'sun_hours': sun})
        except (ValueError, IndexError):
            continue

weather = pd.DataFrame(records)
weather['month'] = weather.apply(lambda r: f"{int(r['year'])}-{int(r['month_num']):02d}", axis=1)
weather['tmean'] = (weather['tmax'] + weather['tmin']) / 2

# Filter to our study period
weather = weather[(weather['year'] >= 2023) & (weather['year'] <= 2026)]
weather = weather[weather['month'].isin([f"{y}-{m:02d}" for y in range(2023,2027) for m in range(1,13)])]
print(f"   Weather records for study period: {len(weather)}")
print(weather[['month', 'tmax', 'tmin', 'tmean', 'rain_mm', 'sun_hours']].to_string(index=False))

weather_features = ['tmax', 'tmin', 'tmean', 'rain_mm', 'sun_hours']

# 2. Load crime + IMD (same as fusion model)
print("\n2. Loading crime + IMD data...")
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

# Join weather (same for all LSOAs — London-wide)
weather_slim = weather[['month'] + weather_features].copy()
df = df.merge(weather_slim, on='month', how='left')

df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

# Lag features
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['time_idx'] = df['month'].map({m: i for i, m in enumerate(all_months)})

# 3. Model variants
print(f"\n3. Preparing model variants...")
df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

lag_cols = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12','month_sin','month_cos','time_idx']

variants = {
    'Crime-only': lag_cols,
    '+ IMD': lag_cols + imd_features,
    '+ Weather': lag_cols + weather_features,
    '+ IMD + Weather': lag_cols + imd_features + weather_features,
}

print(f"   Train: {len(train):,} | Test: {len(test):,}")
for name, cols in variants.items():
    print(f"   {name}: {len(cols)} features")

# 4. Train and evaluate
results = {}
for name, cols in variants.items():
    print(f"\n  Training {name}...", end=" ", flush=True)
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(train[cols], train['crime_count'])
    y_pred = rf.predict(test[cols])
    r2 = r2_score(test['crime_count'], y_pred)
    mae = mean_absolute_error(test['crime_count'], y_pred)
    rmse = np.sqrt(mean_squared_error(test['crime_count'], y_pred))
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R²={r2:.4f}, MAE={mae:.4f}")
    
    # Feature importance for the full model
    if name == '+ IMD + Weather':
        imp = pd.DataFrame({'feature': cols, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
        print("\n  Feature importance (full model):")
        for _, row in imp.head(15).iterrows():
            tag = ""
            if row['feature'] in imd_features: tag = " ← IMD"
            elif row['feature'] in weather_features: tag = " ← WEATHER"
            print(f"    {row['feature']:30s} {row['importance']:.4f}{tag}")

# 5. Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Δ R²':>8}")
print("-"*55)
base_r2 = results['Crime-only']['R²']
for name, m in results.items():
    delta = m['R²'] - base_r2
    print(f"{name:<25} {m['R²']:>8.4f} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {delta:>+8.4f}")

# 6. Plot
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
r2s = [results[n]['R²'] for n in names]
colors = ['#2563eb', '#f59e0b', '#22c55e', '#a855f7']
bars = ax.bar(range(len(names)), r2s, color=colors, width=0.6)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Test R²')
ax.set_title('Data Fusion Comparison: Crime vs +IMD vs +Weather vs All', fontweight='bold')
ax.set_ylim(min(r2s) - 0.005, max(r2s) + 0.003)
for bar, v in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_weather_comparison.png")
plt.close()

db.close()
print(f"\n✅ Figure saved: {FIGURES_DIR / '01_weather_comparison.png'}")
