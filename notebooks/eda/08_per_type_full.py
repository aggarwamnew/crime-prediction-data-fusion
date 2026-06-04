"""
Per-Crime-Type: IMD vs Weather vs Both
Tests which data layer helps which crime type.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "per_type"
db = ThesisDB()

# Load IMD
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
imd_map = {
    'LSOA code (2011)': 'lsoa_code_2011',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Income Score (rate)': 'income_score', 'Employment Score (rate)': 'employment_score',
    'Education, Skills and Training Score': 'education_score',
    'Health Deprivation and Disability Score': 'health_score',
    'Crime Score': 'crime_deprivation_score',
    'Barriers to Housing and Services Score': 'housing_score',
    'Living Environment Score': 'living_environment_score',
}
imd_slim = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_features = list(imd_map.values())[1:]

# Parse weather
weather_path = PROJECT_ROOT / "data/raw/london/weather/heathrow_monthly.txt"
records = []
for line in weather_path.read_text().strip().split('\n'):
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        try:
            y, m = int(parts[0]), int(parts[1])
            tmax = float(parts[2].replace('*','')) if '---' not in parts[2] else None
            tmin = float(parts[3].replace('*','')) if '---' not in parts[3] else None
            rain = float(parts[5].replace('*','')) if '---' not in parts[5] else None
            sun = float(parts[6].replace('*','').replace('#','')) if '---' not in parts[6] else None
            records.append({'month': f"{y}-{m:02d}", 'tmax': tmax, 'tmin': tmin, 'tmean': (tmax+tmin)/2 if tmax and tmin else None, 'rain_mm': rain, 'sun_hours': sun})
        except: continue
weather = pd.DataFrame(records)
weather_features = ['tmax', 'tmin', 'tmean', 'rain_mm', 'sun_hours']

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
print(f"Running for {len(crime_types)} crime types...\n")

results = []
for ct in crime_types:
    print(f"  {ct}...", end=" ", flush=True)
    monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")
    merged = monthly.merge(imd_slim, left_on='lsoa_code', right_on='lsoa_code_2011', how='inner')
    lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= 12].index
    if len(active) < 50:
        print("SKIP"); continue
    merged = merged[merged['lsoa_code'].isin(active)]
    
    all_months = sorted(merged['month'].unique())
    all_lsoas = sorted(merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
    imd_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + imd_features].set_index('lsoa_code')
    df = merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
    df = df.merge(imd_per, on='lsoa_code', how='left')
    df = df.merge(weather[['month'] + weather_features], on='month', how='left')
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    
    df_model = df.dropna()
    if len(df_model) < 100: print("SKIP"); continue
    test_months = all_months[-6:]
    train = df_model[~df_model['month'].isin(test_months)]
    test = df_model[df_model['month'].isin(test_months)]
    if len(test) < 50: print("SKIP"); continue
    
    lag_cols = ['lag_1','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_12','month_sin','month_cos']
    
    variants = {
        'crime_only': lag_cols,
        'imd': lag_cols + imd_features,
        'weather': lag_cols + weather_features,
        'all': lag_cols + imd_features + weather_features,
    }
    
    row = {'crime_type': ct, 'n_lsoas': len(active)}
    for vname, cols in variants.items():
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(train[cols], train['crime_count'])
        row[f'r2_{vname}'] = r2_score(test['crime_count'], rf.predict(test[cols]))
    
    row['delta_imd'] = row['r2_imd'] - row['r2_crime_only']
    row['delta_weather'] = row['r2_weather'] - row['r2_crime_only']
    row['delta_all'] = row['r2_all'] - row['r2_crime_only']
    
    print(f"Δ IMD={row['delta_imd']:+.4f}  Δ Weather={row['delta_weather']:+.4f}  Δ All={row['delta_all']:+.4f}")
    results.append(row)

db.close()

# Results table
res = pd.DataFrame(results).sort_values('delta_all', ascending=False)

print("\n" + "="*100)
print("PER-CRIME-TYPE: IMD vs WEATHER vs COMBINED")
print("="*100)
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ IMD':>8} {'Δ Weath':>8} {'Δ All':>8} {'Best':>10}")
print("-"*80)
for _, r in res.iterrows():
    best = 'IMD' if r['delta_imd'] > r['delta_weather'] else 'Weather'
    print(f"{r['crime_type']:<35} {r['r2_crime_only']:>8.4f} {r['delta_imd']:>+8.4f} {r['delta_weather']:>+8.4f} {r['delta_all']:>+8.4f} {best:>10}")

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(res))
w = 0.25
ax.barh(x - w, res['delta_imd'], w, label='+ IMD', color='#f59e0b', alpha=0.9)
ax.barh(x, res['delta_weather'], w, label='+ Weather', color='#22c55e', alpha=0.9)
ax.barh(x + w, res['delta_all'], w, label='+ Both', color='#a855f7', alpha=0.9)
ax.set_yticks(x)
ax.set_yticklabels(res['crime_type'])
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Δ R² (improvement over crime-only baseline)')
ax.set_title('Data Fusion Impact by Crime Type: IMD vs Weather vs Combined', fontweight='bold', fontsize=13)
ax.legend(loc='lower right')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_imd_vs_weather_by_type.png")
plt.close()
print(f"\n✅ Figure: {FIGURES_DIR / '02_imd_vs_weather_by_type.png'}")
