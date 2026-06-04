"""
Per-Crime-Type: IMD vs Weather vs Demographics vs All
Extends 08_per_type_full to include Census demographics as a 4th data layer.
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
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CENSUS_DIR = PROJECT_ROOT / "data/raw/london/census"
db = ThesisDB()

# ── Load IMD 2025 (better match rate) ──
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
edu_col = [c for c in imd.columns if 'Education' in c and 'Score' in c][0]
imd_map = {
    'LSOA code (2021)': 'lsoa_code',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Income Score (rate)': 'income_score', 'Employment Score (rate)': 'employment_score',
    edu_col: 'education_score',
    'Health Deprivation and Disability Score': 'health_score',
    'Crime Score': 'crime_deprivation_score',
    'Barriers to Housing and Services Score': 'housing_score',
    'Living Environment Score': 'living_environment_score',
}
imd_slim = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_features = list(imd_map.values())[1:]

# ── Load weather ──
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

# ── Load demographics ──
ts006 = pd.read_csv(CENSUS_DIR / "ts006/census2021-ts006-lsoa.csv")
ts006 = ts006.rename(columns={'geography code': 'lsoa_code', 'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})[['lsoa_code', 'pop_density']]

ts007 = pd.read_csv(CENSUS_DIR / "ts007a/census2021-ts007a-lsoa.csv")
ts007 = ts007.rename(columns={'geography code': 'lsoa_code'})
total_col = 'Age: Total'
ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007[total_col]
ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007[total_col]
ts007['pct_25_44'] = (ts007['Age: Aged 25 to 29 years'] + ts007['Age: Aged 30 to 34 years'] + ts007['Age: Aged 35 to 39 years'] + ts007['Age: Aged 40 to 44 years']) / ts007[total_col]
ts007['pct_45_64'] = (ts007['Age: Aged 45 to 49 years'] + ts007['Age: Aged 50 to 54 years'] + ts007['Age: Aged 55 to 59 years'] + ts007['Age: Aged 60 to 64 years']) / ts007[total_col]
ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007[total_col]
ts007['total_pop'] = ts007[total_col]
demo = ts006.merge(ts007[['lsoa_code', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']], on='lsoa_code')
demo_features = ['pop_density', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']

# ── Run per-crime-type ──
crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
print(f"Running for {len(crime_types)} crime types with 5 variants...\n")

results = []
for ct in crime_types:
    print(f"  {ct}...", end=" ", flush=True)
    monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")

    # Join all layers
    merged = monthly.merge(imd_slim, on='lsoa_code', how='inner')
    merged = merged.merge(demo, on='lsoa_code', how='inner')
    lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= 12].index
    if len(active) < 50:
        print("SKIP"); continue
    merged = merged[merged['lsoa_code'].isin(active)]

    all_months = sorted(merged['month'].unique())
    all_lsoas = sorted(merged['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

    imd_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + imd_features].set_index('lsoa_code')
    demo_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + demo_features].set_index('lsoa_code')

    df = merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
    df = df.merge(imd_per, on='lsoa_code', how='left')
    df = df.merge(demo_per, on='lsoa_code', how='left')
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
        'demo': lag_cols + demo_features,
        'all': lag_cols + imd_features + weather_features + demo_features,
    }

    row = {'crime_type': ct, 'n_lsoas': len(active)}
    for vname, cols in variants.items():
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(train[cols], train['crime_count'])
        row[f'r2_{vname}'] = r2_score(test['crime_count'], rf.predict(test[cols]))

    row['delta_imd'] = row['r2_imd'] - row['r2_crime_only']
    row['delta_weather'] = row['r2_weather'] - row['r2_crime_only']
    row['delta_demo'] = row['r2_demo'] - row['r2_crime_only']
    row['delta_all'] = row['r2_all'] - row['r2_crime_only']

    print(f"Δ IMD={row['delta_imd']:+.4f}  Δ Weath={row['delta_weather']:+.4f}  Δ Demo={row['delta_demo']:+.4f}  Δ All={row['delta_all']:+.4f}")
    results.append(row)

db.close()

# ── Results table ──
res = pd.DataFrame(results).sort_values('delta_all', ascending=False)

print("\n" + "=" * 110)
print("PER-CRIME-TYPE: IMD vs WEATHER vs DEMOGRAPHICS vs ALL")
print("=" * 110)
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ IMD':>8} {'Δ Weath':>8} {'Δ Demo':>8} {'Δ All':>8} {'Best Layer':>12}")
print("-" * 95)
for _, r in res.iterrows():
    deltas = {'IMD': r['delta_imd'], 'Weather': r['delta_weather'], 'Demo': r['delta_demo']}
    best = max(deltas, key=deltas.get)
    print(f"{r['crime_type']:<35} {r['r2_crime_only']:>8.4f} {r['delta_imd']:>+8.4f} {r['delta_weather']:>+8.4f} {r['delta_demo']:>+8.4f} {r['delta_all']:>+8.4f} {best:>12}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(14, 9))
x = np.arange(len(res))
w = 0.2
ax.barh(x - 1.5*w, res['delta_imd'], w, label='+ IMD', color='#f59e0b', alpha=0.9)
ax.barh(x - 0.5*w, res['delta_weather'], w, label='+ Weather', color='#22c55e', alpha=0.9)
ax.barh(x + 0.5*w, res['delta_demo'], w, label='+ Demographics', color='#3b82f6', alpha=0.9)
ax.barh(x + 1.5*w, res['delta_all'], w, label='+ All', color='#a855f7', alpha=0.9)
ax.set_yticks(x)
ax.set_yticklabels(res['crime_type'])
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Δ R² (improvement over crime-only baseline)')
ax.set_title('Data Fusion Impact by Crime Type: IMD vs Weather vs Demographics vs All', fontweight='bold', fontsize=13)
ax.legend(loc='lower right')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_all_layers_by_type.png", dpi=150)
plt.close()
print(f"\n✅ Figure: {FIGURES_DIR / '03_all_layers_by_type.png'}")
print("✅ Complete!")
