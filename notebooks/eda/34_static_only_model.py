"""34_static_only_model.py — What do supplementary features predict WITHOUT crime history?

Quantifies the "explains versus predicts" distinction: trains Random Forests with NO
lag or rolling features, using only supplementary data, on the full-fusion panel
(script 20 spec: 4,078 LSOAs with complete data, same temporal split, RF 200/15/5/42).

Configurations:
  1. IMD 2025 only (7 deprivation scores)            — pure deprivation
  2. All static area features (IMD+demo+POI+housing+SS, 31) — everything time-invariant
  3. All 40 supplementary (adds weather + temporal)  — everything except crime history
  4. Crime history baseline (11 features)            — reference

Expected: static-only R2 well below the history baseline, quantifying how much of the
headline R2 is 'explained level' vs 'predicted change'.
"""
import sys
sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')
from calendar import monthrange

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

CENSUS = PROJECT_ROOT / "data/raw/london/census"
db = ThesisDB()

# ── layers (script 20 spec) ──
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
edu_col = [c for c in imd.columns if 'Education' in c and 'Score' in c][0]
imd_map = {'LSOA code (2021)': 'lsoa_code', 'Index of Multiple Deprivation (IMD) Score': 'imd_score',
           'Income Score (rate)': 'income_score', 'Employment Score (rate)': 'employment_score',
           edu_col: 'imd_education_score', 'Health Deprivation and Disability Score': 'health_score',
           'Crime Score': 'crime_dep_score', 'Barriers to Housing and Services Score': 'barriers_score',
           'Living Environment Score': 'living_env_score'}
imd = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_feats = list(imd_map.values())[1:]

records = []
for line in (PROJECT_ROOT / "data/raw/london/weather/heathrow_monthly.txt").read_text().strip().split('\n'):
    p = line.split()
    if len(p) >= 7 and p[0].isdigit():
        try:
            tmax = float(p[2].replace('*', '')); tmin = float(p[3].replace('*', ''))
            rain = float(p[5].replace('*', '')); sun = float(p[6].replace('*', '').replace('#', ''))
            records.append({'month': f"{int(p[0])}-{int(p[1]):02d}", 'tmax': tmax, 'tmin': tmin,
                            'tmean': (tmax + tmin) / 2, 'rain_mm': rain, 'sun_hours': sun})
        except Exception:
            continue
weather = pd.DataFrame(records)
weather_feats = ['tmax', 'tmin', 'tmean', 'rain_mm', 'sun_hours']

ts006 = pd.read_csv(CENSUS / "ts006/census2021-ts006-lsoa.csv").rename(
    columns={'geography code': 'lsoa_code', 'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})[['lsoa_code', 'pop_density']]
ts007 = pd.read_csv(CENSUS / "ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code': 'lsoa_code'})
ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007['Age: Total']
ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007['Age: Total']
ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007['Age: Total']
demo = ts006.merge(ts007[['lsoa_code', 'pct_under15', 'pct_15_24', 'pct_65plus']], on='lsoa_code')
demo_feats = ['pop_density', 'pct_under15', 'pct_15_24', 'pct_65plus']

poi = pd.read_csv(PROJECT_ROOT / "data/raw/london/pois/poi_counts_per_lsoa.csv")
poi_feats = [c for c in poi.columns if c.startswith('poi_')]

hprice = pd.read_excel(PROJECT_ROOT / "data/raw/london/housing/ons_house_prices_lsoa.xlsx", skiprows=4,
                       header=None, names=['la_code', 'la_name', 'lsoa_code', 'lsoa_name', 'median_house_price'])
hprice = hprice[hprice['la_code'].astype(str).str.startswith('E09')].copy()
hprice['median_house_price'] = pd.to_numeric(hprice['median_house_price'], errors='coerce')
hprice = hprice[['lsoa_code', 'median_house_price']].dropna()
q = hprice['median_house_price'].quantile([1 / 3, 2 / 3])
hprice['price_low'] = (hprice['median_house_price'] < q.iloc[0]).astype(int)
hprice['price_mid'] = ((hprice['median_house_price'] >= q.iloc[0]) & (hprice['median_house_price'] < q.iloc[1])).astype(int)
hprice['price_high'] = (hprice['median_house_price'] >= q.iloc[1]).astype(int)
housing_feats = ['price_low', 'price_mid', 'price_high']

holidays = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/london_school_holidays.csv", parse_dates=['start_date', 'end_date'])
bank_hols = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/uk_bank_holidays.csv", parse_dates=['date'])
temporal_rows = []
for ms in pd.date_range('2023-02-01', '2026-01-01', freq='MS'):
    y_, m_ = ms.year, ms.month
    dim = monthrange(y_, m_)[1]
    me = pd.Timestamp(y_, m_, dim)
    hol = sum(1 for dd in pd.date_range(ms, me) if any((dd >= r['start_date']) & (dd <= r['end_date']) for _, r in holidays.iterrows()))
    bh = len(bank_hols[(bank_hols['date'] >= ms) & (bank_hols['date'] <= me)])
    temporal_rows.append({'month': f"{y_}-{m_:02d}", 'pct_holiday': hol / dim, 'pct_bank_holiday': bh / dim})
temporal = pd.DataFrame(temporal_rows)
temporal_feats = ['pct_holiday', 'pct_bank_holiday']

samhi = pd.read_csv(PROJECT_ROOT / "data/raw/london/mental_health/samhi_lsoa.csv")[['lsoa11', 'samhi_index.2022', 'samhi_dec.2022']]
samhi.columns = ['lsoa_code', 'samhi_index', 'samhi_decile']
edu_raw = pd.read_csv(CENSUS / "ts067/census2021-ts067-lsoa.csv")
edu_total = [c for c in edu_raw.columns if 'Total' in c][0]
edu = pd.DataFrame({'lsoa_code': edu_raw['geography code'], '_t': edu_raw[edu_total]})
for orig, short in [('No qualifications', 'pct_no_qual'), ('Level 4 qualifications and above', 'pct_level4_plus'), ('Apprenticeship', 'pct_apprentice')]:
    edu[short] = edu_raw[[c for c in edu_raw.columns if orig in c][0]] / edu['_t'] * 100
hh_raw = pd.read_csv(CENSUS / "ts003/census2021-ts003-lsoa.csv")
hh_total = [c for c in hh_raw.columns if 'Total' in c][0]
hh = pd.DataFrame({'lsoa_code': hh_raw['geography code'], '_t': hh_raw[hh_total]})
for pat, nm in [('One person household; measures', 'pct_one_person'), ('Lone parent family: With dependent', 'pct_lone_parent_dep')]:
    hh[nm] = hh_raw[[c for c in hh_raw.columns if pat in c][0]] / hh['_t'] * 100
ss_feats = ['samhi_index', 'samhi_decile', 'pct_no_qual', 'pct_level4_plus', 'pct_apprentice', 'pct_one_person', 'pct_lone_parent_dep']

# ── panel (script 20 spec) ──
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
active = crime.groupby('lsoa_code')['crime_count'].sum()
active = active[active >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]
all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = crime.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rm_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year - ts.dt.year.min()) * 12 + ts.dt.month
base_feats = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'rm_3', 'rm_6', 'rm_12', 'month_sin', 'month_cos', 'time_idx']

for d_ in [imd, demo, poi, hprice[['lsoa_code'] + housing_feats], samhi,
           edu[['lsoa_code', 'pct_no_qual', 'pct_level4_plus', 'pct_apprentice']],
           hh[['lsoa_code', 'pct_one_person', 'pct_lone_parent_dep']]]:
    df = df.merge(d_, on='lsoa_code', how='left')
df = df.merge(weather[['month'] + weather_feats], on='month', how='left')
df = df.merge(temporal, on='month', how='left')
df[poi_feats] = df[poi_feats].fillna(0)
df_model = df.dropna(subset=base_feats + imd_feats + demo_feats + housing_feats + ['samhi_index'])

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]
print(f"Panel: {df_model['lsoa_code'].nunique():,} LSOAs | train {len(train):,} / test {len(test):,}\n")

static_area = imd_feats + demo_feats + poi_feats + housing_feats + ss_feats
all_supp = static_area + weather_feats + temporal_feats
configs = [
    ('IMD 2025 only (no history)', imd_feats),
    ('All static area features (no history)', static_area),
    ('All 40 supplementary (no history)', all_supp),
    ('Crime history baseline (reference)', base_feats),
]

print(f"{'Configuration':45s} | {'R2':>7s} | {'MAE':>6s} | Feats")
print('-' * 75)
for name, feats in configs:
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(train[feats], train['crime_count'])
    pred = rf.predict(test[feats])
    print(f"{name:45s} | {r2_score(test['crime_count'], pred):.4f} | {mean_absolute_error(test['crime_count'], pred):6.2f} | {len(feats):>3d}", flush=True)

db.close()
print("\nDone.")
