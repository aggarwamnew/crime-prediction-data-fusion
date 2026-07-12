"""33_bootstrap_ci.py — Bootstrap confidence intervals for the headline Delta R2 values.

Re-runs the load-bearing experiments with their EXACT original specifications
(aggregate: 11 base features, MIN_CRIMES=36, RF 200/depth15/leaf5/seed42;
per-type: 8 base features, MIN>=12, RF 100/depth12/leaf5/seed42), then computes
95% confidence intervals via a CLUSTER BOOTSTRAP BY LSOA on the held-out test set:
LSOAs are resampled with replacement (keeping all six test months of each sampled
LSOA), which respects within-LSOA correlation and is more conservative than
resampling rows i.i.d. Models are trained once; only the test set is resampled,
so the intervals quantify test-set sampling uncertainty conditional on the
train/test split (the split itself is not re-randomised).

Experiments covered:
  A1 baseline R2 (script 03)          A2 +IMD 2019 aggregate (script 04)
  A3 +Weather aggregate (script 07)   A4 full fusion, 51 features (script 20)
  P1 weapons +POI (script 13)         P2 drugs +IMD 2025 (script 11)
  P3 bicycle theft +Weather (11)      P4 theft-from-person +Temporal (15)
  P5 weapons +Station taps (30)

Output: data/processed/london/bootstrap_ci.csv + console table.
"""
import sys
sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')
from calendar import monthrange

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

B = 1000
RNG_SEED = 42
CENSUS = PROJECT_ROOT / "data/raw/london/census"
db = ThesisDB()
results = []

# ══════════════════════════════ shared loaders ══════════════════════════════

def load_weather():
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
    return pd.DataFrame(records), ['tmax', 'tmin', 'tmean', 'rain_mm', 'sun_hours']


def load_imd2019():
    imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd_2019_all_scores.csv")
    lsoa_col = [c for c in imd.columns if 'LSOA code' in c][0]
    cols = {
        lsoa_col: 'lsoa_code_2011',
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
    slim = imd[list(cols.keys())].rename(columns=cols)
    return slim, list(cols.values())[1:]


def load_imd2025():
    imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
    edu_col = [c for c in imd.columns if 'Education' in c and 'Score' in c][0]
    m = {'LSOA code (2021)': 'lsoa_code', 'Index of Multiple Deprivation (IMD) Score': 'imd_score',
         'Income Score (rate)': 'income_score', 'Employment Score (rate)': 'employment_score',
         edu_col: 'education_score', 'Health Deprivation and Disability Score': 'health_score',
         'Crime Score': 'crime_deprivation_score', 'Barriers to Housing and Services Score': 'housing_score',
         'Living Environment Score': 'living_environment_score'}
    slim = imd[list(m.keys())].rename(columns=m)
    return slim, list(m.values())[1:]


def load_demo():
    ts006 = pd.read_csv(CENSUS / "ts006/census2021-ts006-lsoa.csv").rename(
        columns={'geography code': 'lsoa_code',
                 'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})[
        ['lsoa_code', 'pop_density']]
    ts007 = pd.read_csv(CENSUS / "ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code': 'lsoa_code'})
    T = 'Age: Total'
    ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007[T]
    ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007[T]
    ts007['pct_25_44'] = (ts007['Age: Aged 25 to 29 years'] + ts007['Age: Aged 30 to 34 years'] + ts007['Age: Aged 35 to 39 years'] + ts007['Age: Aged 40 to 44 years']) / ts007[T]
    ts007['pct_45_64'] = (ts007['Age: Aged 45 to 49 years'] + ts007['Age: Aged 50 to 54 years'] + ts007['Age: Aged 55 to 59 years'] + ts007['Age: Aged 60 to 64 years']) / ts007[T]
    ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007[T]
    ts007['total_pop'] = ts007[T]
    demo = ts006.merge(ts007[['lsoa_code', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']], on='lsoa_code')
    return demo, ['pop_density', 'total_pop', 'pct_under15', 'pct_15_24', 'pct_25_44', 'pct_45_64', 'pct_65plus']


def load_temporal_15():
    """Temporal activity features exactly as script 15 (rounded to 3 dp)."""
    holidays = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/london_school_holidays.csv",
                           parse_dates=['start_date', 'end_date'])
    bank = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/uk_bank_holidays.csv", parse_dates=['date'])
    rows = []
    for ms in pd.date_range('2023-02-01', '2026-01-01', freq='MS'):
        y, m = ms.year, ms.month
        dim = monthrange(y, m)[1]
        me = pd.Timestamp(y, m, dim)
        school = sum(1 for d in pd.date_range(ms, me)
                     if any((d >= r['start_date']) & (d <= r['end_date']) for _, r in holidays.iterrows()))
        bh = bank[(bank['date'] >= ms) & (bank['date'] <= me)].shape[0]
        rows.append({'month': f"{y}-{m:02d}", 'pct_holiday_days': round(school / dim, 3),
                     'pct_bank_holiday_days': round(bh / dim, 3)})
    return pd.DataFrame(rows), ['pct_holiday_days', 'pct_bank_holiday_days']


# ══════════════════════════════ panel builders ══════════════════════════════

def agg_panel(monthly, min_crimes=36):
    """Aggregate panel with the 11 baseline features (script 03/04/07/13 pattern)."""
    totals = monthly.groupby('lsoa_code')['crime_count'].sum()
    active = totals[totals >= min_crimes].index
    monthly = monthly[monthly['lsoa_code'].isin(active)]
    months = sorted(monthly['month'].unique())
    lsoas = sorted(monthly['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([lsoas, months], names=['lsoa_code', 'month'])
    df = monthly.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    for w in [3, 6, 12]:
        df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['time_idx'] = df['month'].map({m: i for i, m in enumerate(months)})
    base = [f'lag_{l}' for l in [1, 2, 3, 6, 12]] + ['rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                                                     'month_sin', 'month_cos', 'time_idx']
    return df, months, base


def pt_panel(ct_monthly, min_crimes=12):
    """Per-type panel with the 8-feature base (script 11/13/15/30 pattern)."""
    totals = ct_monthly.groupby('lsoa_code')['crime_count'].sum()
    active = totals[totals >= min_crimes].index
    ct = ct_monthly[ct_monthly['lsoa_code'].isin(active)]
    months = sorted(ct['month'].unique())
    lsoas = sorted(ct['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([lsoas, months], names=['lsoa_code', 'month'])
    df = ct[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    base = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_12', 'month_sin', 'month_cos']
    return df, months, base


# ══════════════════════════════ bootstrap core ══════════════════════════════

def cluster_ci(test_df, pred_base, pred_fused=None, B=B, seed=RNG_SEED):
    """Paired cluster bootstrap by LSOA. Returns dict of point estimates + 95% CIs."""
    y = test_df['crime_count'].values
    codes, _ = pd.factorize(test_df['lsoa_code'].values)
    n_lsoa = codes.max() + 1
    group_idx = [np.flatnonzero(codes == g) for g in range(n_lsoa)]
    rng = np.random.default_rng(seed)
    r2b = np.empty(B)
    r2d = np.empty(B) if pred_fused is not None else None
    for b in range(B):
        sample = rng.integers(0, n_lsoa, n_lsoa)
        idx = np.concatenate([group_idx[g] for g in sample])
        r2b[b] = r2_score(y[idx], pred_base[idx])
        if pred_fused is not None:
            r2d[b] = r2_score(y[idx], pred_fused[idx]) - r2b[b]
    out = {'r2_base': r2_score(y, pred_base),
           'r2_base_lo': np.percentile(r2b, 2.5), 'r2_base_hi': np.percentile(r2b, 97.5)}
    if pred_fused is not None:
        out['delta'] = r2_score(y, pred_fused) - out['r2_base']
        out['delta_lo'] = np.percentile(r2d, 2.5)
        out['delta_hi'] = np.percentile(r2d, 97.5)
    return out


def run_experiment(name, df, months, base_feats, layer_feats, rf_params):
    """Train base/fused once, bootstrap the test set, record the result row."""
    need = base_feats + layer_feats
    d = df.dropna(subset=need + ['crime_count'])
    test_months = months[-6:]
    train = d[~d['month'].isin(test_months)]
    test = d[d['month'].isin(test_months)].reset_index(drop=True)
    rf1 = RandomForestRegressor(**rf_params)
    rf1.fit(train[base_feats], train['crime_count'])
    pb = rf1.predict(test[base_feats])
    pf = None
    if layer_feats:
        rf2 = RandomForestRegressor(**rf_params)
        rf2.fit(train[base_feats + layer_feats], train['crime_count'])
        pf = rf2.predict(test[base_feats + layer_feats])
    ci = cluster_ci(test, pb, pf)
    row = {'experiment': name, 'n_lsoas': test['lsoa_code'].nunique(), 'n_test': len(test), **ci}
    results.append(row)
    msg = f"{name:38s} R2={ci['r2_base']:.4f} [{ci['r2_base_lo']:.4f},{ci['r2_base_hi']:.4f}]"
    if pf is not None:
        msg += f"  dR2={ci['delta']:+.4f} [{ci['delta_lo']:+.4f},{ci['delta_hi']:+.4f}]"
    print(msg, flush=True)


RF_AGG = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
RF_PT = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)

print("=" * 100)
print(f"BOOTSTRAP CONFIDENCE INTERVALS  (cluster bootstrap by LSOA, B={B}, seed={RNG_SEED})")
print("=" * 100)

weather, weather_feats = load_weather()
monthly_all = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")

# ── A1: baseline (script 03 spec) ──
df, months, base11 = agg_panel(monthly_all.copy())
run_experiment("A1 baseline (all crime)", df, months, base11, [], RF_AGG)

# ── A2: +IMD 2019 (script 04 spec: left join on 2011 codes, drop unmatched) ──
imd19, imd19_feats = load_imd2019()
m = monthly_all.merge(imd19, left_on='lsoa_code', right_on='lsoa_code_2011', how='left')
m = m.dropna(subset=['imd_score'])
df, months, _ = agg_panel(m[['lsoa_code', 'month', 'crime_count']])
imd_per = m.drop_duplicates('lsoa_code')[['lsoa_code'] + imd19_feats]
df = df.merge(imd_per, on='lsoa_code', how='left')
run_experiment("A2 +IMD 2019 (aggregate)", df, months, base11, imd19_feats, RF_AGG)

# ── A3: +Weather on the IMD-matched set (script 07 spec) ──
df = df.merge(weather[['month'] + weather_feats], on='month', how='left')
run_experiment("A3 +Weather (aggregate)", df, months, base11, weather_feats, RF_AGG)

# ── A4: full fusion, 51 features (script 20 spec) ──
imd25, imd25_feats = load_imd2025()
demo20 = pd.read_csv(CENSUS / "ts006/census2021-ts006-lsoa.csv").rename(
    columns={'geography code': 'lsoa_code', 'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})[['lsoa_code', 'pop_density']]
ts007 = pd.read_csv(CENSUS / "ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code': 'lsoa_code'})
ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007['Age: Total']
ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007['Age: Total']
ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007['Age: Total']
demo20 = demo20.merge(ts007[['lsoa_code', 'pct_under15', 'pct_15_24', 'pct_65plus']], on='lsoa_code')
demo20_feats = ['pop_density', 'pct_under15', 'pct_15_24', 'pct_65plus']

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

temporal20_rows = []
holidays = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/london_school_holidays.csv", parse_dates=['start_date', 'end_date'])
bank_hols = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/uk_bank_holidays.csv", parse_dates=['date'])
for ms in pd.date_range('2023-02-01', '2026-01-01', freq='MS'):
    y_, m_ = ms.year, ms.month
    dim = monthrange(y_, m_)[1]
    me = pd.Timestamp(y_, m_, dim)
    hol = sum(1 for dd in pd.date_range(ms, me) if any((dd >= r['start_date']) & (dd <= r['end_date']) for _, r in holidays.iterrows()))
    bh = len(bank_hols[(bank_hols['date'] >= ms) & (bank_hols['date'] <= me)])
    temporal20_rows.append({'month': f"{y_}-{m_:02d}", 'pct_holiday': hol / dim, 'pct_bank_holiday': bh / dim})
temporal20 = pd.DataFrame(temporal20_rows)
temporal20_feats = ['pct_holiday', 'pct_bank_holiday']

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

df20, months20, _ = agg_panel(monthly_all.copy())
for d_ in [imd25, demo20, poi, hprice[['lsoa_code'] + housing_feats], samhi,
           edu[['lsoa_code', 'pct_no_qual', 'pct_level4_plus', 'pct_apprentice']],
           hh[['lsoa_code', 'pct_one_person', 'pct_lone_parent_dep']]]:
    df20 = df20.merge(d_, on='lsoa_code', how='left')
df20 = df20.merge(weather[['month'] + weather_feats], on='month', how='left')
df20 = df20.merge(temporal20, on='month', how='left')
df20[poi_feats] = df20[poi_feats].fillna(0)
all_supp = imd25_feats + weather_feats + demo20_feats + poi_feats + housing_feats + temporal20_feats + ss_feats
df20 = df20.dropna(subset=imd25_feats + demo20_feats + housing_feats + ['samhi_index'])
run_experiment("A4 full fusion (51 features)", df20, months20, base11, all_supp, RF_AGG)

# ── per-type experiments ──
def ct_monthly(ct):
    q = ct.replace("'", "''")
    return db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{q}' GROUP BY lsoa_code, month")

# P1: weapons +POI (script 13 spec: zero-filled POI counts)
poi_full_all = monthly_all[['lsoa_code']].drop_duplicates().merge(poi, on='lsoa_code', how='left').fillna(0)
wp = ct_monthly('Possession of weapons')
df, months, base8 = pt_panel(wp)
df = df.merge(poi_full_all, on='lsoa_code', how='left')
df[poi_feats] = df[poi_feats].fillna(0)
run_experiment("P1 weapons +POI", df, months, base8, poi_feats, RF_PT)

# P2/P3: drugs +IMD2025 and bicycle +Weather on the script-11 panel (inner join IMD2025 + demo)
demo11, demo11_feats = load_demo()
for ct, layer_name, layer_feats_ in [('Drugs', 'IMD 2025', None), ('Bicycle theft', 'Weather', None)]:
    cm = ct_monthly(ct)
    merged = cm.merge(imd25, on='lsoa_code', how='inner').merge(demo11, on='lsoa_code', how='inner')
    df, months, base8 = pt_panel(merged[['lsoa_code', 'month', 'crime_count']])
    imd_per = merged.drop_duplicates('lsoa_code')[['lsoa_code'] + imd25_feats]
    df = df.merge(imd_per, on='lsoa_code', how='left')
    df = df.merge(weather[['month'] + weather_feats], on='month', how='left')
    feats = imd25_feats if ct == 'Drugs' else weather_feats
    run_experiment(f"P{'2' if ct=='Drugs' else '3'} {ct.lower()} +{layer_name if ct=='Drugs' else 'Weather'}",
                   df, months, base8, feats, RF_PT)

# P4: theft from the person +Temporal (script 15 spec)
tf_df, tf_feats = load_temporal_15()
tp = ct_monthly('Theft from the person')
df, months, base8 = pt_panel(tp)
df = df.merge(tf_df, on='month', how='left')
run_experiment("P4 theft-from-person +Temporal", df, months, base8, tf_feats, RF_PT)

# P5: weapons +Station taps (script 30 spec: processed monthly ridership, zero-filled)
ride = pd.read_csv(PROJECT_ROOT / "data/processed/london/station_ridership_monthly.csv")
ride_feats = ['ridership_total', 'station_count', 'ridership_per_station']
df, months, base8 = pt_panel(wp)
df = df.merge(ride[['lsoa_code', 'month'] + ride_feats], on=['lsoa_code', 'month'], how='left')
df[ride_feats] = df[ride_feats].fillna(0)
run_experiment("P5 weapons +Station taps", df, months, base8, ride_feats, RF_PT)

db.close()

out = pd.DataFrame(results)
out_path = PROJECT_ROOT / "data/processed/london/bootstrap_ci.csv"
out.to_csv(out_path, index=False)
print("\n" + "=" * 100)
print(out.to_string(index=False))
print(f"\nSaved -> {out_path}")
