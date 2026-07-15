"""
06_rf_hyperparameter_tuning.py — Capacity / hyperparameter sensitivity of the RF baseline.

Question (examiner robustness, examiner_open_items #7): the thesis fixes the Random
Forest at 200 trees / max_depth 15 / min_samples_leaf 5 a priori (never tuned). Could a
tuned or higher-capacity RF (a) beat the crime-only baseline R^2 = 0.943, or (b) extract
more from the fused layers than the a-priori model (i.e. grow the near-zero fusion lift)?

Design (leakage-free):
  - Hyperparameters are selected on a VALIDATION window (the 6 months immediately before
    the test set); the winner is then evaluated ONCE on the true held-out test set (last 6
    months). The a-priori config (200/15/5) is reported on the same test set as reference.
  - Part A uses the crime-only baseline panel (>=36 crimes, 5,148 LSOAs, 11 features).
  - Part B uses the full-fusion panel (all 10 supplementary layers, 51 features) and reports
    the fusion lift dR^2 = R^2(full) - R^2(baseline) under BOTH the a-priori and the tuned
    (full-model-selected) configuration, on the identical panel.

Parked result; not wired into the thesis body unless the outcome warrants it.

Usage:
    python notebooks/experimental/06_rf_hyperparameter_tuning.py
"""
import sys; sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')
import pandas as pd, numpy as np
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

CENSUS = PROJECT_ROOT / "data/raw/london/census"
OUT = PROJECT_ROOT / "data/processed/london/rf_tuning.csv"
SEED = 42

# A-priori configuration used throughout the thesis.
APRIORI = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, max_features=1.0)

# Curated grid spanning capacity: deeper trees, no depth cap, smaller leaves, more trees,
# and decorrelated splits (max_features='sqrt'). Deliberately biased UPWARD in capacity,
# since the question is whether a *stronger* model can do better.
GRID = [
    dict(n_estimators=200, max_depth=15,   min_samples_leaf=5, max_features=1.0),   # a-priori
    dict(n_estimators=200, max_depth=20,   min_samples_leaf=2, max_features=1.0),
    dict(n_estimators=200, max_depth=25,   min_samples_leaf=5, max_features=1.0),
    dict(n_estimators=200, max_depth=25,   min_samples_leaf=1, max_features=1.0),
    dict(n_estimators=200, max_depth=None, min_samples_leaf=5, max_features=1.0),
    dict(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features=1.0),
    dict(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features=1.0),
    dict(n_estimators=500, max_depth=25,   min_samples_leaf=2, max_features=1.0),
    dict(n_estimators=200, max_depth=15,   min_samples_leaf=1, max_features=1.0),
    dict(n_estimators=200, max_depth=25,   min_samples_leaf=2, max_features='sqrt'),
    dict(n_estimators=500, max_depth=None, min_samples_leaf=2, max_features='sqrt'),
    dict(n_estimators=300, max_depth=20,   min_samples_leaf=2, max_features=1.0),
]


def cfg_str(c):
    d = c['max_depth'] if c['max_depth'] is not None else 'None'
    mf = c['max_features']
    return f"trees={c['n_estimators']}, depth={d}, leaf={c['min_samples_leaf']}, maxfeat={mf}"


def fit_eval(cfg, Xtr, ytr, Xev, yev):
    rf = RandomForestRegressor(n_jobs=-1, random_state=SEED, **cfg)
    rf.fit(Xtr, ytr)
    p = rf.predict(Xev)
    return r2_score(yev, p), mean_absolute_error(yev, p)


def search(df_model, feat_cols, all_months, label):
    """Select hyperparameters on the validation window, report the winner on test."""
    test_months = all_months[-6:]
    val_months = all_months[-12:-6]
    search_train_months = [m for m in all_months if m not in test_months and m not in val_months]
    refit_train_months = [m for m in all_months if m not in test_months]  # includes val

    st = df_model[df_model['month'].isin(search_train_months)]
    va = df_model[df_model['month'].isin(val_months)]
    rf_train = df_model[df_model['month'].isin(refit_train_months)]
    te = df_model[df_model['month'].isin(test_months)]

    print(f"\n[{label}] LSOAs={df_model['lsoa_code'].nunique():,}  feats={len(feat_cols)}  "
          f"search-train={len(st):,}  val={len(va):,}  refit-train={len(rf_train):,}  test={len(te):,}")
    print(f"[{label}] val window: {val_months[0]}..{val_months[-1]}   test window: {test_months[0]}..{test_months[-1]}")

    rows = []
    for cfg in GRID:
        r2v, maev = fit_eval(cfg, st[feat_cols], st['crime_count'], va[feat_cols], va['crime_count'])
        rows.append({'panel': label, **cfg, 'val_r2': r2v, 'val_mae': maev})
        print(f"   {cfg_str(cfg):55s}  val R2={r2v:.4f}")

    res = pd.DataFrame(rows)
    best = res.sort_values('val_r2', ascending=False).iloc[0].to_dict()
    best_cfg = {k: best[k] for k in ('n_estimators', 'max_depth', 'min_samples_leaf', 'max_features')}
    best_cfg['max_depth'] = None if (isinstance(best_cfg['max_depth'], float) and np.isnan(best_cfg['max_depth'])) else best_cfg['max_depth']

    # Report a-priori and tuned winner on the true test set (refit on all non-test months).
    ap_r2, ap_mae = fit_eval(APRIORI, rf_train[feat_cols], rf_train['crime_count'], te[feat_cols], te['crime_count'])
    bt_r2, bt_mae = fit_eval(best_cfg, rf_train[feat_cols], rf_train['crime_count'], te[feat_cols], te['crime_count'])

    print(f"[{label}] TEST  a-priori ({cfg_str(APRIORI)}):  R2={ap_r2:.4f}  MAE={ap_mae:.3f}")
    print(f"[{label}] TEST  tuned    ({cfg_str(best_cfg)}):  R2={bt_r2:.4f}  MAE={bt_mae:.3f}")
    print(f"[{label}] TEST  capacity gain over a-priori: dR2 = {bt_r2 - ap_r2:+.4f}")

    return res, best_cfg, dict(ap_r2=ap_r2, ap_mae=ap_mae, bt_r2=bt_r2, bt_mae=bt_mae,
                               rf_train=rf_train, te=te)


# ═══════════════════════════════════════════════════════════════════════════
# PART A — crime-only baseline panel (>=36 crimes, 11 features)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("PART A — CRIME-ONLY BASELINE PANEL (does a stronger RF beat R2 = 0.943?)")
print("=" * 78)
db = ThesisDB()

mc = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
active = mc.groupby('lsoa_code')['crime_count'].sum()
active = active[active >= 36].index
mc = mc[mc['lsoa_code'].isin(active)]
all_months = sorted(mc['month'].unique())
all_lsoas = sorted(mc['lsoa_code'].unique())
grid_idx = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = mc.set_index(['lsoa_code', 'month']).reindex(grid_idx, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
mnum = pd.to_datetime(df['month']).dt.month
df['month_sin'] = np.sin(2 * np.pi * mnum / 12)
df['month_cos'] = np.cos(2 * np.pi * mnum / 12)
df['time_idx'] = df['month'].map({m: i for i, m in enumerate(all_months)})
base_feats = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
              'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
              'month_sin', 'month_cos', 'time_idx']
df_base = df.dropna(subset=base_feats).copy()

resA, bestA, sumA = search(df_base, base_feats, all_months, 'baseline-11feat')

# ═══════════════════════════════════════════════════════════════════════════
# PART B — full-fusion panel (51 features): does a stronger model grow the lift?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PART B — FULL-FUSION PANEL (does a tuned model extract more from the layers?)")
print("=" * 78)

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

ts006 = pd.read_csv(CENSUS / "ts006/census2021-ts006-lsoa.csv").rename(columns={'geography code': 'lsoa_code', 'Population Density: Persons per square kilometre; measures: Value': 'pop_density'})[['lsoa_code', 'pop_density']]
ts007 = pd.read_csv(CENSUS / "ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code': 'lsoa_code'})
ts007['pct_under15'] = (ts007['Age: Aged 4 years and under'] + ts007['Age: Aged 5 to 9 years'] + ts007['Age: Aged 10 to 14 years']) / ts007['Age: Total']
ts007['pct_15_24'] = (ts007['Age: Aged 15 to 19 years'] + ts007['Age: Aged 20 to 24 years']) / ts007['Age: Total']
ts007['pct_65plus'] = (ts007['Age: Aged 65 to 69 years'] + ts007['Age: Aged 70 to 74 years'] + ts007['Age: Aged 75 to 79 years'] + ts007['Age: Aged 80 to 84 years'] + ts007['Age: Aged 85 years and over']) / ts007['Age: Total']
demo = ts006.merge(ts007[['lsoa_code', 'pct_under15', 'pct_15_24', 'pct_65plus']], on='lsoa_code')
demo_feats = ['pop_density', 'pct_under15', 'pct_15_24', 'pct_65plus']

poi = pd.read_csv(PROJECT_ROOT / "data/raw/london/pois/poi_counts_per_lsoa.csv")
poi_feats = [c for c in poi.columns if c.startswith('poi_')]

hprice = pd.read_excel(PROJECT_ROOT / "data/raw/london/housing/ons_house_prices_lsoa.xlsx", skiprows=4, header=None,
                       names=['la_code', 'la_name', 'lsoa_code', 'lsoa_name', 'median_house_price'])
hprice = hprice[hprice['la_code'].astype(str).str.startswith('E09')].copy()
hprice['median_house_price'] = pd.to_numeric(hprice['median_house_price'], errors='coerce')
hprice = hprice[['lsoa_code', 'median_house_price']].dropna()
hp_thresh = hprice['median_house_price'].quantile([1/3, 2/3])
hp_low, hp_high = hp_thresh.iloc[0], hp_thresh.iloc[1]
hprice['price_low'] = (hprice['median_house_price'] < hp_low).astype(int)
hprice['price_mid'] = ((hprice['median_house_price'] >= hp_low) & (hprice['median_house_price'] < hp_high)).astype(int)
hprice['price_high'] = (hprice['median_house_price'] >= hp_high).astype(int)
housing_feats = ['price_low', 'price_mid', 'price_high']

holidays = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/london_school_holidays.csv", parse_dates=['start_date', 'end_date'])
bank_hols = pd.read_csv(PROJECT_ROOT / "data/raw/london/school_holidays/uk_bank_holidays.csv", parse_dates=['date'])
study_months = pd.date_range('2023-02-01', '2026-01-01', freq='MS')
temporal_rows = []
for ms in study_months:
    y, m = ms.year, ms.month; dim = monthrange(y, m)[1]; me = pd.Timestamp(y, m, dim)
    hol_days = sum(1 for d in pd.date_range(ms, me) if any((d >= r['start_date']) & (d <= r['end_date']) for _, r in holidays.iterrows()))
    bh_days = len(bank_hols[(bank_hols['date'] >= ms) & (bank_hols['date'] <= me)])
    temporal_rows.append({'month': f"{y}-{m:02d}", 'pct_holiday': hol_days / dim, 'pct_bank_holiday': bh_days / dim})
temporal = pd.DataFrame(temporal_rows)
temporal_feats = ['pct_holiday', 'pct_bank_holiday']

samhi = pd.read_csv(PROJECT_ROOT / "data/raw/london/mental_health/samhi_lsoa.csv")[['lsoa11', 'samhi_index.2022', 'samhi_dec.2022']]
samhi.columns = ['lsoa_code', 'samhi_index', 'samhi_decile']
samhi_feats = ['samhi_index', 'samhi_decile']

edu_raw = pd.read_csv(CENSUS / "ts067/census2021-ts067-lsoa.csv")
edu_total = [c for c in edu_raw.columns if 'Total' in c][0]
edu = pd.DataFrame({'lsoa_code': edu_raw['geography code'], '_total': edu_raw[edu_total]})
for orig, short in [('No qualifications', 'pct_no_qual'), ('Level 4 qualifications and above', 'pct_level4_plus'), ('Apprenticeship', 'pct_apprentice')]:
    edu[short] = edu_raw[[c for c in edu_raw.columns if orig in c][0]] / edu['_total'] * 100
edu_feats = ['pct_no_qual', 'pct_level4_plus', 'pct_apprentice']

hh_raw = pd.read_csv(CENSUS / "ts003/census2021-ts003-lsoa.csv")
hh_total = [c for c in hh_raw.columns if 'Total' in c][0]
hh = pd.DataFrame({'lsoa_code': hh_raw['geography code'], '_total': hh_raw[hh_total]})
for pat, name in [('One person household; measures', 'pct_one_person'), ('Lone parent family: With dependent', 'pct_lone_parent_dep')]:
    hh[name] = hh_raw[[c for c in hh_raw.columns if pat in c][0]] / hh['_total'] * 100
hh_feats = ['pct_one_person', 'pct_lone_parent_dep']

all_context = imd_feats + weather_feats + demo_feats + poi_feats + housing_feats
all_supp = all_context + temporal_feats + samhi_feats + edu_feats + hh_feats

mc2 = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
active2 = mc2.groupby('lsoa_code')['crime_count'].sum(); active2 = active2[active2 >= 36].index
mc2 = mc2[mc2['lsoa_code'].isin(active2)]
all_months2 = sorted(mc2['month'].unique()); all_lsoas2 = sorted(mc2['lsoa_code'].unique())
gidx = pd.MultiIndex.from_product([all_lsoas2, all_months2], names=['lsoa_code', 'month'])
df2 = mc2.set_index(['lsoa_code', 'month']).reindex(gidx, fill_value=0).reset_index()
df2 = df2.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df2[f'lag_{lag}'] = df2.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df2[f'rm_{w}'] = df2.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df2['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df2['month']).dt.month / 12)
df2['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df2['month']).dt.month / 12)
tsx = pd.to_datetime(df2['month'])
df2['time_idx'] = (tsx.dt.year - tsx.dt.year.min()) * 12 + tsx.dt.month
base_feats2 = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'rm_3', 'rm_6', 'rm_12', 'month_sin', 'month_cos', 'time_idx']
for d in [imd, demo, poi, hprice, samhi, edu[['lsoa_code'] + edu_feats], hh[['lsoa_code'] + hh_feats]]:
    df2 = df2.merge(d, on='lsoa_code', how='left')
df2 = df2.merge(weather[['month'] + weather_feats], on='month', how='left')
df2 = df2.merge(temporal, on='month', how='left')
df2[poi_feats] = df2[poi_feats].fillna(0)
df2m = df2.dropna(subset=base_feats2 + imd_feats + demo_feats + housing_feats + samhi_feats).copy()
full_feats = [f for f in base_feats2 + all_supp if f in df2m.columns]

# Tune the FULL model (the "stronger model" an examiner imagines), then measure the lift.
resB, bestB, sumB = search(df2m, full_feats, all_months2, 'fusion-full-51feat')

# Fusion lift under a-priori vs under the tuned (full-selected) config, same panel.
rf_train, te = sumB['rf_train'], sumB['te']
ap_base_r2, _ = fit_eval(APRIORI, rf_train[base_feats2], rf_train['crime_count'], te[base_feats2], te['crime_count'])
bt_base_r2, _ = fit_eval(bestB, rf_train[base_feats2], rf_train['crime_count'], te[base_feats2], te['crime_count'])
ap_lift = sumB['ap_r2'] - ap_base_r2
bt_lift = sumB['bt_r2'] - bt_base_r2

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"""
PART A  crime-only baseline panel (5,148-LSOA, 11 feat)
  a-priori (200/15/5)  test R2 = {sumA['ap_r2']:.4f}   MAE = {sumA['ap_mae']:.3f}
  tuned    ({cfg_str(bestA)})
                       test R2 = {sumA['bt_r2']:.4f}   MAE = {sumA['bt_mae']:.3f}
  capacity gain        dR2 = {sumA['bt_r2'] - sumA['ap_r2']:+.4f}

PART B  full-fusion panel (4,078-LSOA, 51 feat)
  baseline feats  a-priori R2 = {ap_base_r2:.4f}   |  tuned R2 = {bt_base_r2:.4f}
  full feats      a-priori R2 = {sumB['ap_r2']:.4f}   |  tuned R2 = {sumB['bt_r2']:.4f}
  FUSION LIFT     a-priori dR2 = {ap_lift:+.4f}  |  tuned dR2 = {bt_lift:+.4f}
  tuned winner    ({cfg_str(bestB)})
""")

# Persist
outdf = pd.concat([resA, resB], ignore_index=True)
outdf.to_csv(OUT, index=False)
summary = pd.DataFrame([
    dict(panel='baseline-11feat', metric='a-priori test R2', value=round(sumA['ap_r2'], 4)),
    dict(panel='baseline-11feat', metric='tuned test R2', value=round(sumA['bt_r2'], 4)),
    dict(panel='baseline-11feat', metric='capacity gain dR2', value=round(sumA['bt_r2'] - sumA['ap_r2'], 4)),
    dict(panel='fusion-51feat', metric='a-priori fusion lift dR2', value=round(ap_lift, 4)),
    dict(panel='fusion-51feat', metric='tuned fusion lift dR2', value=round(bt_lift, 4)),
])
summary.to_csv(PROJECT_ROOT / "data/processed/london/rf_tuning_summary.csv", index=False)
print(f"Saved grid -> {OUT}")
print(f"Saved summary -> data/processed/london/rf_tuning_summary.csv")
db.close()
print("✅ RF hyperparameter tuning complete!")
