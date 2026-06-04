"""20_full_fusion.py — All tiers combined: Baseline + Contextual + Temporal + Socio-Structural."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd, numpy as np
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT
db = ThesisDB()
CENSUS = PROJECT_ROOT / "data/raw/london/census"

# ═══ LOAD ALL DATASETS ═══
print("Loading all datasets...")

# CONTEXTUAL: IMD
imd = pd.read_csv(PROJECT_ROOT / "data/raw/london/imd/imd2025_file7.csv")
edu_col = [c for c in imd.columns if 'Education' in c and 'Score' in c][0]
imd_map = {'LSOA code (2021)':'lsoa_code','Index of Multiple Deprivation (IMD) Score':'imd_score',
    'Income Score (rate)':'income_score','Employment Score (rate)':'employment_score',
    edu_col:'imd_education_score','Health Deprivation and Disability Score':'health_score',
    'Crime Score':'crime_dep_score','Barriers to Housing and Services Score':'barriers_score',
    'Living Environment Score':'living_env_score'}
imd = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_feats = list(imd_map.values())[1:]

# CONTEXTUAL: Weather
records = []
for line in (PROJECT_ROOT/"data/raw/london/weather/heathrow_monthly.txt").read_text().strip().split('\n'):
    p = line.split()
    if len(p)>=7 and p[0].isdigit():
        try:
            tmax=float(p[2].replace('*',''));tmin=float(p[3].replace('*',''));rain=float(p[5].replace('*',''));sun=float(p[6].replace('*','').replace('#',''))
            records.append({'month':f"{int(p[0])}-{int(p[1]):02d}",'tmax':tmax,'tmin':tmin,'tmean':(tmax+tmin)/2,'rain_mm':rain,'sun_hours':sun})
        except: continue
weather = pd.DataFrame(records)
weather_feats = ['tmax','tmin','tmean','rain_mm','sun_hours']

# CONTEXTUAL: Demographics
ts006 = pd.read_csv(CENSUS/"ts006/census2021-ts006-lsoa.csv").rename(columns={'geography code':'lsoa_code','Population Density: Persons per square kilometre; measures: Value':'pop_density'})[['lsoa_code','pop_density']]
ts007 = pd.read_csv(CENSUS/"ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code':'lsoa_code'})
ts007['pct_under15']=(ts007['Age: Aged 4 years and under']+ts007['Age: Aged 5 to 9 years']+ts007['Age: Aged 10 to 14 years'])/ts007['Age: Total']
ts007['pct_15_24']=(ts007['Age: Aged 15 to 19 years']+ts007['Age: Aged 20 to 24 years'])/ts007['Age: Total']
ts007['pct_65plus']=(ts007['Age: Aged 65 to 69 years']+ts007['Age: Aged 70 to 74 years']+ts007['Age: Aged 75 to 79 years']+ts007['Age: Aged 80 to 84 years']+ts007['Age: Aged 85 years and over'])/ts007['Age: Total']
demo = ts006.merge(ts007[['lsoa_code','pct_under15','pct_15_24','pct_65plus']], on='lsoa_code')
demo_feats = ['pop_density','pct_under15','pct_15_24','pct_65plus']

# CONTEXTUAL: POIs
poi = pd.read_csv(PROJECT_ROOT/"data/raw/london/pois/poi_counts_per_lsoa.csv")
poi_feats = [c for c in poi.columns if c.startswith('poi_')]

# CONTEXTUAL: Housing
xlsx = PROJECT_ROOT/"data/raw/london/housing/ons_house_prices_lsoa.xlsx"
hprice = pd.read_excel(xlsx, skiprows=4, header=None, names=['la_code','la_name','lsoa_code','lsoa_name','median_house_price'])
hprice = hprice[hprice['la_code'].astype(str).str.startswith('E09')].copy()
hprice['median_house_price'] = pd.to_numeric(hprice['median_house_price'], errors='coerce')
hprice = hprice[['lsoa_code','median_house_price']].dropna()
# Create tertile bands
hp_thresh = hprice['median_house_price'].quantile([1/3, 2/3])
hp_low, hp_high = hp_thresh.iloc[0], hp_thresh.iloc[1]
hprice['price_low'] = (hprice['median_house_price'] < hp_low).astype(int)
hprice['price_mid'] = ((hprice['median_house_price'] >= hp_low) & (hprice['median_house_price'] < hp_high)).astype(int)
hprice['price_high'] = (hprice['median_house_price'] >= hp_high).astype(int)
housing_feats = ['price_low','price_mid','price_high']

# TEMPORAL: School holidays + bank holidays
holidays = pd.read_csv(PROJECT_ROOT/"data/raw/london/school_holidays/london_school_holidays.csv", parse_dates=['start_date','end_date'])
bank_hols = pd.read_csv(PROJECT_ROOT/"data/raw/london/school_holidays/uk_bank_holidays.csv", parse_dates=['date'])
study_months = pd.date_range('2023-02-01','2026-01-01',freq='MS')
temporal_rows = []
for ms in study_months:
    y,m = ms.year,ms.month; dim=monthrange(y,m)[1]; me=pd.Timestamp(y,m,dim)
    hol_days=sum(1 for d in pd.date_range(ms,me) if any((d>=r['start_date'])&(d<=r['end_date']) for _,r in holidays.iterrows()))
    bh_days=len(bank_hols[(bank_hols['date']>=ms)&(bank_hols['date']<=me)])
    temporal_rows.append({'month':f"{y}-{m:02d}",'pct_holiday':hol_days/dim,'pct_bank_holiday':bh_days/dim})
temporal = pd.DataFrame(temporal_rows)
temporal_feats = ['pct_holiday','pct_bank_holiday']

# SOCIO-STRUCTURAL: SAMHI
samhi = pd.read_csv(PROJECT_ROOT/"data/raw/london/mental_health/samhi_lsoa.csv")[['lsoa11','samhi_index.2022','samhi_dec.2022']]
samhi.columns = ['lsoa_code','samhi_index','samhi_decile']
samhi_feats = ['samhi_index','samhi_decile']

# SOCIO-STRUCTURAL: Education
edu_raw = pd.read_csv(CENSUS/"ts067/census2021-ts067-lsoa.csv")
edu_total = [c for c in edu_raw.columns if 'Total' in c][0]
edu = pd.DataFrame({'lsoa_code':edu_raw['geography code'],'_total':edu_raw[edu_total]})
for orig,short in [('No qualifications','pct_no_qual'),('Level 4 qualifications and above','pct_level4_plus'),('Apprenticeship','pct_apprentice')]:
    edu[short] = edu_raw[[c for c in edu_raw.columns if orig in c][0]] / edu['_total'] * 100
edu_feats = ['pct_no_qual','pct_level4_plus','pct_apprentice']

# SOCIO-STRUCTURAL: Household
hh_raw = pd.read_csv(CENSUS/"ts003/census2021-ts003-lsoa.csv")
hh_total = [c for c in hh_raw.columns if 'Total' in c][0]
hh = pd.DataFrame({'lsoa_code':hh_raw['geography code'],'_total':hh_raw[hh_total]})
for pat,name in [('One person household; measures','pct_one_person'),('Lone parent family: With dependent','pct_lone_parent_dep')]:
    hh[name] = hh_raw[[c for c in hh_raw.columns if pat in c][0]] / hh['_total'] * 100
hh_feats = ['pct_one_person','pct_lone_parent_dep']

all_context = imd_feats + weather_feats + demo_feats + poi_feats + housing_feats
all_temporal = temporal_feats
all_ss = samhi_feats + edu_feats + hh_feats
all_supp = all_context + all_temporal + all_ss
print(f"Features: {len(all_context)} contextual + {len(all_temporal)} temporal + {len(all_ss)} socio-structural = {len(all_supp)} total supplementary")

# ═══ BUILD DATASET ═══
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
active = crime.groupby('lsoa_code')['crime_count'].sum()
active = active[active>=36].index
crime = crime[crime['lsoa_code'].isin(active)]
all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code','month'])
df = crime.set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code','month']).reset_index(drop=True)

for lag in [1,2,3,6,12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3,6,12]:
    df[f'rm_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(w,min_periods=1).mean())
df['month_sin'] = np.sin(2*np.pi*pd.to_datetime(df['month']).dt.month/12)
df['month_cos'] = np.cos(2*np.pi*pd.to_datetime(df['month']).dt.month/12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year-ts.dt.year.min())*12+ts.dt.month
base_feats = ['lag_1','lag_2','lag_3','lag_6','lag_12','rm_3','rm_6','rm_12','month_sin','month_cos','time_idx']

# Merge all
for d in [imd, demo, poi, hprice, samhi, edu[['lsoa_code']+edu_feats], hh[['lsoa_code']+hh_feats]]:
    df = df.merge(d, on='lsoa_code', how='left')
df = df.merge(weather[['month']+weather_feats], on='month', how='left')
df = df.merge(temporal, on='month', how='left')
df[poi_feats] = df[poi_feats].fillna(0)
df_model = df.dropna(subset=base_feats+imd_feats+demo_feats+housing_feats+samhi_feats)

test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]
print(f"LSOAs: {df_model['lsoa_code'].nunique():,}, Train: {len(train):,}, Test: {len(test):,}")

# ═══ TIER COMPARISON ═══
print("\n" + "="*70)
print("FULL FUSION — TIER COMPARISON")
print("="*70)
configs = [
    ('Baseline (crime only)', base_feats),
    ('+ Contextual', base_feats + all_context),
    ('+ Temporal', base_feats + all_temporal),
    ('+ Socio-Structural', base_feats + all_ss),
    ('+ Context + Temporal', base_feats + all_context + all_temporal),
    ('+ Context + SS', base_feats + all_context + all_ss),
    ('FULL (all tiers)', base_feats + all_supp),
]
print(f"\n{'Config':35s} | {'R²':>7s} | {'Δ R²':>8s} | {'MAE':>6s} | {'Feats':>5s}")
print("-"*75)
r2_base = None
for name, feats in configs:
    valid_feats = [f for f in feats if f in train.columns]
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(train[valid_feats], train['crime_count'])
    r2 = r2_score(test['crime_count'], rf.predict(test[valid_feats]))
    mae = mean_absolute_error(test['crime_count'], rf.predict(test[valid_feats]))
    if r2_base is None: r2_base = r2
    print(f"{name:35s} | {r2:.4f}  | {r2-r2_base:+.4f}  | {mae:.2f}  | {len(valid_feats):>3d}")

# Top features in full model
rf_full = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
full_feats = [f for f in base_feats+all_supp if f in train.columns]
rf_full.fit(train[full_feats], train['crime_count'])
imp = pd.Series(rf_full.feature_importances_, index=full_feats).sort_values(ascending=False)
print(f"\nTop 15 features (full model):")
for i,(f,v) in enumerate(imp.head(15).items()):
    tier = 'BASE' if f in base_feats else 'CTX' if f in all_context else 'TEMP' if f in all_temporal else 'SS'
    print(f"  {i+1:2d}. [{tier:4s}] {f:25s}: {v:.4f}")

db.close()
print("\n✅ Full fusion experiment complete!")
