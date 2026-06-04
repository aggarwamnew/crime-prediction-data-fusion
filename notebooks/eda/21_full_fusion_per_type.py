"""21_full_fusion_per_type.py — Per-crime-type full fusion (all 4 tiers)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd, numpy as np
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT
db = ThesisDB()
CENSUS = PROJECT_ROOT / "data/raw/london/census"

# ═══ LOAD ALL DATASETS (same as script 20) ═══
print("Loading datasets...")
imd = pd.read_csv(PROJECT_ROOT/"data/raw/london/imd/imd2025_file7.csv")
edu_col = [c for c in imd.columns if 'Education' in c and 'Score' in c][0]
imd_map = {'LSOA code (2021)':'lsoa_code','Index of Multiple Deprivation (IMD) Score':'imd_score',
    'Income Score (rate)':'income_score','Employment Score (rate)':'employment_score',
    edu_col:'imd_education_score','Health Deprivation and Disability Score':'health_score',
    'Crime Score':'crime_dep_score','Barriers to Housing and Services Score':'barriers_score',
    'Living Environment Score':'living_env_score'}
imd = imd[list(imd_map.keys())].rename(columns=imd_map)
imd_feats = list(imd_map.values())[1:]

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

ts006 = pd.read_csv(CENSUS/"ts006/census2021-ts006-lsoa.csv").rename(columns={'geography code':'lsoa_code','Population Density: Persons per square kilometre; measures: Value':'pop_density'})[['lsoa_code','pop_density']]
ts007 = pd.read_csv(CENSUS/"ts007a/census2021-ts007a-lsoa.csv").rename(columns={'geography code':'lsoa_code'})
ts007['pct_under15']=(ts007['Age: Aged 4 years and under']+ts007['Age: Aged 5 to 9 years']+ts007['Age: Aged 10 to 14 years'])/ts007['Age: Total']
ts007['pct_15_24']=(ts007['Age: Aged 15 to 19 years']+ts007['Age: Aged 20 to 24 years'])/ts007['Age: Total']
ts007['pct_65plus']=(ts007['Age: Aged 65 to 69 years']+ts007['Age: Aged 70 to 74 years']+ts007['Age: Aged 75 to 79 years']+ts007['Age: Aged 80 to 84 years']+ts007['Age: Aged 85 years and over'])/ts007['Age: Total']
demo = ts006.merge(ts007[['lsoa_code','pct_under15','pct_15_24','pct_65plus']], on='lsoa_code')
demo_feats = ['pop_density','pct_under15','pct_15_24','pct_65plus']

poi = pd.read_csv(PROJECT_ROOT/"data/raw/london/pois/poi_counts_per_lsoa.csv")
poi_feats = [c for c in poi.columns if c.startswith('poi_')]

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

samhi = pd.read_csv(PROJECT_ROOT/"data/raw/london/mental_health/samhi_lsoa.csv")[['lsoa11','samhi_index.2022','samhi_dec.2022']]
samhi.columns = ['lsoa_code','samhi_index','samhi_decile']
samhi_feats = ['samhi_index','samhi_decile']

edu_raw = pd.read_csv(CENSUS/"ts067/census2021-ts067-lsoa.csv")
edu_total = [c for c in edu_raw.columns if 'Total' in c][0]
edu = pd.DataFrame({'lsoa_code':edu_raw['geography code'],'_total':edu_raw[edu_total]})
for orig,short in [('No qualifications','pct_no_qual'),('Level 4 qualifications and above','pct_level4_plus'),('Apprenticeship','pct_apprentice')]:
    edu[short] = edu_raw[[c for c in edu_raw.columns if orig in c][0]] / edu['_total'] * 100
edu_feats = ['pct_no_qual','pct_level4_plus','pct_apprentice']

hh_raw = pd.read_csv(CENSUS/"ts003/census2021-ts003-lsoa.csv")
hh_total = [c for c in hh_raw.columns if 'Total' in c][0]
hh = pd.DataFrame({'lsoa_code':hh_raw['geography code'],'_total':hh_raw[hh_total]})
for pat,name in [('One person household; measures','pct_one_person'),('Lone parent family: With dependent','pct_lone_parent_dep')]:
    hh[name] = hh_raw[[c for c in hh_raw.columns if pat in c][0]] / hh['_total'] * 100
hh_feats = ['pct_one_person','pct_lone_parent_dep']

all_supp = imd_feats + weather_feats + demo_feats + poi_feats + housing_feats + temporal_feats + samhi_feats + edu_feats + hh_feats
# Spatial datasets for merge
spatial_dfs = [imd, demo, poi, hprice, samhi, edu[['lsoa_code']+edu_feats], hh[['lsoa_code']+hh_feats]]
print(f"Loaded {len(all_supp)} supplementary features")

# ═══ PER-CRIME-TYPE ═══
print("\n" + "="*90)
print("PER-CRIME-TYPE FULL FUSION: BASE vs CONTEXTUAL vs FULL")
print("="*90)
crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
results = []

for ct in crime_types:
    print(f"  {ct}...", end=" ", flush=True)
    ct_data = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct}' GROUP BY lsoa_code, month")
    ct_totals = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_totals[ct_totals >= 12].index
    if len(ct_active) < 50: print("SKIP"); continue

    ct_lsoas = sorted(ct_active)
    ct_months = sorted(ct_data['month'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code','month'])
    ct_df = ct_data.set_index(['lsoa_code','month']).reindex(ct_grid, fill_value=0).reset_index()
    ct_df = ct_df.sort_values(['lsoa_code','month']).reset_index(drop=True)

    for lag in [1,3,6,12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rm_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3,min_periods=1).mean())
    ct_df['rm_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12,min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    ct_df['month_cos'] = np.cos(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    base = ['lag_1','lag_3','lag_6','lag_12','rm_3','rm_12','month_sin','month_cos']

    for d in spatial_dfs:
        ct_df = ct_df.merge(d, on='lsoa_code', how='left')
    ct_df = ct_df.merge(weather[['month']+weather_feats], on='month', how='left')
    ct_df = ct_df.merge(temporal, on='month', how='left')
    ct_df[poi_feats] = ct_df[poi_feats].fillna(0)
    ct_model = ct_df.dropna(subset=base+imd_feats+demo_feats+housing_feats+samhi_feats)

    ct_test_months = ct_months[-6:]
    ct_train = ct_model[~ct_model['month'].isin(ct_test_months)]
    ct_test = ct_model[ct_model['month'].isin(ct_test_months)]
    if len(ct_test) < 50: print("SKIP"); continue

    ctx_feats = imd_feats + weather_feats + demo_feats + poi_feats + housing_feats
    ss_feats_all = samhi_feats + edu_feats + hh_feats

    configs = {'base': base, 'ctx': base+ctx_feats, 'ctx+temp': base+ctx_feats+temporal_feats, 'full': base+ctx_feats+temporal_feats+ss_feats_all}
    row = {'crime_type': ct}
    for vname, cols in configs.items():
        valid = [c for c in cols if c in ct_train.columns]
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(ct_train[valid], ct_train['crime_count'])
        row[f'r2_{vname}'] = r2_score(ct_test['crime_count'], rf.predict(ct_test[valid]))
    row['d_ctx'] = row['r2_ctx'] - row['r2_base']
    row['d_full'] = row['r2_full'] - row['r2_base']
    row['d_ss_marginal'] = row['r2_full'] - row['r2_ctx+temp']
    print(f"Δctx={row['d_ctx']:+.4f}  Δfull={row['d_full']:+.4f}  ΔSS_marginal={row['d_ss_marginal']:+.4f}")
    results.append(row)

print("\n" + "="*90)
print(f"{'Crime Type':35s} | {'R²base':>7s} | {'Δctx':>7s} | {'Δfull':>7s} | {'ΔSS+':>7s}")
print("-"*75)
res = pd.DataFrame(results).sort_values('d_full', ascending=False)
for _, r in res.iterrows():
    print(f"{r['crime_type']:35s} | {r['r2_base']:.4f}  | {r['d_ctx']:+.4f} | {r['d_full']:+.4f} | {r['d_ss_marginal']:+.4f}")

db.close()
print("\n✅ Per-crime-type full fusion complete!")
