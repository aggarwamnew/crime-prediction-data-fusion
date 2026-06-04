"""25_benefits_fusion.py — DWP claimant count as a dynamic deprivation feature.

Strategy: Get London LSOA list from our DB, download all-England data with
recordlimit=50000, then filter locally. Runs fusion experiment.
"""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd, numpy as np, os, ssl, urllib.request, io
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB

DATA_DIR = str(Path(__file__).resolve().parents[2] / 'data')
CLAIMANT_FILE = os.path.join(DATA_DIR, 'claimant_count_lsoa.csv')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ── Get London LSOAs first ──
db = ThesisDB()
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
london_lsoas = set(crime['lsoa_code'].unique())
db.close()
print(f"London LSOAs: {len(london_lsoas):,}")

# ── Download claimant data ──
if not os.path.exists(CLAIMANT_FILE):
    print("Downloading claimant data from Nomis (paginated, 2 pages × 25K per month)...")
    all_dfs = []
    for y in range(2023, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 1: break
            month_dfs = []
            for offset in [0, 25000]:
                url = (f"https://www.nomisweb.co.uk/api/v01/dataset/NM_162_1.data.csv"
                       f"?geography=TYPE298&date={y}-{m:02d}"
                       f"&gender=0&age=0&measure=1&measures=20100"
                       f"&select=date_name,geography_code,obs_value"
                       f"&RecordOffset={offset}")
                try:
                    with urllib.request.urlopen(url, context=ctx) as resp:
                        chunk = pd.read_csv(io.BytesIO(resp.read()))
                        month_dfs.append(chunk)
                except Exception as e:
                    if offset == 0:
                        print(f"  {y}-{m:02d}: FAILED ({e})")
            if month_dfs:
                month_all = pd.concat(month_dfs, ignore_index=True)
                london_chunk = month_all[month_all['GEOGRAPHY_CODE'].isin(london_lsoas)]
                all_dfs.append(london_chunk)
                print(f"  {y}-{m:02d}: {len(month_all):,} total, {len(london_chunk):,} London")
    
    claimant = pd.concat(all_dfs, ignore_index=True)
    claimant.to_csv(CLAIMANT_FILE, index=False)
    print(f"  Saved {len(claimant):,} London rows")
else:
    print(f"Using cached {CLAIMANT_FILE}")
    claimant = pd.read_csv(CLAIMANT_FILE)

# ── Parse and prep ──
print("\n" + "="*60)
print("DWP CLAIMANT COUNT FUSION EXPERIMENT")
print("="*60)

month_map = {'January':'01','February':'02','March':'03','April':'04',
    'May':'05','June':'06','July':'07','August':'08',
    'September':'09','October':'10','November':'11','December':'12'}

def parse_nomis_date(d):
    parts = d.strip().split(' ')
    return f"{parts[1]}-{month_map[parts[0]]}"

claimant['month'] = claimant['DATE_NAME'].apply(parse_nomis_date)
claimant = claimant.rename(columns={'GEOGRAPHY_CODE':'lsoa_code','OBS_VALUE':'claimant_count'})
claimant = claimant[['lsoa_code','month','claimant_count']]

print(f"  Months covered: {sorted(claimant['month'].unique())}")
print(f"  LSOAs: {claimant['lsoa_code'].nunique():,}")
print(f"  Claimant range: {claimant['claimant_count'].min()}-{claimant['claimant_count'].max()}")
print(f"  Claimant mean: {claimant['claimant_count'].mean():.1f}")

# ── Build model ──
db = ThesisDB()
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
db.close()

lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]
all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code','month'])
df = crime.set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code','month']).reset_index(drop=True)

df = df.merge(claimant, on=['lsoa_code','month'], how='left')
matched = df['claimant_count'].notna().sum()
print(f"\n  Matched: {matched:,} / {len(df):,} ({100*matched/len(df):.1f}%)")
df['claimant_count'] = df['claimant_count'].fillna(df.groupby('lsoa_code')['claimant_count'].transform('median'))
df['claimant_count'] = df['claimant_count'].fillna(df['claimant_count'].median())

# Features
for lag in [1,2,3,6,12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3,6,12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2*np.pi*pd.to_datetime(df['month']).dt.month/12)
df['month_cos'] = np.cos(2*np.pi*pd.to_datetime(df['month']).dt.month/12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year - ts.dt.year.min())*12 + ts.dt.month

base = ['lag_1','lag_2','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_6','rolling_mean_12','month_sin','month_cos','time_idx']
fused = base + ['claimant_count']

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]
print(f"  Train: {len(train):,}  Test: {len(test):,}")

# ── Aggregate ──
rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base], train['crime_count'])
base_r2 = r2_score(test['crime_count'], rf_base.predict(test[base]))
base_mae = mean_absolute_error(test['crime_count'], rf_base.predict(test[base]))

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused], train['crime_count'])
fused_r2 = r2_score(test['crime_count'], rf_fused.predict(test[fused]))
fused_mae = mean_absolute_error(test['crime_count'], rf_fused.predict(test[fused]))

print(f"\n{'Model':<25s} | {'R2':>8s} | {'MAE':>8s}")
print(f"{'-'*25} | {'-'*8} | {'-'*8}")
print(f"{'Baseline':<25s} | {base_r2:8.4f} | {base_mae:8.2f}")
print(f"{'+ Claimant Count':<25s} | {fused_r2:8.4f} | {fused_mae:8.2f}")
print(f"\nΔ R²: {fused_r2 - base_r2:+.4f}")

# ── Per-type ──
print(f"\n{'='*60}")
print("PER-CRIME-TYPE RESULTS")
print(f"{'='*60}")
db2 = ThesisDB()
crime_detail = db2.query("SELECT lsoa_code, month, crime_type, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month, crime_type")
db2.close()

top_types = ['Drugs','Bicycle theft','Burglary','Criminal damage and arson','Public order',
    'Robbery','Shoplifting','Theft from the person','Vehicle crime',
    'Violence and sexual offences','Possession of weapons']

print(f"\n{'Crime Type':<32s} | {'Base R2':>8s} | {'Fused R2':>8s} | {'Δ R2':>8s}")
print(f"{'-'*32} | {'-'*8} | {'-'*8} | {'-'*8}")
for ct in top_types:
    ct_data = crime_detail[crime_detail['crime_type']==ct][['lsoa_code','month','crime_count']].copy()
    ct_lsoas = ct_data.groupby('lsoa_code')['crime_count'].sum()
    ct_active = ct_lsoas[ct_lsoas >= 12].index
    ct_data = ct_data[ct_data['lsoa_code'].isin(ct_active)]
    grid_ct = pd.MultiIndex.from_product([ct_active, all_months], names=['lsoa_code','month'])
    ct_df = ct_data.groupby(['lsoa_code','month'])['crime_count'].sum().reindex(grid_ct, fill_value=0).reset_index(name='crime_count')
    ct_df = ct_df.sort_values(['lsoa_code','month']).reset_index(drop=True)
    ct_df = ct_df.merge(claimant, on=['lsoa_code','month'], how='left')
    ct_df['claimant_count'] = ct_df['claimant_count'].fillna(ct_df.groupby('lsoa_code')['claimant_count'].transform('median'))
    ct_df['claimant_count'] = ct_df['claimant_count'].fillna(ct_df['claimant_count'].median())
    for lag in [1,2,3,6,12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    for w_val in [3,6,12]:
        ct_df[f'rolling_mean_{w_val}'] = ct_df.groupby('lsoa_code')['crime_count'].transform(
            lambda x: x.shift(1).rolling(w_val, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    ct_df['month_cos'] = np.cos(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    ts_ct = pd.to_datetime(ct_df['month'])
    ct_df['time_idx'] = (ts_ct.dt.year - ts_ct.dt.year.min())*12 + ts_ct.dt.month
    ct_model = ct_df.dropna()
    ct_train = ct_model[~ct_model['month'].isin(test_months)]
    ct_test = ct_model[ct_model['month'].isin(test_months)]
    if len(ct_test) < 10: continue
    rf_b = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_b.fit(ct_train[base], ct_train['crime_count'])
    b_r2 = r2_score(ct_test['crime_count'], rf_b.predict(ct_test[base]))
    rf_f = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(ct_train[fused], ct_train['crime_count'])
    f_r2 = r2_score(ct_test['crime_count'], rf_f.predict(ct_test[fused]))
    print(f"{ct:<32s} | {b_r2:8.4f} | {f_r2:8.4f} | {f_r2-b_r2:+8.4f}")

print(f"\n✅ DWP claimant count experiment complete!")
