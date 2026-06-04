"""24_daylight_fusion.py — Daylight hours as a dynamic environmental feature."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd, numpy as np
from datetime import datetime
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.db import ThesisDB

# ── Daylight calculation (astronomical) ──
LONDON_LAT = 51.5074

def daylight_hours(date):
    """Compute daylight hours for a given date at London's latitude."""
    doy = date.timetuple().tm_yday
    # Solar declination (Spencer, 1971)
    B = (360/365) * (doy - 81)
    decl = np.radians(23.45 * np.sin(np.radians(B)))
    lat = np.radians(LONDON_LAT)
    # Hour angle at sunrise/sunset
    cos_ha = -np.tan(lat) * np.tan(decl)
    cos_ha = np.clip(cos_ha, -1, 1)  # Handle polar edge cases
    ha = np.degrees(np.arccos(cos_ha))
    return 2 * ha / 15  # Convert to hours

def monthly_daylight(year, month):
    """Average daylight hours for a month."""
    dim = monthrange(year, month)[1]
    return np.mean([daylight_hours(datetime(year, month, d)) for d in range(1, dim+1)])

# ── Load data ──
print("="*60)
print("DAYLIGHT HOURS FUSION EXPERIMENT")
print("="*60)
db = ThesisDB()
crime = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]
all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code','month'])
df = crime.set_index(['lsoa_code','month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code','month']).reset_index(drop=True)
db.close()

# ── Daylight feature ──
daylight_rows = []
for m in all_months:
    dt = pd.Timestamp(m)
    dl = monthly_daylight(dt.year, dt.month)
    daylight_rows.append({'month': m, 'daylight_hours': dl})
    print(f"  {m}: {dl:.1f}h daylight")
daylight = pd.DataFrame(daylight_rows)
df = df.merge(daylight, on='month')

# ── Base features ──
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
fused = base + ['daylight_hours']

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

print(f"\nTrain: {len(train):,}  Test: {len(test):,}")

# ── Baseline ──
rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_base.fit(train[base], train['crime_count'])
base_r2 = r2_score(test['crime_count'], rf_base.predict(test[base]))
base_mae = mean_absolute_error(test['crime_count'], rf_base.predict(test[base]))

# ── Fused ──
rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train[fused], train['crime_count'])
fused_r2 = r2_score(test['crime_count'], rf_fused.predict(test[fused]))
fused_mae = mean_absolute_error(test['crime_count'], rf_fused.predict(test[fused]))

delta = fused_r2 - base_r2
print(f"\n{'Model':<20s} | {'R2':>8s} | {'MAE':>8s}")
print(f"{'-'*20} | {'-'*8} | {'-'*8}")
print(f"{'Baseline':<20s} | {base_r2:8.4f} | {base_mae:8.2f}")
print(f"{'+ Daylight Hours':<20s} | {fused_r2:8.4f} | {fused_mae:8.2f}")
print(f"\nΔ R²: {delta:+.4f}")

# ── Per-type ──
print(f"\n{'='*60}")
print("PER-CRIME-TYPE RESULTS")
print(f"{'='*60}")
crime_types_df = db if False else None
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
    for lag in [1,2,3,6,12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    for w_val in [3,6,12]:
        ct_df[f'rolling_mean_{w_val}'] = ct_df.groupby('lsoa_code')['crime_count'].transform(
            lambda x: x.shift(1).rolling(w_val, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    ct_df['month_cos'] = np.cos(2*np.pi*pd.to_datetime(ct_df['month']).dt.month/12)
    ts_ct = pd.to_datetime(ct_df['month'])
    ct_df['time_idx'] = (ts_ct.dt.year - ts_ct.dt.year.min())*12 + ts_ct.dt.month
    ct_df = ct_df.merge(daylight, on='month')
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
    d = f_r2 - b_r2
    print(f"{ct:<32s} | {b_r2:8.4f} | {f_r2:8.4f} | {d:+8.4f}")

print(f"\n✅ Daylight hours experiment complete!")
