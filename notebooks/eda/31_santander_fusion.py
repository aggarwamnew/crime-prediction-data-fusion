"""
Script 31: Santander Cycles Processing and Fusion
1. Parses station locations from XML (lat/lon)
2. Loads trip CSVs, aggregates to monthly departures per station
3. Spatial joins stations to LSOAs
4. Aggregates to monthly bike hire volume per LSOA
5. Runs ablation experiment
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import geopandas as gpd
import xml.etree.ElementTree as ET
from pathlib import Path
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

SANTANDER_DIR = PROJECT_ROOT / "data/raw/london/transport/santander_cycles"
TRIPS_DIR = SANTANDER_DIR / "trips"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/london"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXPERIMENT: SANTANDER CYCLES FUSION (Dynamic Transport)")
print("=" * 70)

# ============================================================
# STEP 1: Parse station locations from XML
# ============================================================
print("\n1. Parsing station locations...")
tree = ET.parse(SANTANDER_DIR / "stations.xml")
root = tree.getroot()
stations = []
for s in root.findall('.//station'):
    sid = s.find('id')
    name = s.find('name')
    lat = s.find('lat')
    lon = s.find('long')
    docks = s.find('nbDocks')
    term = s.find('terminalName')
    if sid is not None and lat is not None and lon is not None:
        stations.append({
            'station_id': int(sid.text),
            'station_name': name.text if name is not None else '',
            'terminal': term.text if term is not None else '',
            'lat': float(lat.text),
            'lon': float(lon.text),
            'docks': int(docks.text) if docks is not None else 0
        })
station_df = pd.DataFrame(stations)
print(f"   Stations parsed: {len(station_df):,}")
print(f"   Lat range: {station_df['lat'].min():.4f} to {station_df['lat'].max():.4f}")
print(f"   Lon range: {station_df['lon'].min():.4f} to {station_df['lon'].max():.4f}")

# ============================================================
# STEP 2: Load and aggregate trip data
# ============================================================
print("\n2. Loading trip data (this may take a moment)...")
trip_files = sorted(TRIPS_DIR.glob("*.csv"))
print(f"   Files to process: {len(trip_files)}")

monthly_departures = []
monthly_arrivals = []

for i, f in enumerate(trip_files):
    if (i+1) % 10 == 0 or i == 0:
        print(f"   Processing file {i+1}/{len(trip_files)}: {f.name}...")

    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"   ERROR reading {f.name}: {e}")
        continue

    # Column names vary slightly between files
    # Common: "Start date", "Start station number", "End station number"
    start_date_col = [c for c in df.columns if 'start' in c.lower() and 'date' in c.lower()]
    start_station_col = [c for c in df.columns if 'start' in c.lower() and 'station' in c.lower() and 'number' in c.lower()]
    end_station_col = [c for c in df.columns if 'end' in c.lower() and 'station' in c.lower() and 'number' in c.lower()]

    if not start_date_col or not start_station_col:
        print(f"   WARNING: Could not find required columns in {f.name}")
        print(f"   Available columns: {list(df.columns)}")
        continue

    start_col = start_date_col[0]
    start_stn = start_station_col[0]
    end_stn = end_station_col[0] if end_station_col else None

    # Parse month from start date
    df['start_dt'] = pd.to_datetime(df[start_col], format='mixed', errors='coerce')
    df = df.dropna(subset=['start_dt'])
    df['month'] = df['start_dt'].dt.strftime('%Y-%m')

    # Filter to thesis period
    df = df[(df['month'] >= '2023-01') & (df['month'] <= '2026-01')]
    if len(df) == 0:
        continue

    # Departures per station per month
    dep = df.groupby([start_stn, 'month']).size().reset_index(name='departures')
    dep = dep.rename(columns={start_stn: 'station_number'})
    monthly_departures.append(dep)

    # Arrivals per station per month
    if end_stn:
        arr = df.groupby([end_stn, 'month']).size().reset_index(name='arrivals')
        arr = arr.rename(columns={end_stn: 'station_number'})
        monthly_arrivals.append(arr)

# Concatenate and aggregate
departures = pd.concat(monthly_departures, ignore_index=True)
departures['station_number'] = departures['station_number'].astype(str)
departures = departures.groupby(['station_number', 'month'])['departures'].sum().reset_index()

arrivals = pd.concat(monthly_arrivals, ignore_index=True)
arrivals['station_number'] = arrivals['station_number'].astype(str)
arrivals = arrivals.groupby(['station_number', 'month'])['arrivals'].sum().reset_index()

# Merge departures and arrivals
bike_monthly = departures.merge(arrivals, on=['station_number', 'month'], how='outer').fillna(0)
bike_monthly['total_trips'] = bike_monthly['departures'] + bike_monthly['arrivals']

print(f"\n   Monthly station-level data: {len(bike_monthly):,} rows")
print(f"   Unique stations: {bike_monthly['station_number'].nunique()}")
print(f"   Months: {bike_monthly['month'].nunique()}")

# ============================================================
# STEP 3: Spatial join stations to LSOAs
# ============================================================
print("\n3. Spatial joining stations to LSOAs...")

# Match station numbers to coordinates
# The trip data uses "Start station number" which maps to "terminalName" in the XML
station_df['terminal'] = station_df['terminal'].astype(str)
bike_monthly_coords = bike_monthly.merge(
    station_df[['terminal', 'lat', 'lon']],
    left_on='station_number', right_on='terminal', how='inner'
)
print(f"   Stations with coordinates: {bike_monthly_coords['station_number'].nunique()}")

# Also try matching on station_id
if bike_monthly_coords['station_number'].nunique() < bike_monthly['station_number'].nunique() * 0.5:
    print("   Low match via terminal. Trying station_id match...")
    station_df['station_id_str'] = station_df['station_id'].astype(str)
    alt_match = bike_monthly.merge(
        station_df[['station_id_str', 'lat', 'lon']],
        left_on='station_number', right_on='station_id_str', how='inner'
    )
    if alt_match['station_number'].nunique() > bike_monthly_coords['station_number'].nunique():
        bike_monthly_coords = alt_match
        print(f"   Better match via station_id: {bike_monthly_coords['station_number'].nunique()}")

# Load LSOA boundaries
lsoa_path = PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson"
lsoa_gdf = gpd.read_file(lsoa_path)

# Get unique station locations
station_locs = bike_monthly_coords[['station_number', 'lat', 'lon']].drop_duplicates('station_number')
geometry = [Point(row['lon'], row['lat']) for _, row in station_locs.iterrows()]
station_gdf = gpd.GeoDataFrame(station_locs, geometry=geometry, crs="EPSG:4326")
station_gdf = station_gdf.to_crs(lsoa_gdf.crs)

# Spatial join
joined = gpd.sjoin(station_gdf, lsoa_gdf, how='left', predicate='within')
lsoa_col = [c for c in joined.columns if 'LSOA' in c.upper() and 'CD' in c.upper()][0]
station_lsoa = joined[['station_number', lsoa_col]].rename(columns={lsoa_col: 'lsoa_code'}).dropna()
print(f"   Stations mapped to LSOAs: {len(station_lsoa):,}")
print(f"   Unique LSOAs with bike stations: {station_lsoa['lsoa_code'].nunique()}")

# ============================================================
# STEP 4: Aggregate to LSOA-level monthly bike hire
# ============================================================
print("\n4. Aggregating to LSOA-level...")
bike_lsoa = bike_monthly_coords.merge(station_lsoa, on='station_number', how='inner')
lsoa_bikes = bike_lsoa.groupby(['lsoa_code', 'month']).agg(
    bike_departures=('departures', 'sum'),
    bike_arrivals=('arrivals', 'sum'),
    bike_total=('total_trips', 'sum'),
    bike_stations=('station_number', 'nunique')
).reset_index()
lsoa_bikes['bike_per_station'] = lsoa_bikes['bike_total'] / lsoa_bikes['bike_stations']

print(f"   LSOA-month bike rows: {len(lsoa_bikes):,}")
print(f"   LSOAs with bike data: {lsoa_bikes['lsoa_code'].nunique()}")
print(f"   Mean monthly trips per LSOA: {lsoa_bikes['bike_total'].mean():,.0f}")

# Save
lsoa_bikes.to_csv(PROCESSED_DIR / "santander_monthly.csv", index=False)
print(f"   Saved to {PROCESSED_DIR / 'santander_monthly.csv'}")

# ============================================================
# STEP 5: Fusion experiment
# ============================================================
print(f"\n{'='*70}")
print("AGGREGATE FUSION EXPERIMENT")
print(f"{'='*70}")

db = ThesisDB()
monthly_crime = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")

bike_features = ['bike_total', 'bike_stations', 'bike_per_station']

merged = monthly_crime.merge(lsoa_bikes[['lsoa_code', 'month'] + bike_features],
                              on=['lsoa_code', 'month'], how='left')
merged[bike_features] = merged[bike_features].fillna(0)

MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
n_lsoas = merged['lsoa_code'].nunique()
print(f"\n   Active LSOAs: {n_lsoas:,}")
print(f"   LSOAs with bike data: {merged[merged['bike_total'] > 0]['lsoa_code'].nunique()}")
print(f"   LSOAs without bikes: {merged[merged['bike_total'] == 0]['lsoa_code'].nunique()}")

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

bike_lookup = merged[['lsoa_code', 'month'] + bike_features].drop_duplicates(['lsoa_code', 'month']).set_index(['lsoa_code', 'month'])
crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()
df = df.join(bike_lookup, on=['lsoa_code', 'month'], how='left')
df[bike_features] = df[bike_features].fillna(0)

df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
month_idx = {m: i for i, m in enumerate(all_months)}
df['time_idx'] = df['month'].map(month_idx)

df_model = df.dropna().copy()
test_months = all_months[-6:]
train_df = df_model[~df_model['month'].isin(test_months)]
test_df = df_model[df_model['month'].isin(test_months)]

lag_features_list = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
temporal_features = ['month_sin', 'month_cos', 'time_idx']
crime_only_cols = lag_features_list + temporal_features
fused_cols = lag_features_list + temporal_features + bike_features

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")

rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])

r2_base = r2_score(test_df['crime_count'], rf_crime.predict(test_df[crime_only_cols]))
r2_fused = r2_score(test_df['crime_count'], rf_fused.predict(test_df[fused_cols]))
delta_r2 = r2_fused - r2_base

print(f"\n   Crime-only: R²={r2_base:.4f}")
print(f"   Fused (+Bikes): R²={r2_fused:.4f}")
print(f"   Δ R² = {delta_r2:+.4f}")

# Feature importance
print("\n   Feature importance (bike features):")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
for _, row in imp[imp['feature'].isin(bike_features)].iterrows():
    print(f"     {row['feature']:30s} {row['importance']:.4f}")

# Per-crime-type (abbreviated)
print(f"\n{'='*70}")
print("PER-CRIME-TYPE ANALYSIS")
print(f"{'='*70}")

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
per_type_results = []

for ct in crime_types:
    print(f"\n  {ct}...", end=" ", flush=True)
    ct_monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")
    ct_merged = ct_monthly.merge(lsoa_bikes[['lsoa_code', 'month'] + bike_features],
                                  on=['lsoa_code', 'month'], how='left')
    ct_merged[bike_features] = ct_merged[bike_features].fillna(0)

    lsoa_totals = ct_merged.groupby('lsoa_code')['crime_count'].sum()
    active_ct = lsoa_totals[lsoa_totals >= 12].index
    if len(active_ct) < 50: print("SKIP"); continue
    ct_merged = ct_merged[ct_merged['lsoa_code'].isin(active_ct)]

    ct_months = sorted(ct_merged['month'].unique())
    ct_lsoas = sorted(ct_merged['lsoa_code'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])
    ct_df = ct_merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(ct_grid, fill_value=0).reset_index()
    bike_lk = ct_merged[['lsoa_code','month'] + bike_features].drop_duplicates(['lsoa_code','month']).set_index(['lsoa_code','month'])
    ct_df = ct_df.join(bike_lk, on=['lsoa_code','month'], how='left')
    ct_df[bike_features] = ct_df[bike_features].fillna(0)
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rolling_mean_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rolling_mean_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_model = ct_df.dropna()
    if len(ct_model) < 100: print("SKIP"); continue
    test_mos = ct_months[-6:]
    train = ct_model[~ct_model['month'].isin(test_mos)]
    test = ct_model[ct_model['month'].isin(test_mos)]
    if len(test) < 50: print("SKIP"); continue

    lag_cols = ['lag_1','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_12','month_sin','month_cos']
    rf_b = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_b.fit(train[lag_cols], train['crime_count'])
    r2_b = r2_score(test['crime_count'], rf_b.predict(test[lag_cols]))

    rf_f = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_f.fit(train[lag_cols + bike_features], train['crime_count'])
    r2_f = r2_score(test['crime_count'], rf_f.predict(test[lag_cols + bike_features]))

    delta = r2_f - r2_b
    print(f"Δ={delta:+.4f}")
    per_type_results.append({'crime_type': ct, 'r2_base': r2_b, 'delta': delta, 'n_lsoas': len(active_ct)})

db.close()

res = pd.DataFrame(per_type_results).sort_values('delta', ascending=False)
print(f"\n{'='*90}")
print("SUMMARY: SANTANDER CYCLES IMPACT BY CRIME TYPE")
print(f"{'='*90}")
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ Bike':>8}")
print("-" * 55)
for _, r in res.iterrows():
    print(f"{r['crime_type']:<35} {r['r2_base']:>8.4f} {r['delta']:>+8.4f}")

print(f"\n   Aggregate Δ R² = {delta_r2:+.4f}")
print("\n✅ Santander Cycles fusion experiment complete!")
