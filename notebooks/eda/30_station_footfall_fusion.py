"""
Script 30: Station Footfall Processing and Fusion
1. Loads daily station tap data (entries + exits)
2. Aggregates to monthly per station
3. Gets station coordinates from TfL API
4. Spatial joins stations to LSOAs
5. Aggregates to monthly ridership per LSOA
6. Runs ablation experiment (aggregate + per-crime-type)
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import json
from pathlib import Path
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

TRANSPORT_DIR = PROJECT_ROOT / "data/raw/london/transport/station_taps"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/london"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXPERIMENT: STATION FOOTFALL FUSION (Dynamic Transport)")
print("=" * 70)

# ============================================================
# STEP 1: Load and aggregate station footfall data
# ============================================================
print("\n1. Loading station footfall data...")
dfs = []
for f in sorted(TRANSPORT_DIR.glob("StationFootfall_*.csv")):
    print(f"   Loading {f.name}...")
    df = pd.read_csv(f)
    dfs.append(df)
footfall = pd.concat(dfs, ignore_index=True)
print(f"   Total rows: {len(footfall):,}")
print(f"   Columns: {list(footfall.columns)}")
print(f"   Date range: {footfall['TravelDate'].min()} to {footfall['TravelDate'].max()}")
print(f"   Unique stations: {footfall['Station'].nunique()}")

# Parse date and create month column
footfall['date'] = pd.to_datetime(footfall['TravelDate'], format='%Y%m%d')
footfall['month'] = footfall['date'].dt.to_period('M').astype(str)
# Convert period format "2023-01" to match crime data format "2023-01"
footfall['month'] = footfall['date'].dt.strftime('%Y-%m')

# Filter to thesis period (Jan 2023 to Jan 2026)
footfall = footfall[(footfall['month'] >= '2023-01') & (footfall['month'] <= '2026-01')]
print(f"   After filtering to thesis period: {len(footfall):,} rows")
print(f"   Months: {footfall['month'].nunique()}")

# Aggregate to monthly per station
monthly_station = footfall.groupby(['Station', 'month']).agg(
    entries=('EntryTapCount', 'sum'),
    exits=('ExitTapCount', 'sum')
).reset_index()
monthly_station['total_taps'] = monthly_station['entries'] + monthly_station['exits']
print(f"\n   Monthly station data: {len(monthly_station):,} rows ({monthly_station['Station'].nunique()} stations x {monthly_station['month'].nunique()} months)")

# ============================================================
# STEP 2: Get station coordinates
# ============================================================
print("\n2. Getting station coordinates...")
stations_file = TRANSPORT_DIR / "station_coordinates.csv"

if stations_file.exists():
    print("   Loading cached station coordinates...")
    station_coords = pd.read_csv(stations_file)
else:
    print("   Fetching from TfL API (this may take a moment)...")
    unique_stations = footfall['Station'].unique()
    coords_list = []

    # Try TfL StopPoint API for each mode
    modes = ['tube', 'overground', 'dlr', 'elizabeth-line', 'national-rail']
    all_stops = []
    for mode in modes:
        print(f"     Fetching {mode} stations...", end=" ", flush=True)
        try:
            url = f"https://api.tfl.gov.uk/StopPoint/Mode/{mode}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            stops = data.get('stopPoints', data) if isinstance(data, dict) else data
            for s in stops:
                all_stops.append({
                    'api_name': s.get('commonName', ''),
                    'lat': s.get('lat'),
                    'lon': s.get('lon'),
                    'mode': mode,
                    'naptan': s.get('naptanId', '')
                })
            print(f"{len(stops)} found")
        except Exception as e:
            print(f"ERROR: {e}")

    api_df = pd.DataFrame(all_stops).drop_duplicates(subset=['api_name'])
    print(f"   Total unique API stations: {len(api_df):,}")

    # Match footfall station names to API names
    # Footfall uses names like "Abbey Road DLR", "Baker Street"
    # API uses "Abbey Road DLR Station", "Baker Street Underground Station"
    # Strategy: normalize both, then fuzzy match
    def normalize(name):
        return (name.lower()
                .replace(' underground station', '')
                .replace(' rail station', '')
                .replace(' station', '')
                .replace(' dlr', '')
                .strip())

    api_df['norm'] = api_df['api_name'].apply(normalize)
    footfall_stations = pd.DataFrame({'Station': unique_stations})
    footfall_stations['norm'] = footfall_stations['Station'].apply(normalize)

    matched = footfall_stations.merge(api_df[['norm', 'lat', 'lon']].drop_duplicates('norm'),
                                       on='norm', how='left')
    match_rate = matched['lat'].notna().mean()
    print(f"   Match rate: {match_rate:.1%} ({matched['lat'].notna().sum()}/{len(matched)})")

    # For unmatched, try partial matching
    unmatched = matched[matched['lat'].isna()]['Station'].tolist()
    if unmatched:
        print(f"   Trying partial match for {len(unmatched)} unmatched stations...")
        for station in unmatched:
            norm_s = normalize(station)
            # Find closest match in API names
            candidates = api_df[api_df['norm'].str.contains(norm_s[:10], na=False)]
            if len(candidates) == 0:
                candidates = api_df[api_df['norm'].apply(lambda x: norm_s[:8] in x)]
            if len(candidates) > 0:
                best = candidates.iloc[0]
                matched.loc[matched['Station'] == station, ['lat', 'lon']] = best['lat'], best['lon']

    final_match = matched['lat'].notna().mean()
    print(f"   Final match rate: {final_match:.1%}")

    station_coords = matched[['Station', 'lat', 'lon']].dropna()
    station_coords.to_csv(stations_file, index=False)
    print(f"   Saved to {stations_file}")

print(f"   Stations with coordinates: {len(station_coords):,}")

# ============================================================
# STEP 3: Spatial join stations to LSOAs
# ============================================================
print("\n3. Spatial joining stations to LSOAs...")
lsoa_path = PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson"
lsoa_gdf = gpd.read_file(lsoa_path)
print(f"   LSOA boundaries loaded: {len(lsoa_gdf):,}")

# Create station GeoDataFrame
geometry = [Point(row['lon'], row['lat']) for _, row in station_coords.iterrows()]
station_gdf = gpd.GeoDataFrame(station_coords, geometry=geometry, crs="EPSG:4326")
station_gdf = station_gdf.to_crs(lsoa_gdf.crs)

# Spatial join
joined = gpd.sjoin(station_gdf, lsoa_gdf, how='left', predicate='within')
lsoa_col = [c for c in joined.columns if 'LSOA' in c.upper() and 'CD' in c.upper()][0]
station_lsoa = joined[['Station', lsoa_col]].rename(columns={lsoa_col: 'lsoa_code'}).dropna()
print(f"   Stations mapped to LSOAs: {len(station_lsoa):,}")
print(f"   Unique LSOAs with stations: {station_lsoa['lsoa_code'].nunique()}")

# ============================================================
# STEP 4: Aggregate to LSOA-level monthly ridership
# ============================================================
print("\n4. Aggregating to LSOA-level monthly ridership...")
ridership = monthly_station.merge(station_lsoa, on='Station', how='inner')
lsoa_ridership = ridership.groupby(['lsoa_code', 'month']).agg(
    ridership_total=('total_taps', 'sum'),
    station_count=('Station', 'nunique'),
    ridership_entries=('entries', 'sum'),
    ridership_exits=('exits', 'sum')
).reset_index()
lsoa_ridership['ridership_per_station'] = lsoa_ridership['ridership_total'] / lsoa_ridership['station_count']

print(f"   LSOA-month ridership rows: {len(lsoa_ridership):,}")
print(f"   LSOAs with ridership: {lsoa_ridership['lsoa_code'].nunique()}")
print(f"   Mean monthly ridership per LSOA: {lsoa_ridership['ridership_total'].mean():,.0f}")

# Save processed ridership
lsoa_ridership.to_csv(PROCESSED_DIR / "station_ridership_monthly.csv", index=False)
print(f"   Saved to {PROCESSED_DIR / 'station_ridership_monthly.csv'}")

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

ridership_features = ['ridership_total', 'station_count', 'ridership_per_station']

# Merge: left join so LSOAs without stations get 0
merged = monthly_crime.merge(lsoa_ridership[['lsoa_code', 'month'] + ridership_features],
                              on=['lsoa_code', 'month'], how='left')
merged[ridership_features] = merged[ridership_features].fillna(0)

MIN_CRIMES = 36
lsoa_totals = merged.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
merged = merged[merged['lsoa_code'].isin(active)]
n_lsoas = merged['lsoa_code'].nunique()
print(f"\n   Active LSOAs: {n_lsoas:,}")
print(f"   LSOAs with station data: {merged[merged['ridership_total'] > 0]['lsoa_code'].nunique()}")
print(f"   LSOAs without stations: {merged[merged['ridership_total'] == 0]['lsoa_code'].nunique()}")

all_months = sorted(merged['month'].unique())
all_lsoas = sorted(merged['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

# Get ridership per LSOA-month (dynamic, varies by month)
ridership_lookup = merged[['lsoa_code', 'month'] + ridership_features].drop_duplicates(['lsoa_code', 'month']).set_index(['lsoa_code', 'month'])
crime_grid = merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
df = crime_grid.reindex(grid, fill_value=0).reset_index()
df = df.join(ridership_lookup, on=['lsoa_code', 'month'], how='left')
df[ridership_features] = df[ridership_features].fillna(0)

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

lag_features = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
temporal_features = ['month_sin', 'month_cos', 'time_idx']
crime_only_cols = lag_features + temporal_features
fused_cols = lag_features + temporal_features + ridership_features

print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"   Crime-only features: {len(crime_only_cols)}")
print(f"   Fused features: {len(fused_cols)} (+{len(ridership_features)} ridership)")

rf_crime = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_crime.fit(train_df[crime_only_cols], train_df['crime_count'])

rf_fused = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_fused.fit(train_df[fused_cols], train_df['crime_count'])

for name, model, cols in [("Crime-only", rf_crime, crime_only_cols), ("Fused (+Ridership)", rf_fused, fused_cols)]:
    y_pred = model.predict(test_df[cols])
    r2 = r2_score(test_df['crime_count'], y_pred)
    print(f"\n   {name}: R²={r2:.4f}")

delta_r2 = r2_score(test_df['crime_count'], rf_fused.predict(test_df[fused_cols])) - r2_score(test_df['crime_count'], rf_crime.predict(test_df[crime_only_cols]))
print(f"\n   Δ R² = {delta_r2:+.4f}")

# Feature importance
print("\n   Feature importance (ridership features):")
imp = pd.DataFrame({'feature': fused_cols, 'importance': rf_fused.feature_importances_}).sort_values('importance', ascending=False)
for _, row in imp[imp['feature'].isin(ridership_features)].iterrows():
    print(f"     {row['feature']:30s} {row['importance']:.4f}")

# ── PER-CRIME-TYPE ──
print(f"\n{'='*70}")
print("PER-CRIME-TYPE RIDERSHIP ANALYSIS")
print(f"{'='*70}")

crime_types = db.query("SELECT DISTINCT crime_type FROM crime_clean ORDER BY crime_type")['crime_type'].tolist()
per_type_results = []

for ct in crime_types:
    print(f"\n  {ct}...", end=" ", flush=True)
    ct_monthly = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{ct.replace(chr(39), chr(39)+chr(39))}' GROUP BY lsoa_code, month")
    ct_merged = ct_monthly.merge(lsoa_ridership[['lsoa_code', 'month'] + ridership_features],
                                  on=['lsoa_code', 'month'], how='left')
    ct_merged[ridership_features] = ct_merged[ridership_features].fillna(0)

    lsoa_totals = ct_merged.groupby('lsoa_code')['crime_count'].sum()
    active_ct = lsoa_totals[lsoa_totals >= 12].index
    if len(active_ct) < 50: print("SKIP"); continue
    ct_merged = ct_merged[ct_merged['lsoa_code'].isin(active_ct)]

    ct_months = sorted(ct_merged['month'].unique())
    ct_lsoas = sorted(ct_merged['lsoa_code'].unique())
    ct_grid = pd.MultiIndex.from_product([ct_lsoas, ct_months], names=['lsoa_code', 'month'])
    ct_df = ct_merged[['lsoa_code','month','crime_count']].set_index(['lsoa_code','month']).reindex(ct_grid, fill_value=0).reset_index()
    # Join ridership (dynamic)
    ride_lookup = ct_merged[['lsoa_code','month'] + ridership_features].drop_duplicates(['lsoa_code','month']).set_index(['lsoa_code','month'])
    ct_df = ct_df.join(ride_lookup, on=['lsoa_code','month'], how='left')
    ct_df[ridership_features] = ct_df[ridership_features].fillna(0)
    ct_df = ct_df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        ct_df[f'lag_{lag}'] = ct_df.groupby('lsoa_code')['crime_count'].shift(lag)
    ct_df['rolling_mean_3'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ct_df['rolling_mean_12'] = ct_df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    ct_df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)
    ct_df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(ct_df['month']).dt.month / 12)

    ct_model = ct_df.dropna()
    if len(ct_model) < 100: print("SKIP"); continue
    test_months_ct = ct_months[-6:]
    train = ct_model[~ct_model['month'].isin(test_months_ct)]
    test = ct_model[ct_model['month'].isin(test_months_ct)]
    if len(test) < 50: print("SKIP"); continue

    lag_cols = ['lag_1','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_12','month_sin','month_cos']

    rf_base = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_base.fit(train[lag_cols], train['crime_count'])
    r2_base = r2_score(test['crime_count'], rf_base.predict(test[lag_cols]))

    rf_ride = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_ride.fit(train[lag_cols + ridership_features], train['crime_count'])
    r2_ride = r2_score(test['crime_count'], rf_ride.predict(test[lag_cols + ridership_features]))

    delta = r2_ride - r2_base
    print(f"Δ={delta:+.4f}")
    per_type_results.append({'crime_type': ct, 'r2_base': r2_base, 'r2_ride': r2_ride, 'delta': delta, 'n_lsoas': len(active_ct)})

db.close()

res = pd.DataFrame(per_type_results).sort_values('delta', ascending=False)
print(f"\n{'='*90}")
print("SUMMARY: STATION RIDERSHIP IMPACT BY CRIME TYPE")
print(f"{'='*90}")
print(f"\n{'Crime Type':<35} {'R²(base)':>8} {'Δ Ride':>8}")
print("-" * 55)
for _, r in res.iterrows():
    print(f"{r['crime_type']:<35} {r['r2_base']:>8.4f} {r['delta']:>+8.4f}")

print(f"\n   Aggregate Δ R² = {delta_r2:+.4f}")
print("\n✅ Station footfall fusion experiment complete!")
