"""
Script 32: Concentric Radius Experiment
Tests how transport feature predictive value decays as the analysis boundary expands
from central London outward. Implements the supervisor's "concentric rings" methodology.

Centre point: Charing Cross (51.5074N, 0.1278W) - geographic centre of London

For each radius R (2, 4, 6, 8, 10, 12, 15, 20, 25, 30 km, full London):
  1. Filter LSOAs whose centroid falls within R km of Charing Cross
  2. Run the standard ablation: crime-only vs crime+transport features
  3. Record Δ R² for each transport source and the combined transport block
  4. Plot the "Δ R² vs radius" decay curve

This experiment aims to show:
  - Transport features have genuine predictive value where data is dense (inner rings)
  - Signal decays monotonically as more zero-fill outer LSOAs are included
  - The aggregate near-zero Δ R² is an artefact of spatial sparsity, not feature irrelevance
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "transport"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / "data/processed/london"
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150; plt.rcParams["savefig.dpi"] = 150

# ============================================================
# CONFIGURATION
# ============================================================
CHARING_CROSS = (51.5074, -0.1278)  # lat, lon
RADII_KM = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, None]  # None = full London
RADIUS_LABELS = ['2km', '4km', '6km', '8km', '10km', '12km', '15km', '20km', '25km', '30km', 'All']

print("=" * 70)
print("EXPERIMENT: CONCENTRIC RADIUS TRANSPORT SIGNAL DECAY")
print("=" * 70)
print(f"Centre: Charing Cross ({CHARING_CROSS[0]}, {CHARING_CROSS[1]})")
print(f"Radii: {RADIUS_LABELS}")

# ============================================================
# STEP 1: Compute LSOA centroid distances from Charing Cross
# ============================================================
print("\n1. Computing LSOA centroid distances...")
lsoa_path = PROJECT_ROOT / "data/raw/london/boundaries/lsoa_2021_london.geojson"
lsoa_gdf = gpd.read_file(lsoa_path)

# Find LSOA code column
lsoa_code_col = [c for c in lsoa_gdf.columns if 'LSOA' in c.upper() and 'CD' in c.upper()][0]
print(f"   LSOA code column: {lsoa_code_col}")
print(f"   Total LSOAs: {len(lsoa_gdf):,}")

# Project to British National Grid (EPSG:27700) for accurate distance in metres
lsoa_proj = lsoa_gdf.to_crs(epsg=27700)
centroids = lsoa_proj.geometry.centroid

# Charing Cross in projected coordinates
from shapely.geometry import Point
cx_point = gpd.GeoSeries([Point(CHARING_CROSS[1], CHARING_CROSS[0])], crs="EPSG:4326")
cx_proj = cx_point.to_crs(epsg=27700).iloc[0]

# Compute distances in km
distances = centroids.distance(cx_proj) / 1000.0  # metres to km
lsoa_gdf['dist_km'] = distances.values
lsoa_gdf['lsoa_code'] = lsoa_gdf[lsoa_code_col]

# Distribution summary
print(f"   Distance range: {lsoa_gdf['dist_km'].min():.1f} km to {lsoa_gdf['dist_km'].max():.1f} km")
print(f"   Median distance: {lsoa_gdf['dist_km'].median():.1f} km")
for r in [2, 5, 10, 15, 20, 30]:
    n = (lsoa_gdf['dist_km'] <= r).sum()
    print(f"   LSOAs within {r:>2}km: {n:>5,}")

# Save distance lookup
dist_lookup = lsoa_gdf[['lsoa_code', 'dist_km']].copy()
dist_lookup.to_csv(PROCESSED_DIR / "lsoa_centroid_distances.csv", index=False)

# ============================================================
# STEP 2: Load all transport features
# ============================================================
print("\n2. Loading transport features...")

# PTAL (static, per LSOA)
ptal = pd.read_csv(PROJECT_ROOT / "data/raw/london/transport/ptal/ptal_lsoa_2023.csv")
ptal_cat_map = {'0': 0, '1a': 1, '1b': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6a': 7, '6b': 8}
ptal['ptal_category_num'] = ptal['MEAN_PTAL_'].map(ptal_cat_map)
ptal_features_df = ptal[['LSOA21CD', 'mean_AI', 'MAX_AI', 'ptal_category_num']].rename(columns={
    'LSOA21CD': 'lsoa_code', 'mean_AI': 'ptal_mean_ai', 'MAX_AI': 'ptal_max_ai'
})
ptal_feature_cols = ['ptal_mean_ai', 'ptal_max_ai', 'ptal_category_num']
print(f"   PTAL: {len(ptal_features_df):,} LSOAs, {len(ptal_feature_cols)} features")

# Station ridership (dynamic, per LSOA per month)
ridership = pd.read_csv(PROCESSED_DIR / "station_ridership_monthly.csv")
ridership_features = ['ridership_total', 'station_count', 'ridership_per_station']
print(f"   Station Ridership: {ridership['lsoa_code'].nunique()} LSOAs, {len(ridership_features)} features (dynamic)")

# Santander (dynamic, per LSOA per month)
bikes = pd.read_csv(PROCESSED_DIR / "santander_monthly.csv")
bike_features = ['bike_total', 'bike_stations', 'bike_per_station']
print(f"   Santander Cycles: {bikes['lsoa_code'].nunique()} LSOAs, {len(bike_features)} features (dynamic)")

# All transport features combined
all_transport = ptal_feature_cols + ridership_features + bike_features
print(f"   Combined transport features: {len(all_transport)}")

# ============================================================
# STEP 3: Load crime data
# ============================================================
print("\n3. Loading crime data...")
db = ThesisDB()
monthly_crime = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean GROUP BY lsoa_code, month ORDER BY lsoa_code, month
""")
all_months_global = sorted(monthly_crime['month'].unique())
print(f"   Crime data: {len(monthly_crime):,} rows, {monthly_crime['lsoa_code'].nunique()} LSOAs, {len(all_months_global)} months")
db.close()

# ============================================================
# STEP 4: Run experiment at each radius
# ============================================================
print(f"\n{'='*70}")
print("4. RUNNING CONCENTRIC RADIUS EXPERIMENTS")
print(f"{'='*70}")

results = []

for radius, label in zip(RADII_KM, RADIUS_LABELS):
    if radius is not None:
        ring_lsoas = set(dist_lookup[dist_lookup['dist_km'] <= radius]['lsoa_code'])
    else:
        ring_lsoas = set(dist_lookup['lsoa_code'])

    print(f"\n--- Radius: {label} ({len(ring_lsoas):,} LSOAs) ---")

    # Filter crime to this ring
    crime_ring = monthly_crime[monthly_crime['lsoa_code'].isin(ring_lsoas)].copy()

    # Min crimes filter
    MIN_CRIMES = 36
    lsoa_totals = crime_ring.groupby('lsoa_code')['crime_count'].sum()
    active = lsoa_totals[lsoa_totals >= MIN_CRIMES].index
    crime_ring = crime_ring[crime_ring['lsoa_code'].isin(active)]
    n_active = crime_ring['lsoa_code'].nunique()

    if n_active < 30:
        print(f"   SKIP: Only {n_active} active LSOAs (need >= 30)")
        results.append({
            'radius': label, 'radius_km': radius or 999, 'n_lsoas': n_active,
            'r2_base': np.nan, 'delta_ptal': np.nan, 'delta_ride': np.nan,
            'delta_bike': np.nan, 'delta_combined': np.nan
        })
        continue

    print(f"   Active LSOAs: {n_active:,}")

    # Count transport coverage in this ring
    ptal_in_ring = len(set(ptal_features_df['lsoa_code']) & set(active))
    ride_in_ring = len(set(ridership['lsoa_code'].unique()) & set(active))
    bike_in_ring = len(set(bikes['lsoa_code'].unique()) & set(active))
    print(f"   PTAL coverage: {ptal_in_ring}/{n_active} ({ptal_in_ring/n_active*100:.0f}%)")
    print(f"   Station coverage: {ride_in_ring}/{n_active} ({ride_in_ring/n_active*100:.0f}%)")
    print(f"   Bike coverage: {bike_in_ring}/{n_active} ({bike_in_ring/n_active*100:.0f}%)")

    # Build feature matrix
    all_months = sorted(crime_ring['month'].unique())
    all_lsoas = sorted(crime_ring['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])

    crime_grid = crime_ring[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month'])
    df = crime_grid.reindex(grid, fill_value=0).reset_index()

    # Join PTAL (static)
    ptal_per = ptal_features_df.set_index('lsoa_code')
    df = df.join(ptal_per, on='lsoa_code', how='left')
    df[ptal_feature_cols] = df[ptal_feature_cols].fillna(0)

    # Join ridership (dynamic)
    ride_lk = ridership[['lsoa_code', 'month'] + ridership_features].set_index(['lsoa_code', 'month'])
    df = df.join(ride_lk, on=['lsoa_code', 'month'], how='left')
    df[ridership_features] = df[ridership_features].fillna(0)

    # Join bikes (dynamic)
    bike_lk = bikes[['lsoa_code', 'month'] + bike_features].set_index(['lsoa_code', 'month'])
    df = df.join(bike_lk, on=['lsoa_code', 'month'], how='left')
    df[bike_features] = df[bike_features].fillna(0)

    # Lag features
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_6'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    month_idx = {m: i for i, m in enumerate(all_months)}
    df['time_idx'] = df['month'].map(month_idx)

    # Train/test split
    df_model = df.dropna().copy()
    test_months = all_months[-6:]
    train = df_model[~df_model['month'].isin(test_months)]
    test = df_model[df_model['month'].isin(test_months)]

    if len(test) < 30:
        print(f"   SKIP: Test set too small ({len(test)})")
        results.append({
            'radius': label, 'radius_km': radius or 999, 'n_lsoas': n_active,
            'r2_base': np.nan, 'delta_ptal': np.nan, 'delta_ride': np.nan,
            'delta_bike': np.nan, 'delta_combined': np.nan
        })
        continue

    lag_cols = [f'lag_{l}' for l in [1,2,3,6,12]] + ['rolling_mean_3','rolling_mean_6','rolling_mean_12']
    temporal_cols = ['month_sin', 'month_cos', 'time_idx']
    base_cols = lag_cols + temporal_cols

    # A: Crime-only baseline
    rf_base = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_base.fit(train[base_cols], train['crime_count'])
    r2_base = r2_score(test['crime_count'], rf_base.predict(test[base_cols]))
    mae_base = mean_absolute_error(test['crime_count'], rf_base.predict(test[base_cols]))

    # B: + PTAL only
    ptal_cols = base_cols + ptal_feature_cols
    rf_ptal = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_ptal.fit(train[ptal_cols], train['crime_count'])
    r2_ptal = r2_score(test['crime_count'], rf_ptal.predict(test[ptal_cols]))

    # C: + Station Ridership only
    ride_cols = base_cols + ridership_features
    rf_ride = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_ride.fit(train[ride_cols], train['crime_count'])
    r2_ride = r2_score(test['crime_count'], rf_ride.predict(test[ride_cols]))

    # D: + Santander only
    bike_cols_full = base_cols + bike_features
    rf_bike = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_bike.fit(train[bike_cols_full], train['crime_count'])
    r2_bike = r2_score(test['crime_count'], rf_bike.predict(test[bike_cols_full]))

    # E: All transport combined
    combined_cols = base_cols + all_transport
    rf_combined = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_combined.fit(train[combined_cols], train['crime_count'])
    r2_combined = r2_score(test['crime_count'], rf_combined.predict(test[combined_cols]))
    mae_combined = mean_absolute_error(test['crime_count'], rf_combined.predict(test[combined_cols]))

    delta_ptal = r2_ptal - r2_base
    delta_ride = r2_ride - r2_base
    delta_bike = r2_bike - r2_base
    delta_combined = r2_combined - r2_base

    print(f"   R²(base) = {r2_base:.4f}  MAE(base) = {mae_base:.4f}")
    print(f"   Δ PTAL     = {delta_ptal:+.4f}")
    print(f"   Δ Ridership = {delta_ride:+.4f}")
    print(f"   Δ Bikes     = {delta_bike:+.4f}")
    print(f"   Δ Combined  = {delta_combined:+.4f}  MAE(combined) = {mae_combined:.4f}")

    results.append({
        'radius': label, 'radius_km': radius if radius else 999, 'n_lsoas': n_active,
        'r2_base': r2_base, 'mae_base': mae_base,
        'delta_ptal': delta_ptal, 'delta_ride': delta_ride,
        'delta_bike': delta_bike, 'delta_combined': delta_combined,
        'mae_combined': mae_combined,
        'ptal_coverage': ptal_in_ring/n_active,
        'ride_coverage': ride_in_ring/n_active,
        'bike_coverage': bike_in_ring/n_active,
    })

# ============================================================
# STEP 5: Results table and plot
# ============================================================
res = pd.DataFrame(results)
res.to_csv(PROCESSED_DIR / "concentric_radius_results.csv", index=False)

print(f"\n{'='*90}")
print("CONCENTRIC RADIUS RESULTS")
print(f"{'='*90}")
print(f"\n{'Radius':<8} {'LSOAs':>6} {'R²(base)':>9} {'Δ PTAL':>8} {'Δ Ride':>8} {'Δ Bike':>8} {'Δ Comb':>8} {'Ride%':>6} {'Bike%':>6}")
print("-" * 80)
for _, r in res.iterrows():
    if pd.isna(r['r2_base']): continue
    print(f"{r['radius']:<8} {r['n_lsoas']:>6} {r['r2_base']:>9.4f} {r['delta_ptal']:>+8.4f} {r['delta_ride']:>+8.4f} {r['delta_bike']:>+8.4f} {r['delta_combined']:>+8.4f} {r.get('ride_coverage',0):>5.0%} {r.get('bike_coverage',0):>5.0%}")

# ── PLOT: Δ R² vs Radius ──
valid = res.dropna(subset=['r2_base']).copy()
# Replace 999 with max actual distance + margin for plotting
max_dist = lsoa_gdf['dist_km'].max()
valid.loc[valid['radius_km'] == 999, 'radius_km'] = max_dist

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(valid['radius_km'], valid['delta_combined'], 'o-', color='#dc2626', linewidth=2.5,
        markersize=8, label='Combined Transport', zorder=5)
ax.plot(valid['radius_km'], valid['delta_ptal'], 's--', color='#2563eb', linewidth=1.5,
        markersize=6, label='PTAL (static)', alpha=0.8)
ax.plot(valid['radius_km'], valid['delta_ride'], '^--', color='#16a34a', linewidth=1.5,
        markersize=6, label='Station Ridership', alpha=0.8)
ax.plot(valid['radius_km'], valid['delta_bike'], 'D--', color='#f59e0b', linewidth=1.5,
        markersize=6, label='Santander Cycles', alpha=0.8)

ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Radius from Charing Cross (km)', fontsize=13)
ax.set_ylabel('Δ R² (Transport vs Crime-Only)', fontsize=13)
ax.set_title('Transport Feature Signal Decay by Analysis Radius', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')

# Add LSOA count annotation on secondary axis
ax2 = ax.twinx()
ax2.bar(valid['radius_km'], valid['n_lsoas'], alpha=0.1, color='gray', width=1.5, label='LSOAs')
ax2.set_ylabel('Number of LSOAs', fontsize=11, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_concentric_radius_decay.png", bbox_inches='tight')
plt.close()
print(f"\n   Plot saved: {FIGURES_DIR / '01_concentric_radius_decay.png'}")

# ── PLOT: Transport coverage vs radius ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(valid['radius_km'], valid.get('ptal_coverage', 0) * 100, 's-', color='#2563eb',
        linewidth=2, markersize=7, label='PTAL')
ax.plot(valid['radius_km'], valid.get('ride_coverage', 0) * 100, '^-', color='#16a34a',
        linewidth=2, markersize=7, label='Station Ridership')
ax.plot(valid['radius_km'], valid.get('bike_coverage', 0) * 100, 'D-', color='#f59e0b',
        linewidth=2, markersize=7, label='Santander Cycles')
ax.set_xlabel('Radius from Charing Cross (km)', fontsize=13)
ax.set_ylabel('Transport Data Coverage (%)', fontsize=13)
ax.set_title('Transport Data Density by Analysis Radius', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_transport_coverage_by_radius.png", bbox_inches='tight')
plt.close()
print(f"   Plot saved: {FIGURES_DIR / '02_transport_coverage_by_radius.png'}")

# ── PLOT: Base R² vs radius ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(valid['radius_km'], valid['r2_base'], 'o-', color='#6366f1', linewidth=2, markersize=7)
ax.set_xlabel('Radius from Charing Cross (km)', fontsize=13)
ax.set_ylabel('Baseline R² (Crime-Only)', fontsize=13)
ax.set_title('Baseline Model Performance by Analysis Radius', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_baseline_r2_by_radius.png", bbox_inches='tight')
plt.close()
print(f"   Plot saved: {FIGURES_DIR / '03_baseline_r2_by_radius.png'}")

print("\n✅ Concentric radius experiment complete!")
print(f"   Results saved: {PROCESSED_DIR / 'concentric_radius_results.csv'}")
print(f"   Figures saved: {FIGURES_DIR}")
