"""
Step 3: Schema Validation & Exploratory Data Analysis (EDA)
============================================================

This script performs comprehensive EDA on the London crime dataset:
1. Schema validation
2. Data quality checks (nulls, duplicates, temporal gaps)
3. Temporal analysis (monthly crime trends, seasonality)
4. Crime type breakdown
5. Spatial analysis (LSOA-level crime density)
6. Outcome analysis

Outputs: figures saved to reports/figures/eda/

Usage:
    python notebooks/eda/01_crime_eda.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (12, 6)

db = ThesisDB()

print("=" * 70)
print("STEP 3: SCHEMA VALIDATION & EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ===================================================================
# 1. SCHEMA VALIDATION
# ===================================================================
print("\n" + "=" * 70)
print("1. SCHEMA VALIDATION")
print("=" * 70)

schema = db.table_info()
print("\nTable schema:")
print(schema.to_string())

expected_cols = {
    "crime_id", "month", "reported_by", "falls_within",
    "longitude", "latitude", "location", "lsoa_code",
    "lsoa_name", "crime_type", "outcome", "context"
}
actual_cols = set(schema["column_name"])
assert expected_cols == actual_cols, f"Schema mismatch! Missing: {expected_cols - actual_cols}"
print("\n✅ Schema validated — all 12 expected columns present")

# ===================================================================
# 2. DATA QUALITY CHECKS
# ===================================================================
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECKS")
print("=" * 70)

# 2a. Null analysis
nulls = db.query("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN crime_id IS NULL OR crime_id = '' THEN 1 ELSE 0 END) as null_crime_id,
        SUM(CASE WHEN month IS NULL OR month = '' THEN 1 ELSE 0 END) as null_month,
        SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) as null_longitude,
        SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) as null_latitude,
        SUM(CASE WHEN lsoa_code IS NULL OR lsoa_code = '' THEN 1 ELSE 0 END) as null_lsoa_code,
        SUM(CASE WHEN crime_type IS NULL OR crime_type = '' THEN 1 ELSE 0 END) as null_crime_type,
        SUM(CASE WHEN outcome IS NULL OR outcome = '' THEN 1 ELSE 0 END) as null_outcome
    FROM crime
""")
print("\nNull/empty counts:")
for col in nulls.columns:
    val = nulls[col].iloc[0]
    pct = (val / nulls["total"].iloc[0]) * 100 if col != "total" else 0
    print(f"  {col:20s}: {val:>10,} ({pct:.2f}%)" if col != "total" else f"  {col:20s}: {val:>10,}")

# 2b. Duplicate crime IDs
dupes = db.query("""
    SELECT COUNT(*) as dupe_count
    FROM (
        SELECT crime_id
        FROM crime 
        WHERE crime_id IS NOT NULL AND crime_id != ''
        GROUP BY crime_id
        HAVING COUNT(*) > 1
    )
""")
print(f"\n  Duplicate crime IDs: {dupes['dupe_count'].iloc[0]:,}")

# 2c. Check for crimes with same ID but different details
if dupes['dupe_count'].iloc[0] > 0:
    dupe_sample = db.query("""
        SELECT crime_id, month, crime_type, lsoa_code, COUNT(*) as n
        FROM crime 
        WHERE crime_id IN (
            SELECT crime_id FROM crime 
            WHERE crime_id IS NOT NULL AND crime_id != ''
            GROUP BY crime_id HAVING COUNT(*) > 1
        )
        GROUP BY crime_id, month, crime_type, lsoa_code
        ORDER BY n DESC
        LIMIT 10
    """)
    print("  Sample duplicates:")
    print(dupe_sample.to_string(index=False))

# 2d. Temporal coverage check
months = db.query("""
    SELECT month, COUNT(*) as crime_count
    FROM crime
    GROUP BY month
    ORDER BY month
""")
print(f"\n  Months covered: {len(months)}")
print(f"  Range: {months['month'].iloc[0]} → {months['month'].iloc[-1]}")

# Check for gaps
all_months = pd.date_range(
    start=months['month'].iloc[0], 
    end=months['month'].iloc[-1], 
    freq='MS'
).strftime('%Y-%m').tolist()
missing_months = set(all_months) - set(months['month'].tolist())
if missing_months:
    print(f"  ⚠️  Missing months: {sorted(missing_months)}")
else:
    print("  ✅ No temporal gaps — all months present")

# ===================================================================
# 3. TEMPORAL ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("3. TEMPORAL ANALYSIS")
print("=" * 70)

# 3a. Monthly crime trend
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(months)), months['crime_count'], 'o-', linewidth=2, markersize=4, color='#2563eb')
ax.set_xlabel('Month')
ax.set_ylabel('Total Crimes')
ax.set_title('Monthly Crime Volume — London (Feb 2023 – Jan 2026)', fontsize=14, fontweight='bold')

# Set x-axis labels every 3 months
tick_positions = list(range(0, len(months), 3))
tick_labels = [months['month'].iloc[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

# Add mean line
mean_crimes = months['crime_count'].mean()
ax.axhline(y=mean_crimes, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_crimes/1000:.1f}K/month')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_monthly_crime_trend.png")
plt.close()
print(f"  Mean monthly crimes: {mean_crimes:,.0f}")
print(f"  Min: {months['crime_count'].min():,} ({months.loc[months['crime_count'].idxmin(), 'month']})")
print(f"  Max: {months['crime_count'].max():,} ({months.loc[months['crime_count'].idxmax(), 'month']})")

# 3b. Year-over-year comparison
months_df = months.copy()
months_df['year'] = months_df['month'].str[:4]
months_df['mon'] = months_df['month'].str[5:]
yearly = months_df.groupby('year')['crime_count'].agg(['sum', 'mean', 'count']).reset_index()
yearly.columns = ['year', 'total', 'avg_monthly', 'months']
print("\n  Year-over-year summary:")
print(yearly.to_string(index=False))

# ===================================================================
# 4. CRIME TYPE ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("4. CRIME TYPE ANALYSIS")
print("=" * 70)

crime_types = db.query("""
    SELECT crime_type, COUNT(*) as count,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM crime
    GROUP BY crime_type
    ORDER BY count DESC
""")
print("\nCrime type distribution:")
print(crime_types.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette("Blues_r", len(crime_types))
bars = ax.barh(range(len(crime_types)), crime_types['count'], color=colors)
ax.set_yticks(range(len(crime_types)))
ax.set_yticklabels(crime_types['crime_type'], fontsize=10)
ax.set_xlabel('Number of Crimes')
ax.set_title('Crime Type Distribution — London (Feb 2023 – Jan 2026)', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax.invert_yaxis()

# Add percentage labels
for i, (count, pct) in enumerate(zip(crime_types['count'], crime_types['pct'])):
    ax.text(count + 5000, i, f'{pct:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_crime_type_distribution.png")
plt.close()

# 4b. Crime type trends over time (top 5)
top5 = crime_types['crime_type'].head(5).tolist()
crime_trends = db.query(f"""
    SELECT month, crime_type, COUNT(*) as count
    FROM crime
    WHERE crime_type IN ({','.join(f"'{t}'" for t in top5)})
    GROUP BY month, crime_type
    ORDER BY month, crime_type
""")

fig, ax = plt.subplots(figsize=(14, 7))
for ctype in top5:
    subset = crime_trends[crime_trends['crime_type'] == ctype]
    ax.plot(range(len(subset)), subset['count'], 'o-', label=ctype, markersize=3, linewidth=1.5)

tick_positions = list(range(0, len(months), 3))
tick_labels = [months['month'].iloc[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.set_ylabel('Monthly Crime Count')
ax.set_title('Top 5 Crime Types — Monthly Trends', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_top5_crime_trends.png")
plt.close()

# ===================================================================
# 5. SPATIAL ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("5. SPATIAL ANALYSIS")
print("=" * 70)

# 5a. LSOA-level crime density
lsoa_stats = db.query("""
    SELECT 
        lsoa_code,
        lsoa_name,
        COUNT(*) as total_crimes,
        COUNT(DISTINCT month) as n_months,
        COUNT(DISTINCT crime_type) as n_crime_types,
        ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT month), 1) as avg_monthly
    FROM crime
    WHERE lsoa_code IS NOT NULL AND lsoa_code != ''
    GROUP BY lsoa_code, lsoa_name
    ORDER BY total_crimes DESC
""")

print(f"\n  Total LSOAs with crime data: {len(lsoa_stats):,}")
print(f"\n  Crime density distribution (per LSOA, total 36 months):")
print(f"    Mean:   {lsoa_stats['total_crimes'].mean():,.1f}")
print(f"    Median: {lsoa_stats['total_crimes'].median():,.1f}")
print(f"    Std:    {lsoa_stats['total_crimes'].std():,.1f}")
print(f"    Min:    {lsoa_stats['total_crimes'].min():,}")
print(f"    Max:    {lsoa_stats['total_crimes'].max():,}")

print("\n  Top 10 hotspot LSOAs:")
print(lsoa_stats.head(10)[['lsoa_code', 'lsoa_name', 'total_crimes', 'avg_monthly']].to_string(index=False))

print("\n  Bottom 10 (lowest crime) LSOAs:")
print(lsoa_stats.tail(10)[['lsoa_code', 'lsoa_name', 'total_crimes', 'avg_monthly']].to_string(index=False))

# Histogram of crime density
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(lsoa_stats['total_crimes'], bins=100, color='#2563eb', edgecolor='white', alpha=0.8)
ax.axvline(lsoa_stats['total_crimes'].mean(), color='red', linestyle='--', label=f'Mean: {lsoa_stats["total_crimes"].mean():,.0f}')
ax.axvline(lsoa_stats['total_crimes'].median(), color='orange', linestyle='--', label=f'Median: {lsoa_stats["total_crimes"].median():,.0f}')
ax.set_xlabel('Total Crimes per LSOA (36 months)')
ax.set_ylabel('Number of LSOAs')
ax.set_title('Crime Density Distribution Across LSOAs', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_lsoa_crime_density.png")
plt.close()

# ===================================================================
# 6. OUTCOME ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("6. OUTCOME ANALYSIS")
print("=" * 70)

outcomes = db.query("""
    SELECT 
        COALESCE(outcome, 'Unknown/Missing') as outcome,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM crime
    GROUP BY outcome
    ORDER BY count DESC
""")
print("\nOutcome distribution:")
print(outcomes.to_string(index=False))

# ===================================================================
# 7. COORDINATE ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("7. COORDINATE ANALYSIS")
print("=" * 70)

coords = db.query("""
    SELECT 
        MIN(longitude) as min_lon, MAX(longitude) as max_lon,
        MIN(latitude) as min_lat, MAX(latitude) as max_lat,
        AVG(longitude) as avg_lon, AVG(latitude) as avg_lat,
        COUNT(CASE WHEN longitude IS NOT NULL AND latitude IS NOT NULL THEN 1 END) as has_coords,
        COUNT(CASE WHEN longitude IS NULL OR latitude IS NULL THEN 1 END) as no_coords
    FROM crime
""")
print(f"\n  Longitude range: {coords['min_lon'].iloc[0]:.4f} → {coords['max_lon'].iloc[0]:.4f}")
print(f"  Latitude range:  {coords['min_lat'].iloc[0]:.4f} → {coords['max_lat'].iloc[0]:.4f}")
print(f"  Centre: ({coords['avg_lon'].iloc[0]:.4f}, {coords['avg_lat'].iloc[0]:.4f})")
print(f"  With coordinates: {coords['has_coords'].iloc[0]:,}")
print(f"  Without coordinates: {coords['no_coords'].iloc[0]:,}")

# ===================================================================
# 8. FORCE BREAKDOWN
# ===================================================================
print("\n" + "=" * 70)
print("8. FORCE BREAKDOWN")
print("=" * 70)

forces = db.query("""
    SELECT falls_within, COUNT(*) as count,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM crime
    GROUP BY falls_within
    ORDER BY count DESC
""")
print(forces.to_string(index=False))

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)
print(f"""
Total crimes:           {nulls['total'].iloc[0]:,}
Date range:             {months['month'].iloc[0]} → {months['month'].iloc[-1]} ({len(months)} months)
Crime types:            {len(crime_types)}
Unique LSOAs:           {len(lsoa_stats):,}
Null LSOA codes:        {nulls['null_lsoa_code'].iloc[0]:,} ({nulls['null_lsoa_code'].iloc[0]/nulls['total'].iloc[0]*100:.2f}%)
Duplicate crime IDs:    {dupes['dupe_count'].iloc[0]:,}
Temporal gaps:          {'None' if not missing_months else sorted(missing_months)}
Mean monthly crimes:    {mean_crimes:,.0f}

Figures saved to: {FIGURES_DIR}
""")

# List saved figures
for fig_path in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  📊 {fig_path.name}")

db.close()
print("\n✅ EDA complete!")
