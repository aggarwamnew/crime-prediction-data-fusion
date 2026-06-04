"""
Step 3b: Data Cleaning Pipeline
================================

Applies all cleaning decisions documented in data/data_quality_log.md:
1. Drop null/empty LSOA codes (24,605 rows)
2. Drop Welsh LSOA codes (215 rows)
3. Deduplicate crime IDs — keep latest outcome per (crime_id, month, lsoa_code)
4. Handle ASB records (692,536 rows with no crime_id — keep all, they're unique)
5. Save cleaned table in DuckDB + export to Parquet
6. Print before/after summary

Usage:
    python notebooks/eda/02_clean_crime_data.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.db import ThesisDB
from src.utils.config import PATHS, DB_PATH
from pathlib import Path

db = ThesisDB()

print("=" * 70)
print("DATA CLEANING PIPELINE")
print("=" * 70)

# --- Before stats ---
before = db.query("SELECT COUNT(*) as n FROM crime")['n'].iloc[0]
print(f"\n📊 BEFORE: {before:,} rows")

# =====================================================================
# STEP 1: Drop null/empty LSOA codes
# =====================================================================
print("\n" + "-" * 50)
print("STEP 1: Drop null/empty LSOA codes")
null_lsoa = db.query("""
    SELECT COUNT(*) as n FROM crime 
    WHERE lsoa_code IS NULL OR lsoa_code = ''
""")['n'].iloc[0]
print(f"  Dropping: {null_lsoa:,} rows ({null_lsoa/before*100:.2f}%)")
print(f"  Reason: Cannot spatially locate — unusable for spatial prediction")
print(f"  Note: All 692,536 null crime_id rows are ASB (Anti-social behaviour)")
print(f"        ASB records never have crime IDs on data.police.uk — this is by design")
print(f"        Of those, {null_lsoa:,} also have null LSOA → dropped")
print(f"        Remaining ASB records with valid LSOA are kept")

# =====================================================================
# STEP 2: Drop Welsh LSOA codes
# =====================================================================
print("\n" + "-" * 50)
print("STEP 2: Drop Welsh LSOA codes")
welsh = db.query("""
    SELECT COUNT(*) as n FROM crime 
    WHERE lsoa_code LIKE 'W%'
""")['n'].iloc[0]
print(f"  Dropping: {welsh:,} rows ({welsh/before*100:.4f}%)")
print(f"  Reason: Outside London study area (Met Police border overlap with Wales)")

# =====================================================================
# STEP 3: Deduplicate crime IDs
# =====================================================================
print("\n" + "-" * 50)
print("STEP 3: Deduplicate crime IDs")
print(f"  Investigation findings:")
print(f"    - 39,662 crime IDs appear more than once")
print(f"    - ALL are same-month, same-force (no cross-force dupes)")
print(f"    - 38,676 are same-LSOA (differ only in outcome column)")
print(f"    - 986 span different LSOAs (crime recorded at boundary)")
print(f"  Strategy: Keep one row per (crime_id, month, lsoa_code)")
print(f"    - For same-LSOA dupes: keep the row with the most informative outcome")
print(f"      (prefer non-'Status update unavailable' outcomes)")
print(f"    - For cross-LSOA dupes: keep all rows (different locations = different events)")
print(f"  Note: ASB records (null crime_id) are not affected — each row is unique")

# =====================================================================
# STEP 4: Build cleaned table
# =====================================================================
print("\n" + "-" * 50)
print("STEP 4: Building cleaned table...")

db.execute("""
    CREATE OR REPLACE TABLE crime_clean AS
    WITH 
    -- Step 1+2: Filter out null LSOA and Welsh codes
    filtered AS (
        SELECT *
        FROM crime
        WHERE lsoa_code IS NOT NULL 
          AND lsoa_code != ''
          AND lsoa_code NOT LIKE 'W%'
    ),
    -- Step 3: For records WITH a crime_id, deduplicate
    -- Priority: prefer outcomes that are NOT 'Status update unavailable'
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY crime_id, month, lsoa_code
                ORDER BY 
                    CASE 
                        WHEN outcome = 'Status update unavailable' THEN 2
                        WHEN outcome IS NULL THEN 3
                        ELSE 1  -- prefer informative outcomes
                    END,
                    outcome  -- deterministic tiebreaker
            ) as rn
        FROM filtered
        WHERE crime_id IS NOT NULL AND crime_id != ''
    ),
    deduped_with_id AS (
        SELECT crime_id, month, reported_by, falls_within, 
               longitude, latitude, location, lsoa_code, lsoa_name,
               crime_type, outcome, context
        FROM ranked
        WHERE rn = 1
    ),
    -- ASB records (null crime_id) — keep all
    asb_records AS (
        SELECT *
        FROM filtered
        WHERE crime_id IS NULL OR crime_id = ''
    )
    -- Combine
    SELECT * FROM deduped_with_id
    UNION ALL
    SELECT * FROM asb_records
""")

# --- After stats ---
after = db.query("SELECT COUNT(*) as n FROM crime_clean")['n'].iloc[0]
dropped = before - after
print(f"\n✅ Cleaned table created: crime_clean")
print(f"   Before: {before:,}")
print(f"   After:  {after:,}")
print(f"   Dropped: {dropped:,} ({dropped/before*100:.2f}%)")

# Breakdown of what was dropped
print(f"\n   Breakdown:")
print(f"     Null LSOA:          ~{null_lsoa:,}")
print(f"     Welsh LSOA:         ~{welsh:,}")
print(f"     Dedup (outcome):    ~{dropped - null_lsoa - welsh:,}")

# =====================================================================
# STEP 5: Validation
# =====================================================================
print("\n" + "-" * 50)
print("STEP 5: Post-cleaning validation")

# Verify no nulls
null_check = db.query("""
    SELECT 
        SUM(CASE WHEN lsoa_code IS NULL OR lsoa_code = '' THEN 1 ELSE 0 END) as null_lsoa,
        SUM(CASE WHEN lsoa_code LIKE 'W%' THEN 1 ELSE 0 END) as welsh_lsoa,
        SUM(CASE WHEN month IS NULL THEN 1 ELSE 0 END) as null_month,
        SUM(CASE WHEN crime_type IS NULL OR crime_type = '' THEN 1 ELSE 0 END) as null_crime_type
    FROM crime_clean
""")
print(f"  Null LSOA codes:  {null_check['null_lsoa'].iloc[0]}")
print(f"  Welsh LSOA codes: {null_check['welsh_lsoa'].iloc[0]}")
print(f"  Null months:      {null_check['null_month'].iloc[0]}")
print(f"  Null crime types: {null_check['null_crime_type'].iloc[0]}")

# Verify remaining duplicates
remaining_dupes = db.query("""
    SELECT COUNT(*) as n
    FROM (
        SELECT crime_id, month, lsoa_code
        FROM crime_clean
        WHERE crime_id IS NOT NULL AND crime_id != ''
        GROUP BY crime_id, month, lsoa_code
        HAVING COUNT(*) > 1
    )
""")['n'].iloc[0]
print(f"  Remaining duplicate (crime_id, month, lsoa_code): {remaining_dupes}")

# Summary stats
summary = db.query("""
    SELECT
        COUNT(*) as total_crimes,
        COUNT(DISTINCT month) as n_months,
        MIN(month) as first_month,
        MAX(month) as last_month,
        COUNT(DISTINCT lsoa_code) as n_lsoas,
        COUNT(DISTINCT crime_type) as n_crime_types
    FROM crime_clean
""")
print(f"\n📊 Cleaned dataset summary:")
print(f"   Total crimes:  {summary['total_crimes'].iloc[0]:,}")
print(f"   Months:        {summary['n_months'].iloc[0]} ({summary['first_month'].iloc[0]} → {summary['last_month'].iloc[0]})")
print(f"   LSOAs:         {summary['n_lsoas'].iloc[0]:,}")
print(f"   Crime types:   {summary['n_crime_types'].iloc[0]}")

# Crime type distribution after cleaning
types_after = db.query("""
    SELECT crime_type, COUNT(*) as count,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM crime_clean
    GROUP BY crime_type
    ORDER BY count DESC
""")
print(f"\n   Crime type distribution (after cleaning):")
print(types_after.to_string(index=False))

# =====================================================================
# STEP 6: Export to Parquet
# =====================================================================
print("\n" + "-" * 50)
print("STEP 6: Export to Parquet")

parquet_path = PATHS["processed_london"] / "crime_clean.parquet"
parquet_path.parent.mkdir(parents=True, exist_ok=True)

db.execute(f"""
    COPY crime_clean TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

import os
parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
print(f"  ✅ Exported to: {parquet_path}")
print(f"     Size: {parquet_size:.1f} MB")

db.close()
print("\n✅ Data cleaning complete!")
