# Data Quality Log

> **Purpose:** Document all data issues found, decisions made about cleaning/dropping, and justifications. This feeds directly into the Methodology chapter of the thesis.

---

## 1. Crime Data (data.police.uk)

**Source:** `data.police.uk/data/` custom download  
**Date range:** Feb 2023 → Jan 2026 (36 months)  
**Forces:** Metropolitan Police Service + City of London Police  
**Raw rows:** 3,442,413  

### Issue 1.1: Null LSOA codes
- **Count:** 24,605 rows (0.71%)
- **Cause:** data.police.uk cannot geolocate some crimes (e.g., online fraud, crimes with suppressed locations for victim safety)
- **Decision:** **DROP** — these rows cannot be assigned to a spatial unit, so they're unusable for spatial prediction
- **Impact:** Minimal (0.71%)

### Issue 1.2: Welsh LSOA codes
- **Count:** 215 rows with `W%` LSOA codes
- **Cause:** Met Police jurisdiction occasionally overlaps Welsh border areas
- **Decision:** **DROP** — outside our London study area, negligible count
- **Impact:** Negligible (0.006%)

### Issue 1.3: Unmatched LSOA codes (2011 vs 2021 boundaries)
- **Count:** 224 unique LSOA codes in crime data not found in 2021 LSOA boundaries
- **Cause:** ONS redesigned LSOA boundaries for the 2021 census. Some 2011 LSOAs were merged, split, or recoded. data.police.uk may still reference old codes for earlier months.
- **Decision:** **PENDING** — investigate in EDA. Options: (a) use ONS LSOA 2011→2021 lookup table to remap, or (b) drop if volume is small
- **Impact:** TBD — need to quantify how many crimes use these codes

### Issue 1.4: Date range starts Feb 2023 (not Jan 2023)
- **Cause:** data.police.uk custom download dropdown only goes back to Feb 2023
- **Decision:** **ACCEPT** — 36 months covers 3 full seasonal cycles; archive data available if needed
- **Impact:** None — still sufficient temporal coverage

### Issue 1.5: Duplicate crime IDs (found in EDA)
- **Count:** 39,662 unique crime IDs appear more than once
- **Cause:** data.police.uk is known to assign the same crime ID to records that appear under both Met Police and City of London, or when a crime spans LSOA boundaries. Also, some crimes are re-reported with updated outcomes.
- **Decision:** **INVESTIGATE** — need to determine if these are true duplicates or cross-boundary records. If cross-boundary, keep one per unique (crime_id, month) pair.
- **Impact:** ~1.2% of total records

### Issue 1.6: Extreme right-skew in LSOA crime distribution (found in EDA)
- **Stats:** Mean = 294 crimes/LSOA, Median = 4, Max = 41,390 (Westminster 013G)
- **Cause:** ~9,000 LSOAs outside core London have only 1-10 crimes (peripheral Met Police jurisdiction). Westminster/City of London LSOAs are major hotspots.
- **Decision:** **FILTER** — for modelling, consider filtering to LSOAs with ≥12 crimes (at least 1/month average) to avoid extreme sparsity. Document threshold in methodology.
- **Impact:** Will significantly reduce LSOA count but improve model signal

### Issue 1.7: Seasonal pattern confirmed (found in EDA)
- **Finding:** Clear seasonality — summer peaks (June-July), winter troughs (Feb-March)
- **Mean monthly crimes:** 95,623
- **Decision:** **ACCEPT** — validates need for seasonal features and 36-month coverage

---

## 2. LSOA Boundaries (ONS Open Geography Portal)

**Source:** ONS ArcGIS Feature Server (`LSOA_2021_EW_BSC_V4_RUC`)  
**Version:** December 2021, Super Generalised Clipped (BSC V4)  
**Total downloaded:** 35,672 (all England & Wales)  
**Filtered to crime data:** 11,279 LSOAs  
**File:** `data/raw/london/boundaries/lsoa_2021_london.geojson` (7.4 MB)  

### Issue 2.1: More LSOAs than expected for London
- **Expected:** ~4,835 LSOAs in Greater London
- **Got:** 11,279 matching crime data (11,632 including unmatched boundary codes)
- **Cause:** Met Police jurisdiction extends beyond Greater London administrative boundary (covers some areas in Essex, Hertfordshire, Surrey, etc.)
- **Decision:** **ACCEPT** — use all LSOAs that have crime data, as this reflects the actual Met Police coverage area
- **Impact:** More spatial units = richer model, but note this in the thesis

### Issue 2.2: Generalised vs Full Resolution boundaries
- **Decision:** Using **BSC (Super Generalised Clipped)** rather than BFC (Full Resolution Clipped)
- **Rationale:** Generalised boundaries are sufficient for aggregation/joining. Full resolution would add ~100x file size with no analytical benefit for count-based prediction. Generalised maps render faster in Kepler.gl.

---

## 3. Cleaning Pipeline

| Step | Action | Rows affected | Status | Justification |
|------|--------|--------------|--------|---------------|
| 3.1 | Drop null LSOA codes | 24,605 (0.71%) | ✅ Done | Cannot spatially locate |
| 3.2 | Drop Welsh LSOA codes | 215 (0.006%) | ✅ Done | Outside study area |
| 3.3 | Remap old LSOA codes | Deferred | ⏭️ Skipped for now | Will revisit when joining to IMD/census data |
| 3.4 | Deduplicate crime IDs | 42,520 rows (~1.2%) | ✅ Done | Kept most informative outcome per (crime_id, month, lsoa_code) |
| 3.5 | Check for missing months | 0 gaps | ✅ Confirmed | All 36 months present |
| 3.6 | Filter sparse LSOAs | Deferred | ⏭️ Deferred | Will apply at modelling stage (Step 4) |

---

## 4. Duplicate Crime ID Investigation (detailed)

**Finding:** 39,662 unique crime IDs appear more than once

| Check | Result |
|-------|--------|
| Cross-force duplicates | **0** — all dupes are within the same force |
| Different-month duplicates | **0** — all dupes occur in the same month |
| Different-LSOA duplicates | **986** — crime recorded at LSOA boundary |
| Same-LSOA duplicates | **38,676** — differ **only** in outcome column |
| Null crime IDs | **692,536** — all are Anti-social behaviour (ASB never has crime IDs on data.police.uk) |

**Root cause:** data.police.uk publishes multiple rows when a crime's outcome is updated. The original row (often "Status update unavailable") persists alongside the updated outcome.

**Decision:** Keep one row per `(crime_id, month, lsoa_code)`, preferring the most informative outcome. Cross-LSOA duplicates (986 crime IDs) are kept as separate spatial events.

---

## 5. Cleaning Execution Results

**Script:** `notebooks/eda/02_clean_crime_data.py`  
**Date:** 2026-02-27

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Total rows | 3,442,413 | 3,375,073 | -67,340 (1.96%) |
| Null LSOA | 24,605 | 0 | -24,605 |
| Welsh LSOA | 215 | 0 | -215 |
| Duplicate (id, month, lsoa) | 39,662+ | 0 | -42,520 |
| Months | 36 | 36 | 0 |
| LSOAs | 11,632 | 11,341 | -291 |
| Crime types | 14 | 14 | 0 |

**Output files:**
- DuckDB table: `crime_clean` in `data/processed/london/thesis.duckdb`
- Parquet export: `data/processed/london/crime_clean.parquet` (141.2 MB)

---

## 6. Post-Cleaning Key Statistics

| Metric | Value |
|--------|-------|
| Total crimes | 3,375,073 |
| Date range | Feb 2023 → Jan 2026 (36 months) |
| Crime types | 14 |
| Unique LSOAs | 11,341 |
| Mean monthly crimes | ~93,752 |
| Top crime type | Violence & sexual offences (22.8%) |
| ASB share | 20.5% (692,440 records, no crime_id by design) |
| Top outcome | No suspect identified (~51%) |

---

*Updated: 2026-02-27 (post-cleaning)*
