# Phase 1 Data Sources

> **Purpose:** Concrete, actionable data source catalogue for the 5 priority dataset categories + crime data, covering both London (primary) and Vancouver (cross-city validation).

---

## 1. Crime Data

### London (Primary City)

| Attribute | Detail |
|-----------|--------|
| **Source** | data.police.uk (Home Office) |
| **URL** | https://data.police.uk/data/ |
| **Coverage** | Street-level crime, outcomes, stop & search |
| **Temporal** | Monthly, from Dec 2010 onwards |
| **Spatial** | Anonymised to nearest map point (snap to street); LSOA codes included |
| **Format** | CSV (bulk download by force/month) |
| **API** | Yes -- REST API with endpoints for street-level crimes, crimes at location, stop & search, crime categories, outcomes |
| **API Docs** | https://data.police.uk/docs/ |
| **Forces** | Metropolitan Police Service (MPS) + City of London Police |
| **Licence** | Open Government Licence v3.0 |
| **Update** | Monthly (approx. 6-week lag) |
| **Notes** | Location coordinates are snapped to anonymous map points (not exact addresses). Each record includes: crime type, month, LSOA code/name, latitude/longitude, outcome status. |

**Supplementary London crime source:**

| Attribute | Detail |
|-----------|--------|
| **Source** | London Datastore (Greater London Authority) |
| **URL** | https://data.london.gov.uk/dataset/recorded_crime_summary |
| **Coverage** | MPS recorded crime counts by borough, ward, and LSOA |
| **Temporal** | Monthly, some categories from Jan 2008 |
| **Format** | CSV / Excel |
| **Notes** | Aggregated counts (not individual incidents). Useful for borough/ward-level analysis and historical comparisons. |

### Vancouver (Cross-City Validation)

| Attribute | Detail |
|-----------|--------|
| **Source** | Vancouver Police Department (VPD) GeoDASH |
| **URL** | https://geodash.vpd.ca/ |
| **Coverage** | Crime incidents across Vancouver |
| **Temporal** | 2003 onwards (updated weekly, every Sunday) |
| **Spatial** | Neighbourhood-level; "Offence Against a Person" locations are randomised/offset for privacy (BC FIPPA) |
| **Format** | CSV (download by year + neighbourhood) |
| **API** | City of Vancouver Open Data Portal has API access for some datasets |
| **Portal** | https://opendata.vancouver.ca/ |
| **Licence** | Open Government Licence -- Vancouver |
| **Notes** | Data extracted from PRIME BC Police Records Management System. Crime types are categorised by VPD. No exact time or street name provided for person offences. Also available on Kaggle in historical CSV. |

---

## 2. Socioeconomic Data

### London

| Attribute | Detail |
|-----------|--------|
| **Source** | English Indices of Deprivation (IoD) / Index of Multiple Deprivation (IMD) |
| **Publisher** | Ministry of Housing, Communities & Local Government (MHCLG) |
| **Editions** | **IoD 2019** (current standard) + **IoD 2025** (released Nov 2025, latest) |
| **URL (2019)** | https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019 |
| **URL (2025)** | https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025 |
| **Spatial** | LSOA level (32,844 LSOAs in England; ~4,835 in London) |
| **Domains** | Income, Employment, Education, Health, Crime, Barriers to Housing, Living Environment |
| **Format** | Excel / CSV |
| **London-specific** | London Datastore provides pre-filtered London LSOA + borough data: https://data.london.gov.uk/dataset/indices-of-deprivation |
| **Licence** | Open Government Licence |
| **Notes** | IMD is a relative ranking (not absolute). Each LSOA receives a rank and score for overall deprivation and each of 7 domains. The "Income" and "Employment" domains are the most directly relevant to crime prediction. IoD 2025 uses updated methodology; consider using both editions for temporal comparison. |

### Vancouver

| Attribute | Detail |
|-----------|--------|
| **Source 1** | Statistics Canada -- Census 2021 (Income & Poverty) |
| **URL** | https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm |
| **Coverage** | Employment income, total income, household income, low-income measures (LICO, LIM, MBM) |
| **Spatial** | Census tracts, census subdivisions, dissemination areas |
| **Format** | CSV / TAB / IVT (bulk download) |
| **Download** | Census Profile Downloads: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger.cfm |
| **Source 2** | City of Vancouver -- Census Local Area Profiles |
| **URL** | https://opendata.vancouver.ca/ (search "census local area profiles") |
| **Coverage** | Income, households, families by Vancouver's 22 local planning areas |
| **Format** | XLSX / CSV |
| **Census years** | 2001, 2006, 2011, 2016, 2021 |
| **Notes** | Canada does not have a single "deprivation index" equivalent to the UK IMD. Instead, combine income, employment, and education indicators from Census profiles. The Canadian Index of Multiple Deprivation (CIMD) exists at the dissemination area level from Statistics Canada and may serve as an equivalent. |

---

## 3. Weather / Climate Data

### London

| Attribute | Detail |
|-----------|--------|
| **Source 1** | Met Office MIDAS Open (via CEDA) |
| **URL** | https://catalogue.ceda.ac.uk/uuid/dbd451271eb04662beade68da43546e1 |
| **Coverage** | Land surface observations: temperature, rainfall, wind, humidity, sunshine |
| **Temporal** | 1853 onwards (updated annually, data up to end of prior year) |
| **Resolution** | Hourly + daily |
| **Stations** | Multiple London stations (Heathrow, St James's Park, Kew Gardens, etc.) |
| **Format** | CSV (station-based files) |
| **Access** | Free; requires CEDA account registration |
| **Licence** | Open Government Licence |
| **Source 2** | Met Office Weather DataHub -- Land Observations API |
| **URL** | https://www.metoffice.gov.uk/services/data/met-office-weather-datahub |
| **Coverage** | Recent historical hourly observations from ground stations |
| **Notes** | For this thesis, MIDAS Open is the primary source (long historical archive). The DataHub API is better for recent/near-real-time data. Key variables: daily max/min temperature, total rainfall, mean wind speed, hours of sunshine. |

### Vancouver

| Attribute | Detail |
|-----------|--------|
| **Source** | Environment and Climate Change Canada (ECCC) -- Historical Climate Data |
| **URL** | https://climate.weather.gc.ca/historical_data/search_historic_data_e.html |
| **Coverage** | Temperature, precipitation (rain, snow), wind speed/direction |
| **Temporal** | Varies by station; Vancouver Int'l Airport has data from 1937 onwards |
| **Resolution** | Hourly, daily, monthly |
| **Stations** | ~18 stations within 25 km of Vancouver; primary: Vancouver Int'l Airport (YVR) |
| **Format** | CSV (downloadable per station/year/month) |
| **API** | Bulk download possible via URL parameters |
| **Licence** | Open Government Licence -- Canada |
| **Notes** | Select "Daily" interval for alignment with crime data. Key variables: max/min temperature, total precipitation, snow on ground. Can script bulk downloads by iterating over years. |

---

## 4. Points of Interest (POIs)

### Both Cities (Universal Source)

| Attribute | Detail |
|-----------|--------|
| **Source** | OpenStreetMap (OSM) |
| **Coverage** | Global; community-maintained; bars, restaurants, ATMs, shops, transit stops, parks, schools, nightclubs, etc. |
| **Licence** | Open Data Commons Open Database License (ODbL) |

**Extraction methods (ranked by suitability):**

| Method | Best For | Format | Notes |
|--------|----------|--------|-------|
| **OSMnx** (Python) | Programmatic extraction | GeoDataFrame | `osmnx.features_from_place("London, UK", tags={"amenity": "bar"})` -- cleanest for this project |
| **Overpass API / Turbo** | Interactive queries | GeoJSON / XML | https://overpass-turbo.eu/ -- good for prototyping queries |
| **Geofabrik regional extracts** | Full city dump | PBF / Shapefile | London extract available directly; Vancouver requires clipping from Canada extract |

**Key POI tags for crime prediction:**

| Tag | Category | Relevance |
|-----|----------|-----------|
| `amenity=bar` / `amenity=pub` | Bars / Pubs | Alcohol-related crime hotspots |
| `amenity=nightclub` | Nightclubs | Late-night crime clusters |
| `amenity=atm` / `amenity=bank` | Financial | Robbery targets |
| `shop=*` | Retail / Commercial | Theft and shoplifting zones |
| `amenity=fast_food` / `amenity=restaurant` | Food | Footfall generators |
| `highway=bus_stop` / `railway=station` | Transit | Movement corridors |
| `leisure=park` | Parks / Green space | Both protective and opportunity factor |
| `amenity=school` / `amenity=university` | Education | Youth-related crime |
| `amenity=place_of_worship` | Community | Potential protective factor |
| `tourism=hotel` / `tourism=hostel` | Accommodation | Transient populations |

**Recommended approach:** Use OSMnx in Python to extract all POIs within London boroughs and Vancouver neighbourhoods. Count POI density per spatial unit (LSOA / census tract). Export as GeoJSON for spatial joining with crime data.

---

## 5. Demographics

### London

| Attribute | Detail |
|-----------|--------|
| **Source** | ONS Census 2021 |
| **URL (population density)** | https://www.ons.gov.uk/datasets/TS006 (population density by LSOA) |
| **URL (age)** | https://www.ons.gov.uk/datasets/TS007 (age by single year, LSOA) |
| **URL (bulk)** | https://www.nomisweb.co.uk/ (Nomis table finder for custom queries) |
| **Spatial** | LSOA level (~4,835 LSOAs in London) |
| **Format** | CSV / Excel |
| **Key variables** | Population density (persons/km²), age distribution (single year or 5-year bands), sex, household composition, ethnic group, country of birth |
| **Licence** | Open Government Licence |
| **Notes** | Census 2021 was conducted 21 March 2021. LSOA boundaries align with 2011 boundaries (minor adjustments). ONS recommends 5-year age bands for LSOA-level analysis (single-year counts can be small). Nomis is the most flexible tool for custom geographic/variable selections. |

### Vancouver

| Attribute | Detail |
|-----------|--------|
| **Source** | Statistics Canada -- Census 2021 |
| **URL** | https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm |
| **Bulk download** | https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger.cfm |
| **Boundary files** | https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index-eng.cfm |
| **Spatial** | Census tracts, dissemination areas, census subdivisions |
| **Format** | CSV / TAB / IVT |
| **Key variables** | Population density, age distribution, sex, household size, visible minority status, immigration status |
| **Notes** | Vancouver had 662,248 residents in 2021 (median age 39.6). Highest population density of any Canadian municipality (>5,700/km²). Download Census Profile files and filter by Vancouver geographic codes. Boundary shapefiles available for spatial joining. |

---

## 6. Housing Data

### London

| Attribute | Detail |
|-----------|--------|
| **Source 1** | HM Land Registry -- Price Paid Data |
| **URL** | https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads |
| **Coverage** | All residential property transactions in England & Wales |
| **Temporal** | From 1995 onwards, updated monthly |
| **Format** | CSV (bulk download; ~29M records total) |
| **Key fields** | Transaction price, property type (detached/semi/terraced/flat), date, postcode, new build flag |
| **Licence** | Open Government Licence |
| **Source 2** | ONS -- House Price Statistics for Small Areas |
| **URL** | https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/medianhousepricefornationalandsubnationalgeographiesquarterlyrollingyearhpssadataset09 |
| **Spatial** | LSOA level (median price paid) |
| **Format** | Excel |
| **Source 3** | London Datastore -- LSOA Atlas |
| **URL** | https://data.london.gov.uk/dataset/lsoa-atlas |
| **Coverage** | Composite LSOA-level data including housing, house prices, vacant dwellings |
| **Source 4** | London Datastore -- Dwellings by Build Period & Type |
| **URL** | https://data.london.gov.uk/dataset/property-build-period-lsoa |
| **Spatial** | LSOA and MSOA |
| **Notes** | For vacancy rates, the London Datastore "LSOA Atlas" includes vacant dwellings. Commercial vacancy is available at borough level. The Price Paid Data can be aggregated to LSOA by postcode-to-LSOA lookup (ONS provides this mapping). |

### Vancouver

| Attribute | Detail |
|-----------|--------|
| **Source 1** | CMHC -- Housing Market Information Portal |
| **URL** | https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research |
| **Coverage** | Rental market data (average rents, vacancy rates), housing starts, household characteristics |
| **Spatial** | Neighbourhood level within City of Vancouver; also CMA level |
| **Format** | CSV export / PDF |
| **Key data** | Purpose-built rental vacancy rates (1.6% in 2024, highest in 20 years); average rents by neighbourhood; housing starts |
| **Source 2** | BC Assessment |
| **URL** | https://www.bcassessment.ca/ |
| **Coverage** | Property assessed values for all BC properties |
| **Notes** | Individual property lookups are free. Bulk data may require data sharing agreement. |
| **Source 3** | City of Vancouver Open Data |
| **URL** | https://opendata.vancouver.ca/ (search "property tax report") |
| **Coverage** | Property tax assessment data by neighbourhood |
| **Notes** | CMHC is the most accessible source for rental/vacancy data. Property values can be proxied from BC Assessment or from CMHC's average home price data. Unlike London, there is no single "price paid" transaction registry that is freely downloadable in bulk. |

---

## 7. Mental Health Data

### London

| Attribute | Detail |
|-----------|--------|
| **Source** | Place-based Longitudinal Data Resource (PLDR) |
| **Dataset** | Small Area Mental Health Index (SAMHI) |
| **Version** | v5.00 |
| **URL** | https://pldr.org/dataset/small-area-mental-health-index-samhi-2noyv |
| **Download** | https://pldr.org/download/2noyv/q3n/samhi_21_01_v5.00_2011_2022_LSOA.csv |
| **Coverage** | Composite mental health index for all LSOAs in England |
| **Temporal** | Annual, 2011–2022 (year used: **2022**, latest available) |
| **Spatial** | LSOA 2011 codes (32,844 LSOAs in England; **4,819 matched** to London crime data) |
| **Format** | CSV (wide format: one row per LSOA, columns per year) |
| **File size** | 8.2 MB |
| **Features used** | `samhi_index` (continuous composite score, higher = worse mental health), `samhi_decile` (1=best, 10=worst) |
| **Components** | NHS mental health hospital attendances, antidepressant prescribing, QOF depression prevalence, DWP disability claimants (mental health + learning difficulties) |
| **Licence** | Open Government Licence |
| **Download date** | 28 February 2026 |
| **Nature** | Static (single snapshot per year) |

**Experiment results (script 16):**
- Aggregate Δ R² = +0.0003 (negligible)
- Per-type: theft from person +0.0051, robbery +0.0033, burglary +0.0025, drugs +0.0025
- SAMHI index range in London: -2.257 to 6.542 (mean 0.238)
- London skews toward better mental health (decile 1 has 2,654 LSOAs vs decile 10 has 491)

---

## 8. Education Attainment Data

### London

| Attribute | Detail |
|-----------|--------|
| **Source** | ONS Census 2021 |
| **Dataset** | TS067 — Highest Level of Qualification |
| **URL** | https://www.nomisweb.co.uk/output/census/2021/census2021-ts067.zip |
| **Coverage** | Usual residents aged 16+ by highest qualification level |
| **Temporal** | Census day (21 March 2021) — static snapshot |
| **Spatial** | LSOA 2021 codes (35,672 LSOAs in England; **4,997 matched** to London crime data) |
| **Format** | CSV (one row per LSOA, 8 qualification categories) |
| **File size** | 3.5 MB (zip) |
| **Features used** | 7 percentage features: `pct_no_qual`, `pct_level1`, `pct_level2`, `pct_apprentice`, `pct_level3`, `pct_level4_plus`, `pct_other_qual` |
| **Licence** | Open Government Licence |
| **Download date** | 28 February 2026 |
| **Nature** | Static |

**London education profile:**
- No qualifications: mean 17.4% (range 0.8-48.8%)
- Level 4+ (degree): mean 38.9% (range 9.8-87.2%)
- Apprenticeship: mean 4.3%

**Experiment results (script 17):**
- Aggregate Δ R² = +0.0003 (negligible)
- Per-type: drugs +0.0179 (strongest single-layer lift for drugs), burglary +0.0059, bicycle theft +0.0058, robbery +0.0038

---

### 10. Household Composition Data

| Field | Details |
|-------|---------|
| **Source** | ONS Census 2021 — TS003 (Household Composition) |
| **Dataset** | census2021-ts003 |
| **URL** | https://www.nomisweb.co.uk/output/census/2021/census2021-ts003.zip |
| **Coverage** | All households by composition type |
| **Temporal** | Census day (21 March 2021) — static snapshot |
| **Spatial** | LSOA 2021 codes (35,672 LSOAs; **11,118 matched** to London crime data) |
| **Format** | CSV (one row per LSOA, 26 household categories) |
| **Features used** | 6 percentage features: `pct_one_person`, `pct_one_person_66plus`, `pct_lone_parent_dep`, `pct_married_dep`, `pct_cohabiting`, `pct_other_household` |
| **Licence** | Open Government Licence |
| **Download date** | 28 February 2026 |
| **Nature** | Static |

**London household profile:**
- Single person: mean 29.3% (range 6.1-78.7%)
- Elderly alone (66+): mean 10.8%
- Lone parent with dependents: mean 7.4%
- Other households (HMOs, students): mean 9.8% (range 0.9-71.6%)

**Experiment results (script 18):**
- Aggregate Δ R² = +0.0005
- Per-type: drugs +0.0171, burglary +0.0079, bicycle theft +0.0056, public order +0.0041

---

## Spatial Alignment Summary

A critical consideration for the ETL pipeline is aligning all datasets to a common spatial unit.

| City | Primary Spatial Unit | Approximate Count | Avg. Population |
|------|---------------------|-------------------|-----------------|
| **London** | LSOA (Lower Super Output Area) | ~4,835 in London | ~1,500 residents |
| **Vancouver** | Census Tract (CT) | ~115 in Vancouver | ~2,500-8,000 residents |

**London alignment:**
- Crime data (data.police.uk) includes LSOA codes natively
- IMD, Census, and ONS housing data are all published at LSOA level
- POIs and weather require spatial joining (POI point-in-LSOA; weather station assignment)

**Vancouver alignment:**
- Crime data includes neighbourhood; need to map to census tracts via spatial join
- Census data is natively at CT level
- POIs require point-in-polygon spatial join
- Weather is city-wide (single major station); less spatial variation

---

## Data Availability Assessment

| Category | London | Vancouver | Cross-City Comparable? |
|----------|--------|-----------|----------------------|
| **Crime** | ✅ Excellent (LSOA, monthly, 2010+) | ✅ Good (neighbourhood, weekly, 2003+) | ⚠️ Different crime type taxonomies |
| **Socioeconomic** | ✅ Excellent (IMD at LSOA) | ✅ Good (Census income at CT) | ⚠️ Different indices (IMD vs Census income) |
| **Weather** | ✅ Excellent (MIDAS, 1853+) | ✅ Excellent (ECCC, 1937+) | ✅ Same variables available |
| **POIs** | ✅ Excellent (OSM, comprehensive) | ✅ Good (OSM, slightly less complete) | ✅ Same source, same tags |
| **Demographics** | ✅ Excellent (ONS Census at LSOA) | ✅ Good (StatCan Census at CT) | ✅ Similar variables, different boundaries |
| **Housing** | ✅ Excellent (Land Registry + ONS at LSOA) | ⚠️ Moderate (CMHC neighbourhood; no bulk transaction registry) | ⚠️ Different data structures |

---

## Recommended Next Steps

1. **Download crime data** for both cities first (foundation dataset)
2. **Download IMD 2019 + 2025** for London; Census income profiles for Vancouver
3. **Download MIDAS Open** for London weather stations; ECCC daily for Vancouver YVR
4. **Extract POIs** via OSMnx for both cities
5. **Download Census 2021** demographics for London LSOAs and Vancouver CTs
6. **Download housing data** from Land Registry + ONS (London) and CMHC (Vancouver)
7. **Obtain boundary shapefiles** (LSOA boundaries from ONS; CT boundaries from StatCan) for spatial joins
8. **Build ETL pipeline** to align all sources to common spatial/temporal resolution
