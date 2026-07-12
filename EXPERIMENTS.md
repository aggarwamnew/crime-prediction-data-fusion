# Experiment Index

Complete index of experiments in this study. London scripts are in `notebooks/eda/` (01-35) and `notebooks/experimental/` (DL-01, DL-02); the cross-city replications are in `notebooks/chicago/` and `notebooks/vancouver/`.

## Execution Order

Scripts should be run sequentially. Scripts 01-02 prepare the data; script 03 establishes the baseline; scripts 04-32 run ablation experiments.

## Experiment Table

| # | Script | Description | Key Finding | In Model? |
|---|--------|-------------|-------------|-----------|
| 01 | `01_crime_eda.py` | Data ingestion and schema validation | 3.4M records, 14 crime types | - |
| 02 | `02_clean_crime_data.py` | Data cleaning pipeline | 67K duplicates removed (1.96%) | - |
| 03 | `03_baseline_model.py` | Baseline Random Forest model | R² = 0.943, MAE = 4.38, MASE = 0.81 | Yes |
| 04 | `04_fused_model.py` | IMD fusion (aggregate) | Delta R² = +0.0005 | Yes |
| 05 | `05_per_type_experiment.py` | Per-type IMD fusion | Drugs +0.020, Burglary +0.006 | Yes |
| 06 | `06_xgboost_comparison.py` | XGBoost comparison | Algorithm-independent finding confirmed | Yes |
| 07 | `07_weather_fusion.py` | Weather fusion | Delta R² = +0.0025 (5x larger than IMD) | Yes |
| 08 | `08_per_type_full.py` | Per-type IMD + Weather | Drugs: IMD wins; Bicycle: Weather wins | Yes |
| 09 | `09_imd_robustness.py` | IMD robustness (2019 vs 2025) | Consistent: +0.0007 vs +0.0008 | Yes |
| 10 | `10_demographics_fusion.py` | Demographics fusion | Delta R² = +0.0007 | Yes |
| 11 | `11_per_type_all_layers.py` | Per-type all layers comparison | Demographics best for 7/14 types | Yes |
| 12 | `12_poi_extraction.py` | POI extraction (Overpass API) | 101,424 POIs across 10 categories | - |
| 13 | `13_poi_fusion.py` | POI fusion | Weapons +0.049 (bus stop density) | Yes |
| 14 | `14_housing_fusion.py` | Housing price fusion | Delta R² = +0.0006 | Yes |
| 15 | `15_school_holiday_fusion.py` | Temporal activity fusion | Zero aggregate; Theft-person +0.014 | Yes |
| 16 | `16_mental_health_fusion.py` | SAMHI mental health fusion | Delta R² = +0.0003 | Yes |
| 17 | `17_education_fusion.py` | Education attainment fusion | Drugs +0.018 (strongest for drugs) | Yes |
| 18 | `18_household_fusion.py` | Household composition fusion | Drugs +0.017 | Yes |
| 19 | `19_socio_structural_combined.py` | Combined socio-structural | Delta R² = +0.0012 (subadditive) | Yes |
| 20 | `20_full_fusion.py` | Full fusion (aggregate) | 51 features, R² = 0.9374 | Yes |
| 21 | `21_full_fusion_per_type.py` | Full fusion (per-type) | Drugs +0.027 (best overall per-type lift) | Yes |
| 22 | `22_full_fusion_chart.py` | Full fusion visualisation | Per-type Delta R² bar chart | - |
| 23 | `23_per_layer_chart.py` | Per-layer comparison chart | Comparative Delta R² visualisation | - |
| 23b | `23_shap_with_ss.py` | SHAP analysis with SS features | 6 distinct driver profiles identified | Yes |
| 24 | `24_daylight_fusion.py` | Daylight hours fusion | +0.0003 (redundant with month encoding) | Excluded |
| 25 | `25_benefits_fusion.py` | DWP claimant count fusion | -0.0000 (rounded data destroys signal) | Excluded |
| 26 | `26_noise_floor_analysis.py` | Noise floor analysis | Model MAE approximately equals Poisson floor | - |
| 27 | `27_r2_by_crime_level.py` | R² stratified by crime level | R² varies 0.11-0.93; MASE stable at 0.81 | - |
| 28 | `28_error_analysis_map.py` | Spatial error analysis | Error is volume-driven, not geographic | - |
| 29 | `29_ptal_fusion.py` | PTAL transport fusion | Delta R² = +0.0003 (static accessibility) | Yes |
| 30 | `30_station_footfall_fusion.py` | Station footfall fusion | Per-type join had target leakage; corrected by script 35 (weapons ~0) | Yes |
| 31 | `31_santander_fusion.py` | Santander Cycles fusion | Delta R² = +0.0001; per-type see script 35 | Yes |
| 32 | `32_concentric_radius.py` | Concentric radius diagnostic | Double ceiling effect confirmed | - |
| 33 | `33_bootstrap_ci.py` | Cluster-bootstrap 95% CIs for headline results | Aggregate lifts tiny but real; per-type robust | - |
| 34 | `34_static_only_model.py` | Supplementary-only models (no crime history) | IMD alone R2 = 0.905 vs 0.935 with history | - |
| 35 | `35_transport_join_audit.py` | Leaky vs corrected transport join audit | Weapons +0.1371 -> +0.003 (n.s.); only drugs keeps a small robust lift | - |
| 36 | `36_uncertainty_decomposition.py` | Pre-fusion epistemic diagnostics vs realised gains | Rule-out valid, rule-in fails (rho ~ 0): gap size does not reveal gap source | - |
| DL-01 | `experimental/01_lstm_baseline.py` | LSTM baseline (11 features) | R² = 0.9075 (RF wins by +0.035) | - |
| DL-02 | `experimental/02_lstm_full_fusion.py` | LSTM full fusion (51 features) | R² = 0.9040 (fusion actually hurts LSTM) | - |

## Chicago Replication (`notebooks/chicago/`)

Same modelling logic as London (11 features, RF 200/depth-15/leaf-5/seed-42, last-6-months split), on 814 census tracts. Shared helpers: `_fusion.py` (fusion harness), `layers.py` (layer loaders). Detailed results in [`notebooks/chicago/RESULTS.md`](notebooks/chicago/RESULTS.md).

| # | Script | Description | Key Finding |
|---|--------|-------------|-------------|
| 01 | `01_download_crime.py` | Crime pull (Chicago Data Portal, Socrata) | 776,933 incidents, 99.3% geocoded |
| 02 | `02_download_boundaries.py` | Census tracts (TIGER 2020, Cook County) | 814 active tracts |
| 03 | `03_baseline_model.py` | Baseline Random Forest | R² = 0.902 (rolling_mean_12 = 0.73) |
| 04 | `04_svi_fusion.py` | Deprivation (CDC/ATSDR SVI) | Delta R² = +0.0005 |
| 05 | `05_weather_fusion.py` | Weather (NOAA GHCN-Daily, O'Hare) | Delta R² = +0.0022 |
| 06 | `06_poi_fusion.py` | POIs (OpenStreetMap / Overpass) | Delta R² = +0.0004 |
| 07 | `07_mental_health_fusion.py` | Mental health (CDC PLACES) | Delta R² = +0.0001 |
| 08 | `08_demographics_fusion.py` | Demographics (ACS via SVI counts) | Delta R² = -0.0002 |
| 09 | `09_cta_fusion.py` | Transit (CTA 'L' station ridership) | +0.0018 (station tracts) |
| 10 | `10_divvy_fusion.py` | Bike-share (Divvy trips) | Delta R² = +0.0002 |
| 11 | `11_full_fusion.py` | Full fusion (contextual + SS) | Delta R² = +0.0021 (subadditive) |
| 12 | `12_per_type.py` | Per-crime-type ablation | Weapons+Weather +0.092; Robbery+Transit +0.025 |

## Vancouver Replication (`notebooks/vancouver/`)

Same logic on Census dissemination areas. Property crime only at DA level (violent-offence coordinates suppressed under BC FIPPA); a 24-neighbourhood run includes all crime. Shared helper: `_fusion.py`.

| # | Script | Description | Key Finding |
|---|--------|-------------|-------------|
| 01 | `01_download_crime.py` | Crime pull (VPD GeoDASH) | 919K records; violent crime coordinate-suppressed |
| 02 | `02_download_boundaries.py` | Dissemination areas (StatCan 2021) | filtered to City of Vancouver |
| 03 | `03_baseline_model.py` | Baseline Random Forest (property crime) | R² = 0.925 (461 DAs) |
| 04 | `04_cimd_fusion.py` | Deprivation (CIMD) | Delta R² = -0.0001 |
| 05 | `05_weather_fusion.py` | Weather (ECCC, YVR) | Delta R² = -0.0001 (mild climate) |
| 06 | `06_poi_fusion.py` | POIs (OpenStreetMap / Overpass) | Delta R² = -0.0000 |
| 07 | `07_neighbourhood_allcrime.py` | All-crime at 24 neighbourhoods (incl. violent) | R² = 0.971 |
| 08 | `08_full_fusion.py` | Full fusion (contextual layers) | static layers negligible |

## Cross-City Comparison

| Script | Description |
|--------|-------------|
| `notebooks/cross_city_charts.py` | Generates the London/Chicago/Vancouver comparison figures in `reports/figures/cross_city/` |

## Notes

- **"In Model?"** indicates whether the data layer's features are included in the final model configuration.
- Scripts marked "Excluded" were tested but deliberately omitted from the final model due to redundancy or data quality issues.
- Scripts marked "-" are analytical/diagnostic tools that do not contribute features to the model.
- All scripts use a temporal train/test split (Feb 2024-Jul 2025 / Aug 2025-Jan 2026) to prevent data leakage.
