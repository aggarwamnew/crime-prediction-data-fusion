# Crime Prediction with Multi-Source Data Fusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MSc Thesis Research** | Trinity College Dublin | 2025-2026

> *Uncovering What Drives Crime: A Data Fusion Approach to Spatio-Temporal Prediction and Contributing Factor Analysis*

## Overview

This repository contains the complete experiment code for an MSc thesis investigating whether supplementary open data sources (deprivation indices, weather, demographics, points of interest, housing prices, transport ridership, mental health, education, and household composition) can improve monthly neighbourhood-level crime prediction in London beyond what crime history alone provides.

**The short answer: at the aggregate level, no. At the per-crime-type level, dramatically yes.**

### Key Findings

| Finding | Detail |
|---------|--------|
| **Baseline** | Crime history alone achieves R² = 0.943 (Random Forest, 11 features) |
| **Aggregate ceiling** | 13 supplementary data layers produce a combined lift of only +0.002 |
| **Noise floor** | Model error is statistically indistinguishable from the theoretical Poisson limit for 90% of neighbourhoods |
| **Per-type signal** | Station footfall lifts weapons prediction by +0.1371, the largest single improvement in the study |
| **Algorithm-independent** | Random Forest, XGBoost, and LSTM all agree on the ceiling |
| **Double ceiling** | Where transport data is dense (inner London), the baseline is too strong to improve; where it is weaker, transport data does not exist |
| **6 SHAP profiles** | Each crime type has a distinct driver profile (transit-node, deprivation, seasonality, commercial-proximity, pure-persistence, crowd-dynamics) |

### Study Area

London, UK. 5,148 Lower Layer Super Output Areas (LSOAs), 36 months (Feb 2023 to Jan 2026), 3.4 million street-level crime records across 14 offence categories.

## Project Structure

```
crime-prediction-data-fusion/
├── README.md
├── LICENSE
├── EXPERIMENTS.md              # Complete experiment index (34 scripts)
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── README.md               # Data acquisition guide (step-by-step)
│   ├── data_sources.md         # Full data source catalogue with URLs
│   └── data_quality_log.md     # Data quality observations
│
├── notebooks/
│   ├── eda/                    # 33 experiment scripts (01-32 + 23_shap)
│   │   ├── 01_crime_eda.py
│   │   ├── 02_clean_crime_data.py
│   │   ├── 03_baseline_model.py
│   │   ├── ...
│   │   └── 32_concentric_radius.py
│   └── experimental/           # Deep learning experiments
│       ├── data_loader.py
│       ├── 01_lstm_baseline.py
│       └── 02_lstm_full_fusion.py
│
├── scripts/                    # Data download utilities
│   ├── download_santander.py
│   └── download_santander_trips.sh
│
├── src/                        # Reusable Python modules
│   ├── data/                   # Data loading and ingestion
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions
│   ├── utils/                  # Configuration, paths
│   └── visualization/          # Streamlit explorer app
│
└── reports/
    └── figures/                # All generated plots and charts
        ├── baseline/
        ├── eda/
        ├── error_analysis/
        ├── fusion/
        ├── per_type/
        ├── shap/
        ├── transport/
        ├── weather/
        └── xgboost/
```

## Setup

### Prerequisites

- Python 3.10 or later
- DuckDB (installed via pip)
- Approximately 6 GB disk space for raw data downloads

### Installation

```bash
# Clone the repository
git clone https://github.com/aggarwamnew/crime-prediction-data-fusion.git
cd crime-prediction-data-fusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Acquisition

Raw data is not included in this repository (5 GB, licensing constraints). See [`data/README.md`](data/README.md) for step-by-step download instructions for all 13 data sources, with exact URLs, expected file sizes, and directory placement.

All data sources are published under the **Open Government Licence** (UK) or equivalent open licences.

## Running Experiments

Scripts are numbered in execution order. Each script is self-contained and produces console output with metrics plus saved figures.

```bash
# Run from the project root
python notebooks/eda/01_crime_eda.py          # EDA and data overview
python notebooks/eda/02_clean_crime_data.py   # Data cleaning pipeline
python notebooks/eda/03_baseline_model.py     # Baseline RF model (R² = 0.943)
python notebooks/eda/04_fused_model.py        # IMD fusion experiment
# ... continue through script 32
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the complete experiment index with descriptions and key findings.

### Streamlit Explorer

An interactive explorer for browsing LSOA-level predictions and feature importance:

```bash
# Precompute data (run after baseline model)
python src/visualization/precompute_data.py

# Launch the explorer
streamlit run src/visualization/app.py
```

## Results Summary

### Aggregate Fusion (All Crime Types Combined)

| Data Layer | Type | Delta R² | Features |
|------------|------|----------|----------|
| Crime history (baseline) | - | R² = 0.9430 | 11 |
| + Weather | Dynamic | +0.0025 | 5 |
| + IMD 2025 | Static | +0.0008 | 9 |
| + Demographics | Static | +0.0007 | 7 |
| + Housing | Static | +0.0006 | 3 |
| + Household comp. | Static | +0.0005 | 2 |
| + POIs | Static | +0.0004 | 11 |
| + Mental health | Static | +0.0003 | 2 |
| + Education | Static | +0.0003 | 3 |
| + Temporal activity | Dynamic | +0.0001 | 2 |
| **Full fusion (51 features)** | | **+0.0020** | **51** |

### Per-Crime-Type Highlights

| Crime Type | Best Data Layer | Delta R² | Mechanism |
|------------|----------------|----------|-----------|
| Weapons | Station footfall | **+0.1371** | Transport hub convergence |
| Drugs | IMD (deprivation) | +0.020 | Socioeconomic clustering |
| Bicycle theft | Weather | +0.013 | Warm weather = more cyclists |
| Theft from person | Temporal activity | +0.014 | Holiday crowds = more targets |
| Burglary | Demographics | +0.008 | Age structure and density |
| Shoplifting | (none) | ~0 | Pure historical persistence |

## Data Sources

All 13 primary data sources plus 3 transport sources are documented in [`data/data_sources.md`](data/data_sources.md). Summary:

| Tier | Sources | Nature |
|------|---------|--------|
| **Baseline** | data.police.uk crime records | Dynamic |
| **Contextual** | IMD, Weather, Demographics, POIs, Housing | Mixed |
| **Temporal** | School holidays, Bank holidays | Dynamic |
| **Socio-Structural** | SAMHI mental health, Education (TS067), Household (TS003) | Static |
| **Transport** | PTAL, TfL Station Footfall, Santander Cycles | Mixed |

**Note on Vancouver:** The data catalogue includes Vancouver sources for planned cross-city validation. This was scoped but not implemented in the current study. London remains the sole study area.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{aggarwal2026crime,
  title     = {Uncovering What Drives Crime: A Data Fusion Approach to
               Spatio-Temporal Prediction and Contributing Factor Analysis},
  author    = {Aggarwal, Mohit},
  school    = {Trinity College Dublin},
  year      = {2026},
  type      = {MSc Thesis},
  program   = {Computer Science (Intelligent Systems)}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Supervisor: Trinity College Dublin, School of Computer Science and Statistics
- Crime data: Home Office via [data.police.uk](https://data.police.uk/) (Open Government Licence)
- Transport data: Transport for London Open Data
- Deprivation indices: Ministry of Housing, Communities and Local Government
- Census data: Office for National Statistics
- Points of interest: OpenStreetMap contributors (ODbL)
- Mental health index: Place-based Longitudinal Data Resource (PLDR)
