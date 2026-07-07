# Chicago Cross-City Replication — Results

Replication of the London data-fusion crime-prediction thesis for **Chicago**, using the
exact same pipeline (11 baseline features, RF 200/depth-15/leaf-5/seed-42, last-6-months
test split, inner-join per-layer ablation). Window: **Jan 2023 – Jan 2026** (37 months).
Spatial unit: **census tracts** (814 active, after MIN_CRIMES=36 filter, from Cook County
TIGER 2020). Crime: 776,933 incidents (99.3% geocoded), Chicago Data Portal `ijzp-q8t2`.

## Baseline

| Metric | London (LSOA) | Chicago (tract) |
|---|---|---|
| Test R² | 0.9430 | **0.9019** |
| Test MAE | 4.38 | 4.73 |
| MAE / mean | 24.2% | 20.0% |
| Top feature | rolling_mean_12 (0.30) | rolling_mean_12 (0.73) |

Historical-crime (autoregressive) dominance replicates — even more strongly in Chicago.

## Single-layer fusion (aggregate Δ R²)

| Layer | Source | London | Chicago |
|---|---|---|---|
| Deprivation | IMD → CDC/ATSDR SVI 2022 (tract) | +0.0005 | **+0.0005** |
| Demographics | Census → SVI ACS counts (density, age) | +0.0007 | **−0.0002** |
| Mental health | SAMHI → CDC PLACES MHLTH (tract) | +0.0003 | **+0.0001** |
| Weather | Met Office → NOAA GHCN-Daily O'Hare | +0.0025 | **+0.0022** |
| POIs | OSM/Overpass (10 categories, 33,959 POIs) | +0.0004 | **+0.0004** |
| Transit ridership | TfL taps → CTA 'L' monthly (101 station tracts) | ~0 agg | **+0.0018** |
| Bike-share | Santander → Divvy trips (683 tracts) | +0.0001 | **+0.0002** |

All static layers negligible (≤+0.0005); dynamic weather strongest — same ranking as London.

## Full fusion (contextual + socio-structural, mirror London script 20)

| | London | Chicago |
|---|---|---|
| Δ R² (all layers) | +0.0020 | **+0.0021** |
| Subadditive? | yes (full < context+SS) | yes (full ≈ weather alone) |

## Per-crime-type Δ R² by layer (mirror London Table 5.6)

| Crime type | Base R² | SVI | Weather | Demo | MentHlth | CTA | Divvy | POI | Best |
|---|---|---|---|---|---|---|---|---|---|
| Theft | 0.857 | +.001 | +.001 | +.000 | +.000 | −.016 | −.005 | +.001 | SVI |
| Battery | 0.713 | +.003 | +.010 | +.003 | +.001 | +.005 | +.004 | +.005 | Weather |
| Criminal damage | 0.431 | +.011 | +.006 | +.004 | +.005 | +.005 | +.006 | +.007 | SVI |
| Assault | 0.514 | +.000 | +.005 | +.000 | +.004 | +.004 | +.003 | +.003 | Weather |
| Robbery | 0.288 | −.007 | +.012 | −.003 | −.001 | **+.025** | +.002 | +.001 | **CTA** |
| Motor vehicle theft | 0.441 | +.005 | +.002 | +.005 | −.000 | +.006 | +.005 | +.003 | CTA |
| Burglary | 0.305 | +.001 | +.004 | +.003 | −.005 | −.003 | +.007 | **+.016** | **POI** |
| Narcotics | 0.505 | −.012 | −.030 | +.011 | +.001 | −.007 | +.012 | +.001 | Divvy |
| Weapons violation | 0.183 | −.010 | **+.092** | −.011 | −.017 | +.005 | −.016 | −.008 | **Weather** |
| Deceptive practice | 0.513 | +.006 | +.009 | +.006 | +.000 | +.012 | −.000 | +.014 | POI |

**Key per-type findings (replicate London's heterogeneity):**
- Per-type R² varies 0.18 → 0.86 (London: 0.11 → 0.93). High-volume crime predictable; sparse violent crime not.
- Layers concentrate on distinct crime types: **transit ↔ robbery (+0.025)**, **weather ↔ weapons (+0.092)**, **POI ↔ burglary (+0.016)**, **deprivation ↔ criminal damage (+0.011)**, **bikes ↔ narcotics (+0.012)**.
- These per-type lifts vanish at the aggregate level — same conclusion as London.

## Verdict

The thesis fully replicates in Chicago on independent data: autoregressive dominance,
negligible aggregate value of static area-level layers, weather as the strongest
contextual signal, near-identical subadditive full-fusion lift (+0.0021 vs +0.0020), and
per-type heterogeneity with layers concentrating on specific crime types. Strong evidence
the "predictive ceiling of area-level open data" is a general property of urban crime,
not a London artifact.

## Data substitutions vs London (auth-gated US sources)
- Deprivation: CDC/ATSDR **SVI** (tract, no auth) instead of ADI (block-group, login-gated).
- Demographics: ACS counts pulled from the SVI file (Census API now requires a key).
- All other layers are direct open-data equivalents.
