# Cross-City Feasibility & Replication Study: London, Chicago, Vancouver

> Empirical companion to the thesis. Tests whether the London findings (autoregressive
> dominance, negligible aggregate value of static area-level layers, per-type
> heterogeneity, predictive ceiling) replicate in two other cities using the SAME pipeline
> (11 features, RF 200/depth-15/leaf-5/seed-42, last-6-months test split, inner-join
> per-layer ablation). Window: 36-37 months ending late 2025/early 2026.
> Results are real (data downloaded and models run), not projected. Date: 2026-06-29.

## 1. Headline verdict

The thesis **replicates in all three cities on independent data across two continents**.
Historical crime dominates everywhere (R² 0.90-0.94 from lags alone); static area-level
layers add negligible aggregate lift; layers matter only per-crime-type. This is strong
evidence the "predictive ceiling of area-level open data" is a **general property of urban
crime**, not a London artifact.

| City | Unit | Active units | Crime | Baseline Test R² | Top feature |
|---|---|---|---|---|---|
| London | LSOA | 5,148 | all 14 types | 0.943 | rolling_mean_12 |
| Chicago | census tract | 814 | all 31 types (point-level) | 0.902 | rolling_mean_12 (0.73) |
| Vancouver | dissemination area | 461 | property only* | 0.925 | rolling_mean_3 |

\* Vancouver: violent/person offences are coordinate-suppressed in VPD open data, so only
property crime is geocodable at the DA level.

**Vancouver all-crime (24 neighbourhoods, incl. violent):** when violent crime IS included
at its available resolution (the 24-neighbourhood level, where person-offences carry a
neighbourhood even though point coordinates are suppressed), the model reaches **Test
R² = 0.971** — autoregressive dominance holds with violent crime in the mix, just at a
coarser unit (top features lag_12, lag_3, rolling_mean_12).

## 2. Single-layer fusion — aggregate Δ R² (the core finding)

| Layer | London | Chicago | Vancouver |
|---|---|---|---|
| Deprivation (IMD / SVI / CIMD) | +0.0005 | +0.0005 | −0.0001 |
| Demographics | +0.0007 | −0.0002 | n/r |
| Mental health (SAMHI / PLACES / —) | +0.0003 | +0.0001 | **no source** |
| Weather | +0.0025 | +0.0022 | −0.0001 |
| POIs | +0.0004 | +0.0004 | −0.0000 |
| Education | +0.0003 | +0.0005 | +0.0005 |
| Household | +0.0005 | +0.0003 | −0.0006 |
| Housing (assessed tertiles) | +0.0006 | +0.0002 | −0.0004 |
| Temporal (school terms + holidays) | +0.0001 | +0.0008 | −0.0003 |
| Transit ridership | ~0 agg | +0.0018 | annual-only (open); monthly gated |
| Bike-share | +0.0001 | +0.0002 (Divvy) | +0.0002 (Mobi, scripted via Drive+CityBikes) |

**Update 2026-07-13: FULL LAYER PARITY REACHED.** Every closable layer is now closed in
both replication cities (Chicago: all ten London constructs; Vancouver: everything not
gated). Vancouver's all-layer full fusion (30 supplementary features): **Δ R² = +0.0000,
CI [−0.0027, +0.0019]**. Only the gated tier remains (violent-crime coordinates, small-area
mental health, monthly ridership), which is the study's open-vs-gated finding.

**Every static deprivation/demographic layer is negligible in every city.** Weather is the
strongest contextual signal in London and Chicago but **negligible in Vancouver** — an
interpretable result: Vancouver's mild oceanic climate has little seasonal temperature
range, and its usable crime is property-only (less weather-sensitive than violent crime).

## 3. Full fusion (contextual + socio-structural)

| | London | Chicago |
|---|---|---|
| Δ R² (all layers) | +0.0020 | +0.0021 |
| Subadditive | yes | yes (≈ weather alone) |

The headline full-fusion lift is **near-identical** (London +0.0020, Chicago +0.0021) and
subadditive in both — adding static layers on top of weather + history adds nothing.

## 4. Per-crime-type heterogeneity (Chicago, mirrors London Table 5.6)

Per-type R² ranges 0.18-0.86 (London 0.11-0.93). Layers concentrate on distinct types:

| Crime type | Best layer | Δ R² |
|---|---|---|
| Weapons violation | Weather | +0.092 |
| Robbery | Transit (CTA) | +0.025 |
| Burglary | POIs | +0.016 |
| Deceptive practice | POIs / transit | +0.014 |
| Narcotics | Bike-share / demographics | +0.012 |
| Criminal damage | Deprivation (SVI) | +0.011 |

Direct London parallels: transit↔convergence crime (London weapons↔bus-stop POIs +0.049;
Chicago robbery↔CTA +0.025), POIs↔property crime (London burglary↔restaurants; Chicago
burglary↔POIs +0.016). Every per-type lift vanishes at the aggregate level — same as London.
(Note: London's originally reported weapons↔station-taps +0.137 was later found to be a
target-leakage artefact of the per-type join in scripts 30/31 and corrected to ≈0; see
`notebooks/eda/35_transport_join_audit.py`. The POI-based weapons lift +0.049 is
bootstrap-robust and stands.)

## 5. Data availability & access friction (verified 2026-06)

| Layer | London | Chicago | Vancouver |
|---|---|---|---|
| Crime | data.police.uk (easy) | Data Portal `ijzp-q8t2`, point-level, 1-click API (easiest) | VPD GeoDASH zip; **violent crime coord-suppressed**; City portal dropped it |
| Boundaries | ONS LSOA | TIGER tracts (easy) | StatCan DA national file (197 MB), filter by province |
| Deprivation | IMD | SVI (no auth); ADI needs login | CIMD (easy, DA-level) |
| Demographics | Census | ACS (now needs API key) → via SVI | Census 2021 (easy) |
| Mental health | SAMHI | **CDC PLACES (better than London)** | **none at DA level (gap)** |
| Weather | Met Office | NOAA GHCN-Daily (no token via direct CSV) | ECCC YVR (easy) |
| POIs | OSM | OSM (Overpass overloaded but works) | OSM |
| Transit | TfL taps | CTA monthly 2001+ (better than London) | TransLink **annual-only** (gap) |
| Bike-share | Santander | Divvy S3 (easy) | Mobi **Google-Drive only** (friction) |

**Chicago is the easiest city and the strongest validation** (point-level crime, a
literature anchor in Chattopadhyay et al. 2022, and two layers *better* than London's:
CDC PLACES mental health and CTA monthly ridership).

**Vancouver: the data is not missing — it is GATED.** A broader search (VPD GeoDASH FAQ,
PopData BC, TransLink TSPR, BCCDC) confirms VPD and BC hold rich, granular data; the OPEN
releases are deliberately coarsened, and the granular versions sit behind research
agreements:

| Layer | Open release (coarsened) | Granular version (gated) |
|---|---|---|
| Crime | 11 categories; violent crime offset to intersection, no time/street; SkyTrain excluded; BC FIPPA | Full PRIME BC RMS detail via VPD research data-sharing agreement |
| Mental health | none open at DA level (BCIMD only at CHSA, 195 areas) | Person-level mental-health records, 1985+, via **PopData BC** access request |
| Transit ridership | per-station **annual** average-daily (TSPR, downloadable CSV) | monthly station-level **Compass tap** data via TransLink data request |
| Bike-share | Mobi trip files (Google-Drive links, manual) | — |

This is the **Vancouver analogue of London's own ceiling**: London's individual-level and
violent detail need ONS Approved-Researcher status; Vancouver's need VPD/PopData BC/TransLink
agreements. Same structural barrier, different jurisdiction — it strengthens the thesis's
"open data has a ceiling; the next tier is gated" argument across both countries.

Vancouver's value is therefore twofold: (a) a **stress test** — the ceiling finding survives
even in a sparser, property-only, mild-climate environment (DA R²=0.925, all-crime
neighbourhood R²=0.971, deprivation and weather both ≈0); and (b) a **second jurisdiction
demonstrating the open-vs-gated data ceiling**, reinforcing the PhD-pathway argument.

## 6. Recommendation

- **Primary cross-city validation: Chicago.** Clean full replication, literature anchor,
  and it strengthens the data-fusion story.
- **Secondary stress test: Vancouver.** Frame its gaps as a contribution — the predictive
  ceiling holds even where data is sparser and the crime mix is property-dominated.

## 7. Reproducibility

Pipelines: `notebooks/chicago/` and `notebooks/vancouver/` (download + per-layer fusion +
full fusion + per-type). All raw data is gitignored and re-downloadable via the scripts.
Substitutions for auth-gated US sources (SVI for ADI, ACS-via-SVI counts) are documented in
`notebooks/chicago/RESULTS.md`.
