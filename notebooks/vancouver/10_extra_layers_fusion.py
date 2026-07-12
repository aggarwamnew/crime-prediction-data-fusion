"""10_extra_layers_fusion.py — Close Vancouver's available layer gaps (Session 21).

  1. Population density: DA population from the CIMD file / land area from the StatCan
     boundary file (LANDAREA, km2). Zero new downloads. ~ London TS006.
  2. Temporal activity: BC statutory holiday fraction per month (`holidays` lib,
     Canada/BC). ~ London bank holidays. (VSB school terms remain a manual gap.)

Remaining gaps that stay open: age structure + education + household (Census 2021 DA
profile bulk download), housing (City property tax report parcel->DA join), Mobi
bike-share (manual Google Drive), and the gated items (violent crime, mental health,
monthly ridership).
"""
import sys
import zipfile
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import holidays as pyhol
import pandas as pd

from _fusion import run_fusion, load_panel_with_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"

print("=" * 70)
print("VANCOUVER EXTRA LAYERS (population density, BC holidays)")
print("=" * 70)

# 1. density = CIMD DA population / boundary LANDAREA
z = zipfile.ZipFile(RAW / "cimd/bc_scores_quintiles.zip")
name = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
pop_col = [c for c in cimd.columns if "Population" in c][0]
pop = cimd[[da_col, pop_col]].rename(columns={da_col: "DAUID", pop_col: "da_pop"})
pop["DAUID"] = pop["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
pop["da_pop"] = pd.to_numeric(pop["da_pop"], errors="coerce")

bounds = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "LANDAREA"]]
bounds["LANDAREA"] = pd.to_numeric(bounds["LANDAREA"], errors="coerce")
dens = pop.merge(bounds, on="DAUID", how="inner")
dens = dens[(dens["da_pop"] > 0) & (dens["LANDAREA"] > 0)]
dens["pop_density"] = dens["da_pop"] / dens["LANDAREA"]
print(f"   density built for {len(dens):,} DAs")
run_fusion("Density (CIMD pop / area)", ["pop_density", "da_pop"],
           static_by_da=dens[["DAUID", "pop_density", "da_pop"]])

# 2. BC statutory holiday fraction per month
bc = pyhol.country_holidays("CA", subdiv="BC", years=range(2023, 2026))
rows = []
for ms in pd.date_range("2023-01-01", "2025-12-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    nh = sum(1 for d in pd.date_range(ms, ms + pd.offsets.MonthEnd(0)) if d.date() in bc)
    rows.append({"month": f"{ms.year}-{ms.month:02d}", "pct_holiday_days": nh / dim})
hol = pd.DataFrame(rows)
run_fusion("Temporal (BC holidays)", ["pct_holiday_days"], month_only=hol)
