"""15_extra_layers_fusion.py — Close Chicago's layer gaps vs London (Session 21).

Adds the London tiers Chicago lacked, using zero-download sources where possible:
  1. Education (SVI EP_NOHSDP: % adults without high-school diploma)  ~ London TS067
  2. Household composition (SVI EP_SNGPNT: % single-parent households) ~ London TS003
  3. Housing cost burden (SVI EP_HBURD: % cost-burdened households)    ~ London housing tier
  4. Temporal activity (US federal + Illinois holiday fraction per month, `holidays` lib)
     ~ London bank holidays. (CPS school-term calendar remains a manual gap.)
"""
import sys
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holidays as pyhol
import pandas as pd

from _fusion import run_fusion, load_panel_with_features

ROOT = Path(__file__).resolve().parents[2]

svi = pd.read_csv(ROOT / "data/raw/chicago/svi/svi_2022_illinois.csv", dtype={"FIPS": str})
svi = svi.rename(columns={"FIPS": "GEOID"})
for c in ["EP_NOHSDP", "EP_SNGPNT", "EP_HBURD"]:
    svi[c] = pd.to_numeric(svi[c], errors="coerce")
    svi.loc[svi[c] < 0, c] = pd.NA          # SVI missing code -999

print("=" * 70)
print("CHICAGO EXTRA LAYERS (education, household, housing burden, holidays)")
print("=" * 70)

run_fusion("Education (EP_NOHSDP)", ["EP_NOHSDP"],
           static_by_geoid=svi[["GEOID", "EP_NOHSDP"]].dropna())
run_fusion("Household (EP_SNGPNT)", ["EP_SNGPNT"],
           static_by_geoid=svi[["GEOID", "EP_SNGPNT"]].dropna())
run_fusion("Housing burden (EP_HBURD)", ["EP_HBURD"],
           static_by_geoid=svi[["GEOID", "EP_HBURD"]].dropna())

# temporal activity: fraction of days per month that are US federal + IL holidays
il = pyhol.country_holidays("US", subdiv="IL", years=range(2023, 2027))
rows = []
for ms in pd.date_range("2023-01-01", "2026-01-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    nh = sum(1 for d in pd.date_range(ms, ms + pd.offsets.MonthEnd(0)) if d.date() in il)
    rows.append({"month": f"{ms.year}-{ms.month:02d}", "pct_holiday_days": nh / dim})
hol = pd.DataFrame(rows)

panel, _ = load_panel_with_features()
dyn = panel[["GEOID", "month"]].drop_duplicates().merge(hol, on="month", how="inner")
run_fusion("Temporal (US+IL holidays)", ["pct_holiday_days"],
           dynamic_panel=dyn[["GEOID", "month", "pct_holiday_days"]])
