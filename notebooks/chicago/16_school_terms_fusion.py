"""16_school_terms_fusion.py — Chicago temporal-activity layer with school terms.

Completes the London temporal-activity analogue: fraction of days per month in CPS
school breaks + fraction that are US federal/Illinois holidays. Break dates verified
against CPS announcements/official calendars (WTTW 2022-03-22; NBC Chicago; Block Club
2023-02-23; Chicago Sun-Times 2024-02-01; official CPS 2025-26 PDF):
  SY2022-23: classes Aug 22 2022 - Jun 7 2023; winter Dec 23-Jan 6; spring Apr 3-7 2023
  SY2023-24: classes Aug 21 2023 - Jun 6 2024; winter Dec 22-Jan 5; spring Mar 25-Apr 1
  SY2024-25: classes Aug 26 2024 - Jun 12 2025; Thanksgiving Nov 25-29 (full week from
             this year); winter Dec 23-Jan 3; spring Mar 24-28 2025
  SY2025-26: classes Aug 18 2025 - Jun 4 2026; Thanksgiving Nov 24-28; winter Dec 22-Jan 2
Summer = day after last day .. day before first day. Pro-D/report-card single days omitted.
"""
import sys
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holidays as pyhol
import pandas as pd

from _fusion import run_fusion, load_panel_with_features

BREAKS = [
    ("2023-01-01", "2023-01-06"),   # winter 22-23 (in-window part)
    ("2023-04-03", "2023-04-07"),   # spring 22-23
    ("2023-06-08", "2023-08-20"),   # summer 2023
    ("2023-12-22", "2024-01-05"),   # winter 23-24
    ("2024-03-25", "2024-04-01"),   # spring 23-24
    ("2024-06-07", "2024-08-25"),   # summer 2024
    ("2024-11-25", "2024-11-29"),   # Thanksgiving week 24-25
    ("2024-12-23", "2025-01-03"),   # winter 24-25
    ("2025-03-24", "2025-03-28"),   # spring 24-25
    ("2025-06-13", "2025-08-17"),   # summer 2025
    ("2025-11-24", "2025-11-28"),   # Thanksgiving week 25-26
    ("2025-12-22", "2026-01-02"),   # winter 25-26
]
break_days = set()
for a, b in BREAKS:
    break_days.update(d.date() for d in pd.date_range(a, b))

il = pyhol.country_holidays("US", subdiv="IL", years=range(2023, 2027))

rows = []
for ms in pd.date_range("2023-01-01", "2026-01-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0))
    rows.append({
        "month": f"{ms.year}-{ms.month:02d}",
        "pct_school_break": sum(1 for d in days if d.date() in break_days) / dim,
        "pct_holiday_days": sum(1 for d in days if d.date() in il) / dim,
    })
tf = pd.DataFrame(rows)
print(tf.to_string(index=False))

panel, _ = load_panel_with_features()
dyn = panel[["GEOID", "month"]].drop_duplicates().merge(tf, on="month", how="inner")
run_fusion("Temporal (CPS terms + holidays)", ["pct_school_break", "pct_holiday_days"],
           dynamic_panel=dyn)
