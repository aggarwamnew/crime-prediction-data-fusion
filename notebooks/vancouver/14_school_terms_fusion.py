"""14_school_terms_fusion.py — Vancouver temporal-activity layer with school terms.

Completes the London temporal-activity analogue for Vancouver: fraction of days per
month in VSB school vacations + fraction that are BC statutory holidays. Term dates
verified against the official VSB compact calendars (2022-23 PDF via media.vsb.bc.ca
docs archive; 2023-24 and 2024-25 medialib PDFs; 2025-26 school-year-calendar page):
  SY2022-23: winter Dec 19-Jan 02; spring Mar 13-24 2023; last instruction Jun 29 2023
  SY2023-24: first day Sep 05 2023; winter Dec 25-Jan 07; spring Mar 18-28 2024;
             last instruction Jun 27 2024
  SY2024-25: first day Sep 03 2024; winter Dec 23-Jan 03; spring Mar 17-28 2025;
             last instruction Jun 26 2025
  SY2025-26: first day Sep 02 2025; winter break from Dec 22 2025
Summer = day after last instruction .. day before first day. Pro-D single days omitted
(as in the London construct). Statutory holidays enter as a separate feature.
"""
import sys
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holidays as pyhol
import pandas as pd

from _fusion import run_fusion

BREAKS = [
    ("2023-01-01", "2023-01-02"),   # winter 22-23 tail (school reopens Jan 3)
    ("2023-03-13", "2023-03-24"),   # spring 22-23
    ("2023-06-30", "2023-09-04"),   # summer 2023 (first day SY23-24 = Sep 5)
    ("2023-12-25", "2024-01-07"),   # winter 23-24 (reopens Jan 8)
    ("2024-03-18", "2024-03-28"),   # spring 23-24
    ("2024-06-28", "2024-09-02"),   # summer 2024 (first day SY24-25 = Sep 3)
    ("2024-12-23", "2025-01-03"),   # winter 24-25 (reopens Jan 6)
    ("2025-03-17", "2025-03-28"),   # spring 24-25
    ("2025-06-27", "2025-09-01"),   # summer 2025 (first day SY25-26 = Sep 2)
    ("2025-12-22", "2025-12-31"),   # winter 25-26 within study window
]
break_days = set()
for a, b in BREAKS:
    break_days.update(d.date() for d in pd.date_range(a, b))

bc = pyhol.country_holidays("CA", subdiv="BC", years=range(2023, 2026))

rows = []
for ms in pd.date_range("2023-01-01", "2025-12-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0))
    rows.append({
        "month": f"{ms.year}-{ms.month:02d}",
        "pct_school_break": sum(1 for d in days if d.date() in break_days) / dim,
        "pct_holiday_days": sum(1 for d in days if d.date() in bc) / dim,
    })
tf = pd.DataFrame(rows)
print(tf.to_string(index=False))

run_fusion("Temporal (VSB terms + BC holidays)", ["pct_school_break", "pct_holiday_days"],
           month_only=tf)
