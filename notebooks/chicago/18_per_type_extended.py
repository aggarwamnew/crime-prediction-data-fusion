"""
Chicago PER-CRIME-TYPE fusion, extended to the parity layers (Session 22).

Script 12 covered SVI/Weather/Demo/MentHlth/CTA/Divvy/POI per type. This script adds
the layers closed at aggregate level in scripts 15-17, so the per-type table matches
London Table 5.6 construct for construct:
  Education (EP_NOHSDP), Household (EP_SNGPNT), Housing burden (EP_HBURD),
  Housing assessed-value tertiles (Cook County), Temporal (CPS terms + US/IL holidays),
  and the combined Socio-Structural block (mhlth + EP_NOHSDP + EP_SNGPNT, London's SS).
Same harness as script 12: per-layer inner join, per-layer baseline, RF 200/15/5/42.
Merges the new columns into data/processed/chicago/per_type_fusion.csv.
"""
import sys
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holidays as pyhol
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import layers as L

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/chicago"
PROC = ROOT / "data/processed/chicago"
BASE = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month_sin", "month_cos", "time_idx"]

TYPES = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "ROBBERY",
         "MOTOR VEHICLE THEFT", "BURGLARY", "NARCOTICS", "WEAPONS VIOLATION", "DECEPTIVE PRACTICE"]

print("=" * 70)
print("CHICAGO PER-CRIME-TYPE FUSION - PARITY-LAYER EXTENSION")
print("=" * 70)

cnt = pd.read_parquet(PROC / "tract_month_type.parquet")
active = pd.read_parquet(PROC / "tract_month_panel.parquet")["GEOID"].unique()
all_months = sorted(cnt["month"].unique())
test_months = all_months[-6:]
month_idx = {m: i for i, m in enumerate(all_months)}


def build_features(sub):
    sub = sub.sort_values(["GEOID", "month"])
    for lag in [1, 2, 3, 6, 12]:
        sub[f"lag_{lag}"] = sub.groupby("GEOID")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        sub[f"rolling_mean_{w}"] = sub.groupby("GEOID")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    mn = sub["month"].str[-2:].astype(int)
    sub["month_sin"] = np.sin(2 * np.pi * mn / 12)
    sub["month_cos"] = np.cos(2 * np.pi * mn / 12)
    sub["time_idx"] = sub["month"].map(month_idx)
    return sub


def evals(df, feats):
    d = df.dropna(subset=feats + ["crime_count"])
    tr = d[~d["month"].isin(test_months)]
    te = d[d["month"].isin(test_months)]
    if len(te) < 50 or tr["crime_count"].sum() == 0:
        return None
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(tr[feats], tr["crime_count"])
    return r2_score(te["crime_count"], rf.predict(te[feats]))


# ── new layer specs ──
svi = pd.read_csv(RAW / "svi/svi_2022_illinois.csv", dtype={"FIPS": str}).rename(columns={"FIPS": "GEOID"})
for c in ["EP_NOHSDP", "EP_SNGPNT", "EP_HBURD"]:
    svi[c] = pd.to_numeric(svi[c], errors="coerce")
    svi.loc[svi[c] < 0, c] = pd.NA

# combined socio-structural block (London SS = mental health + education + household)
_, mh, _ = L.mental_health()
ss = svi[["GEOID", "EP_NOHSDP", "EP_SNGPNT"]].merge(mh, on="GEOID", how="inner").dropna()

# housing tertiles from cached Cook County files (cache the tract medians for reuse)
med_cache = PROC / "housing_tertiles.csv"
if med_cache.exists():
    med = pd.read_csv(med_cache, dtype={"GEOID": str})
else:
    vals = pd.read_csv(RAW / "housing/assessed_2025.csv", dtype=str)
    uni = pd.read_csv(RAW / "housing/parcel_tracts.csv", dtype=str,
                      usecols=["pin", "census_tract_geoid"])
    vals["certified_tot"] = pd.to_numeric(vals["certified_tot"], errors="coerce")
    vals = vals.dropna(subset=["certified_tot"])
    vals = vals[vals["certified_tot"] > 0]
    uni = uni.dropna(subset=["census_tract_geoid"]).drop_duplicates("pin")
    m = vals.merge(uni, on="pin", how="inner")
    med = m.groupby("census_tract_geoid")["certified_tot"].median().reset_index()
    med = med.rename(columns={"census_tract_geoid": "GEOID", "certified_tot": "median_value"})
    q = med["median_value"].quantile([1 / 3, 2 / 3])
    med["price_low"] = (med["median_value"] < q.iloc[0]).astype(int)
    med["price_mid"] = ((med["median_value"] >= q.iloc[0]) & (med["median_value"] < q.iloc[1])).astype(int)
    med["price_high"] = (med["median_value"] >= q.iloc[1]).astype(int)
    med.to_csv(med_cache, index=False)
print(f"   housing tertiles: {len(med):,} tracts")

# temporal: CPS terms (verified dates, script 16) + US/IL holidays, month-level join
BREAKS = [
    ("2023-01-01", "2023-01-06"), ("2023-04-03", "2023-04-07"), ("2023-06-08", "2023-08-20"),
    ("2023-12-22", "2024-01-05"), ("2024-03-25", "2024-04-01"), ("2024-06-07", "2024-08-25"),
    ("2024-11-25", "2024-11-29"), ("2024-12-23", "2025-01-03"), ("2025-03-24", "2025-03-28"),
    ("2025-06-13", "2025-08-17"), ("2025-11-24", "2025-11-28"), ("2025-12-22", "2026-01-02"),
]
break_days = set()
for a, b in BREAKS:
    break_days.update(d.date() for d in pd.date_range(a, b))
il = pyhol.country_holidays("US", subdiv="IL", years=range(2023, 2027))
rows = []
for ms in pd.date_range("2023-01-01", "2026-01-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0))
    rows.append({"month": f"{ms.year}-{ms.month:02d}",
                 "pct_school_break": sum(1 for d in days if d.date() in break_days) / dim,
                 "pct_holiday_days": sum(1 for d in days if d.date() in il) / dim})
tf = pd.DataFrame(rows)

LSPECS = {
    "Education": ("static", svi[["GEOID", "EP_NOHSDP"]].dropna(), ["EP_NOHSDP"]),
    "Household": ("static", svi[["GEOID", "EP_SNGPNT"]].dropna(), ["EP_SNGPNT"]),
    "HousBurd": ("static", svi[["GEOID", "EP_HBURD"]].dropna(), ["EP_HBURD"]),
    "Housing": ("static", med[["GEOID", "price_low", "price_mid", "price_high"]],
                ["price_low", "price_mid", "price_high"]),
    "Temporal": ("weather_monthly", tf, ["pct_school_break", "pct_holiday_days"]),
    "SS": ("static", ss, ["mhlth", "EP_NOHSDP", "EP_SNGPNT"]),
}

results = []
for t in TYPES:
    sub = cnt[cnt["primary_type"] == t]
    grid = pd.MultiIndex.from_product([active, all_months], names=["GEOID", "month"])
    panel = (sub.groupby(["GEOID", "month"])["n"].sum().reindex(grid, fill_value=0)
             .reset_index(name="crime_count"))
    panel = build_features(panel)
    row = {"type": t}
    for name, (kind, tbl, cols) in LSPECS.items():
        if kind == "weather_monthly":
            m = panel.merge(tbl, on="month", how="inner")
        else:
            m = panel.merge(tbl, on="GEOID", how="inner")
        m = m.dropna(subset=cols)
        b = evals(m, BASE)
        f = evals(m, BASE + cols)
        row[name] = (f - b) if (b is not None and f is not None) else np.nan
    results.append(row)
    print(f"   {t:22s} " + "  ".join(f"{k}={row[k]:+.4f}" for k in LSPECS), flush=True)

new = pd.DataFrame(results).set_index("type")
old = pd.read_csv(PROC / "per_type_fusion.csv").set_index("type")
old = old.drop(columns=[c for c in new.columns if c in old.columns])
res = old.join(new)
res.to_csv(PROC / "per_type_fusion.csv")
pd.set_option("display.width", 250, "display.max_columns", 25)
print("\nFULL PER-TYPE TABLE (all layers):")
print(res.round(4).to_string())
print("\nBest layer per type:")
layer_cols = [c for c in res.columns if c != "base_R2"]
for t in res.index:
    best = res.loc[t, layer_cols].astype(float).idxmax()
    print(f"   {t:22s} -> {best} ({res.loc[t, best]:+.4f})")
