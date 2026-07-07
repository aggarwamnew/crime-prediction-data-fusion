"""
Vancouver ALL-CRIME run at the 24-NEIGHBOURHOOD level.

Violent/person offences are coordinate-suppressed (no X/Y) but DO carry a neighbourhood,
so at neighbourhood resolution we can include ALL 11 crime types (incl. violent). Coarser
unit (24 areas) than the DA run, but complete crime coverage. Same features/RF/split.
This complements the DA property-crime run and tests whether all-crime is predictable
where violent crime can be placed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
CRIME = ROOT / "data/raw/vancouver/crime/vancouver_crime_2023_2025.csv"

print("=" * 70)
print("VANCOUVER ALL-CRIME @ 24 NEIGHBOURHOODS (incl. violent)")
print("=" * 70)

df = pd.read_csv(CRIME)
df.columns = [c.lower() for c in df.columns]
df = df.dropna(subset=["neighbourhood"]).copy()
df["month"] = df["year"].astype(str) + "-" + df["month"].astype(int).map(lambda m: f"{m:02d}")
print(f"   incidents (all types, with neighbourhood): {len(df):,}")
print(f"   neighbourhoods: {df['neighbourhood'].nunique()} | types: {df['type'].nunique()}")

monthly = df.groupby(["neighbourhood", "month"]).size().reset_index(name="crime_count")
all_months = sorted(monthly["month"].unique())
hoods = sorted(monthly["neighbourhood"].unique())
grid = pd.MultiIndex.from_product([hoods, all_months], names=["neighbourhood", "month"])
panel = monthly.set_index(["neighbourhood", "month"]).reindex(grid, fill_value=0).reset_index()
panel = panel.sort_values(["neighbourhood", "month"])

for lag in [1, 2, 3, 6, 12]:
    panel[f"lag_{lag}"] = panel.groupby("neighbourhood")["crime_count"].shift(lag)
for w in [3, 6, 12]:
    panel[f"rolling_mean_{w}"] = panel.groupby("neighbourhood")["crime_count"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
mn = panel["month"].str[-2:].astype(int)
panel["month_sin"] = np.sin(2 * np.pi * mn / 12)
panel["month_cos"] = np.cos(2 * np.pi * mn / 12)
panel["time_idx"] = panel["month"].map({m: i for i, m in enumerate(all_months)})

feats = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
         "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month_sin", "month_cos", "time_idx"]
mdf = panel.dropna().copy()
test_months = all_months[-6:]
tr = mdf[~mdf["month"].isin(test_months)]
te = mdf[mdf["month"].isin(test_months)]
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(tr[feats], tr["crime_count"])
pred = rf.predict(te[feats])
print(f"\n   {len(hoods)} neighbourhoods x {len(all_months)} months | train {len(tr)}/test {len(te)}")
print(f"   Test R2:  {r2_score(te['crime_count'], pred):.4f}")
print(f"   Test MAE: {mean_absolute_error(te['crime_count'], pred):.2f} (mean {te['crime_count'].mean():.1f})")
imp = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])[:3]
print("   Top 3: " + ", ".join(f"{f} ({v:.3f})" for f, v in imp))
