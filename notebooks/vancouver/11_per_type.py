"""11_per_type.py — Vancouver per-crime-type fusion (property crime types).

Per-type ablation across the available layers (CIMD, weather, POIs) for the geocodable
property crime types, using the same per-type recipe as the aggregate DA model. Lets
Vancouver join the cross-city matched-type comparison (violent types remain excluded:
coordinates suppressed under BC FIPPA).
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"
RF = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)

# ── layers (specs as scripts 04-06/09) ──
z = zipfile.ZipFile(RAW / "cimd/bc_scores_quintiles.zip")
name = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
cimd_cols = [c for c in cimd.columns if c.strip().endswith("Scores")]
cimd = cimd[[da_col] + cimd_cols].rename(columns={da_col: "DAUID"})
cimd["DAUID"] = cimd["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in cimd_cols:
    cimd[c] = pd.to_numeric(cimd[c], errors="coerce")

frames = [pd.read_csv(RAW / f"weather/yvr_{y}.csv") for y in (2023, 2024, 2025)]
w = pd.concat(frames, ignore_index=True)
w.columns = [c.strip() for c in w.columns]
col = lambda s: [c for c in w.columns if s.lower() in c.lower()][0]
w["month"] = pd.to_datetime(w[col("Date/Time")]).dt.strftime("%Y-%m")
for cc in (col("Max Temp"), col("Min Temp"), col("Total Precip")):
    w[cc] = pd.to_numeric(w[cc], errors="coerce")
wagg = w.groupby("month").agg(tmax_mean=(col("Max Temp"), "mean"), tmin_mean=(col("Min Temp"), "mean"),
                              prcp_total=(col("Total Precip"), "sum")).reset_index()
wcols = ["tmax_mean", "tmin_mean", "prcp_total"]

pois = pd.read_csv(RAW / "pois/vancouver_pois_raw.csv")
gp = gpd.GeoDataFrame(pois, geometry=[Point(xy) for xy in zip(pois.lon, pois.lat)], crs="EPSG:4326")
da_g = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "geometry"]].to_crs("EPSG:4326")
jj = gpd.sjoin(gp, da_g, how="inner", predicate="within")
poi_counts = jj.groupby(["DAUID", "category"]).size().unstack(fill_value=0).reset_index()
poi_cols = [c for c in poi_counts.columns if c != "DAUID"]
poi_counts["poi_total"] = poi_counts[poi_cols].sum(axis=1)
poi_cols = poi_cols + ["poi_total"]

# ── crime, per type ──
crime = pd.read_csv(RAW / "crime/vancouver_crime_2023_2025.csv")
crime.columns = [c.lower() for c in crime.columns]
crime = crime[crime["x"].fillna(0) != 0].copy()
crime["month"] = crime["year"].astype(str) + "-" + crime["month"].astype(int).map(lambda m: f"{m:02d}")
pts = gpd.GeoDataFrame(crime, geometry=[Point(xy) for xy in zip(crime["x"], crime["y"])], crs="EPSG:32610")
pts = pts.to_crs(da_g.crs)
joined = gpd.sjoin(pts, da_g, how="inner", predicate="within")
cnt = joined.groupby(["DAUID", "month", "type"]).size().reset_index(name="crime_count")

TYPES = ["Theft from Vehicle", "Other Theft", "Mischief", "Break and Enter Residential/Other",
         "Break and Enter Commercial", "Theft of Bicycle", "Theft of Vehicle"]

def evals(d, feats, test_months):
    dd = d.dropna(subset=feats + ["crime_count"])
    tr, te = dd[~dd["month"].isin(test_months)], dd[dd["month"].isin(test_months)]
    if len(te) < 50:
        return None
    rf = RandomForestRegressor(**RF)
    rf.fit(tr[feats], tr["crime_count"])
    return r2_score(te["crime_count"], rf.predict(te[feats]))

BASE = ["lag_1", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_12", "month_sin", "month_cos"]
rows = []
for t in TYPES:
    sub = cnt[cnt["type"] == t]
    totals = sub.groupby("DAUID")["crime_count"].sum()
    active = totals[totals >= 12].index
    if len(active) < 50:
        print(f"{t}: SKIP (few DAs)"); continue
    months = sorted(sub["month"].unique())
    grid = pd.MultiIndex.from_product([active, months], names=["DAUID", "month"])
    p = (sub[sub["DAUID"].isin(active)].groupby(["DAUID", "month"])["crime_count"].sum()
         .reindex(grid, fill_value=0).reset_index()).sort_values(["DAUID", "month"])
    for lag in [1, 3, 6, 12]:
        p[f"lag_{lag}"] = p.groupby("DAUID")["crime_count"].shift(lag)
    p["rolling_mean_3"] = p.groupby("DAUID")["crime_count"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    p["rolling_mean_12"] = p.groupby("DAUID")["crime_count"].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    mn = p["month"].str[-2:].astype(int)
    p["month_sin"] = np.sin(2 * np.pi * mn / 12)
    p["month_cos"] = np.cos(2 * np.pi * mn / 12)
    tmv = months[-6:]

    m = p.merge(cimd, on="DAUID", how="inner").merge(wagg, on="month", how="left")
    m = m.merge(poi_counts[["DAUID"] + poi_cols], on="DAUID", how="left")
    m[poi_cols] = m[poi_cols].fillna(0)
    m = m.dropna(subset=cimd_cols + wcols)

    base = evals(m, BASE, tmv)
    if base is None:
        print(f"{t}: SKIP (test too small)"); continue
    row = {"type": t, "n_das": len(active), "base_R2": base}
    for nm, cols in [("CIMD", cimd_cols), ("Weather", wcols), ("POI", poi_cols)]:
        r2f = evals(m, BASE + cols, tmv)
        row[nm] = (r2f - base) if r2f is not None else np.nan
    rows.append(row)
    print(f"{t:36s} n={len(active):3d}  base={base:+.3f}  "
          f"CIMD={row['CIMD']:+.4f}  Weather={row['Weather']:+.4f}  POI={row['POI']:+.4f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "data/processed/vancouver/per_type_fusion.csv", index=False)
print("\nSaved -> data/processed/vancouver/per_type_fusion.csv")
