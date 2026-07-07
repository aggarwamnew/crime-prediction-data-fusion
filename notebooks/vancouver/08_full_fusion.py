"""
Vancouver full fusion: combine the available layers (CIMD + weather + POIs) on the
DA panel and compare to baseline, to fill the cross-city full-fusion comparison.
(Mental health has no open DA-level source; demographics not built for Vancouver.)
"""
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from _fusion import load_panel_with_features, _train_eval, BASE_FEATS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"
DA = RAW / "boundaries/vancouver_da_2021.geojson"

print("=" * 70)
print("VANCOUVER FULL FUSION (CIMD + weather + POIs)")
print("=" * 70)

panel, months = load_panel_with_features()
test_months = months[-6:]
df = panel.copy()
cols = []

# --- CIMD (static, by DAUID) ---
z = zipfile.ZipFile(RAW / "cimd/bc_scores_quintiles.zip")
name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
score_cols = [c for c in cimd.columns if c.strip().endswith("Scores")]
cimd = cimd[[da_col] + score_cols].rename(columns={da_col: "DAUID"})
cimd["DAUID"] = cimd["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in score_cols:
    cimd[c] = pd.to_numeric(cimd[c], errors="coerce")
df = df.merge(cimd, on="DAUID", how="inner")
cols += score_cols

# --- Weather (monthly, city-wide) ---
frames = [pd.read_csv(RAW / f"weather/yvr_{y}.csv") for y in (2023, 2024, 2025)]
w = pd.concat(frames, ignore_index=True)
w.columns = [c.strip() for c in w.columns]
def col(sub): return [c for c in w.columns if sub.lower() in c.lower()][0]
w["month"] = pd.to_datetime(w[col("Date/Time")]).dt.strftime("%Y-%m")
tmax, tmin, prcp = col("Max Temp"), col("Min Temp"), col("Total Precip")
for c in (tmax, tmin, prcp):
    w[c] = pd.to_numeric(w[c], errors="coerce")
agg = w.groupby("month").agg(tmax_mean=(tmax, "mean"), tmin_mean=(tmin, "mean"),
                             prcp_total=(prcp, "sum")).reset_index()
wcols = ["tmax_mean", "tmin_mean", "prcp_total"]
df = df.merge(agg, on="month", how="inner")
cols += wcols

# --- POIs (static, by DAUID via spatial join) ---
p = pd.read_csv(RAW / "pois/vancouver_pois_raw.csv")
gp = gpd.GeoDataFrame(p, geometry=[Point(xy) for xy in zip(p.lon, p.lat)], crs="EPSG:4326")
da = gpd.read_file(DA)[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(gp, da, how="inner", predicate="within")
counts = j.groupby(["DAUID", "category"]).size().unstack(fill_value=0).reset_index()
cat_cols = [c for c in counts.columns if c != "DAUID"]
counts["poi_total"] = counts[cat_cols].sum(axis=1)
pcols = cat_cols + ["poi_total"]
df = df.merge(counts[["DAUID"] + pcols], on="DAUID", how="inner")
cols += pcols

df = df.dropna(subset=cols)
n = df["DAUID"].nunique()
base_r2, base_mae, ntr, nte = _train_eval(df, BASE_FEATS, test_months)
full_r2, full_mae, _, _ = _train_eval(df, BASE_FEATS + cols, test_months)
print(f"\n   DAs matched: {n} | train {ntr:,}/test {nte:,} | features {len(BASE_FEATS)+len(cols)}")
print(f"   Baseline R2: {base_r2:.4f}")
print(f"   FULL fusion: {full_r2:.4f}")
print(f"   Delta R2:    {full_r2 - base_r2:+.4f}")
print(f"\n   (London full fusion +0.0020, Chicago +0.0021)")
