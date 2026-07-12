"""09_verification.py — Vancouver verification parity with London (Session 21).

Cluster-bootstrap 95% CIs (by DA, B=1000) for the quoted Vancouver numbers:
baseline R2, and Delta R2 for CIMD, weather, POIs, and the combined available layers.
Plus CIMD-only (no history) static model, the Vancouver analogue of London script 34.
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

from _fusion import load_panel_with_features, BASE_FEATS

B = 1000
RF = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"


def cluster_ci(test_df, pb, pf=None, seed=42):
    y = test_df["crime_count"].values
    codes, _ = pd.factorize(test_df["DAUID"].values)
    n = codes.max() + 1
    gidx = [np.flatnonzero(codes == g) for g in range(n)]
    rng = np.random.default_rng(seed)
    r2b = np.empty(B)
    d = np.empty(B) if pf is not None else None
    for b in range(B):
        idx = np.concatenate([gidx[g] for g in rng.integers(0, n, n)])
        r2b[b] = r2_score(y[idx], pb[idx])
        if pf is not None:
            d[b] = r2_score(y[idx], pf[idx]) - r2b[b]
    out = {"r2": r2_score(y, pb), "r2_lo": np.percentile(r2b, 2.5), "r2_hi": np.percentile(r2b, 97.5)}
    if pf is not None:
        out.update({"delta": r2_score(y, pf) - out["r2"],
                    "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5)})
    return out


def fit_pair(df, months, base, extra):
    d = df.dropna(subset=base + extra + ["crime_count"])
    tmv = months[-6:]
    tr, te = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)].reset_index(drop=True)
    rf1 = RandomForestRegressor(**RF); rf1.fit(tr[base], tr["crime_count"])
    pb = rf1.predict(te[base])
    pf = None
    if extra:
        rf2 = RandomForestRegressor(**RF); rf2.fit(tr[base + extra], tr["crime_count"])
        pf = rf2.predict(te[base + extra])
    return te, pb, pf


# ── layers (same specs as scripts 04-06) ──
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
snowc = [c for c in w.columns if "Total Snow" in c]
if snowc:
    w[snowc[0]] = pd.to_numeric(w[snowc[0]], errors="coerce")
    wagg = wagg.merge(w.groupby("month")[snowc[0]].sum().rename("snow_total").reset_index(), on="month")
    wcols = ["tmax_mean", "tmin_mean", "prcp_total", "snow_total"]
else:
    wcols = ["tmax_mean", "tmin_mean", "prcp_total"]

pois = pd.read_csv(RAW / "pois/vancouver_pois_raw.csv")
gp = gpd.GeoDataFrame(pois, geometry=[Point(xy) for xy in zip(pois.lon, pois.lat)], crs="EPSG:4326")
da = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(gp, da, how="inner", predicate="within")
poi_counts = j.groupby(["DAUID", "category"]).size().unstack(fill_value=0).reset_index()
poi_cols = [c for c in poi_counts.columns if c != "DAUID"]
poi_counts["poi_total"] = poi_counts[poi_cols].sum(axis=1)
poi_cols = poi_cols + ["poi_total"]

print("=" * 96)
print(f"VANCOUVER VERIFICATION (cluster bootstrap by DA, B={B})")
print("=" * 96)

panel, months = load_panel_with_features()

te, pb, _ = fit_pair(panel, months, BASE_FEATS, [])
ci = cluster_ci(te, pb)
print(f"Baseline R2 = {ci['r2']:.4f}  [{ci['r2_lo']:.4f}, {ci['r2_hi']:.4f}]")

for nm, tbl, cols, on in [("CIMD", cimd, cimd_cols, "DAUID"),
                          ("Weather", wagg, wcols, "month"),
                          ("POIs", poi_counts[["DAUID"] + poi_cols], poi_cols, "DAUID")]:
    df = panel.merge(tbl, on=on, how="inner").dropna(subset=cols)
    te, pb, pf = fit_pair(df, months, BASE_FEATS, cols)
    ci = cluster_ci(te, pb, pf)
    print(f"{nm:8s} dR2 = {ci['delta']:+.4f}  [{ci['d_lo']:+.4f}, {ci['d_hi']:+.4f}]  (base {ci['r2']:.4f})", flush=True)

# combined available layers (script 08 spec)
df = (panel.merge(cimd, on="DAUID", how="inner")
      .merge(wagg, on="month", how="inner")
      .merge(poi_counts[["DAUID"] + poi_cols], on="DAUID", how="left"))
df[poi_cols] = df[poi_cols].fillna(0)
allc = cimd_cols + wcols + poi_cols
df = df.dropna(subset=cimd_cols + wcols)
te, pb, pf = fit_pair(df, months, BASE_FEATS, allc)
ci = cluster_ci(te, pb, pf)
print(f"Combined dR2 = {ci['delta']:+.4f}  [{ci['d_lo']:+.4f}, {ci['d_hi']:+.4f}]  (base {ci['r2']:.4f})")

# CIMD-only (no history)
d = panel.merge(cimd, on="DAUID", how="inner").dropna(subset=cimd_cols + ["crime_count"])
tmv = months[-6:]
tr, te2 = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)]
rf = RandomForestRegressor(**RF); rf.fit(tr[cimd_cols], tr["crime_count"])
print(f"CIMD-only (no history) R2 = {r2_score(te2['crime_count'], rf.predict(te2[cimd_cols])):.4f}")
