"""16_full_fusion_all.py — Vancouver full fusion across ALL available layers.

Supersedes script 08 (which combined only CIMD+weather+POI). Combines every layer now
integrated: CIMD, weather, POIs, density, age structure, education, household, housing
tertiles, temporal (VSB terms + BC holidays), and Mobi bike-share. This is the honest
'full fusion' figure for the cross-city comparison table.
"""
import sys
import zipfile
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import holidays as pyhol
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from _fusion import load_panel_with_features, BASE_FEATS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"
RF = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)

panel, months = load_panel_with_features()
df = panel.copy()
all_cols = []

# CIMD
z = zipfile.ZipFile(RAW / "cimd/bc_scores_quintiles.zip")
name = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
cimd_cols = [c for c in cimd.columns if c.strip().endswith("Scores")]
pop_col = [c for c in cimd.columns if "Population" in c][0]
cimd = cimd[[da_col, pop_col] + cimd_cols].rename(columns={da_col: "DAUID", pop_col: "da_pop"})
cimd["DAUID"] = cimd["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in cimd_cols + ["da_pop"]:
    cimd[c] = pd.to_numeric(cimd[c], errors="coerce")
df = df.merge(cimd, on="DAUID", how="inner"); all_cols += cimd_cols

# density
bounds = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "LANDAREA"]]
bounds["LANDAREA"] = pd.to_numeric(bounds["LANDAREA"], errors="coerce")
df = df.merge(bounds, on="DAUID", how="left")
df["pop_density"] = df["da_pop"] / df["LANDAREA"]
all_cols += ["pop_density", "da_pop"]

# weather
frames = [pd.read_csv(RAW / f"weather/yvr_{y}.csv") for y in (2023, 2024, 2025)]
w = pd.concat(frames, ignore_index=True); w.columns = [c.strip() for c in w.columns]
col = lambda s: [c for c in w.columns if s.lower() in c.lower()][0]
w["month"] = pd.to_datetime(w[col("Date/Time")]).dt.strftime("%Y-%m")
for cc in (col("Max Temp"), col("Min Temp"), col("Total Precip")):
    w[cc] = pd.to_numeric(w[cc], errors="coerce")
wagg = w.groupby("month").agg(tmax_mean=(col("Max Temp"), "mean"), tmin_mean=(col("Min Temp"), "mean"),
                              prcp_total=(col("Total Precip"), "sum")).reset_index()
df = df.merge(wagg, on="month", how="left"); all_cols += ["tmax_mean", "tmin_mean", "prcp_total"]

# POIs
pois = pd.read_csv(RAW / "pois/vancouver_pois_raw.csv")
gp = gpd.GeoDataFrame(pois, geometry=[Point(xy) for xy in zip(pois.lon, pois.lat)], crs="EPSG:4326")
da_g = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "geometry"]].to_crs("EPSG:4326")
j = gpd.sjoin(gp, da_g, how="inner", predicate="within")
pc = j.groupby(["DAUID", "category"]).size().unstack(fill_value=0).reset_index()
pcols = [c for c in pc.columns if c != "DAUID"]; pc["poi_total"] = pc[pcols].sum(axis=1)
pcols = pcols + ["poi_total"]
df = df.merge(pc[["DAUID"] + pcols], on="DAUID", how="left"); df[pcols] = df[pcols].fillna(0)
all_cols += pcols

# census layers (from cached zip)
z2 = zipfile.ZipFile(RAW / "census/census2021_da_bc.zip")
csvs = [n for n in z2.namelist() if "data_BritishColumbia" in n and n.lower().endswith(".csv")]
TARGETS = {"Population, 2021": "pop_total", "0 to 14 years": "age_0_14", "65 years and over": "age_65p",
           "No certificate, diploma or degree": "no_qual", "One-person households": "one_person"}
want = set(df["DAUID"].astype(str).unique())
parts = []
with z2.open(csvs[0]) as fh:
    for chunk in pd.read_csv(fh, chunksize=1_000_000, encoding="latin-1", dtype=str, on_bad_lines="skip"):
        ch = chunk[["ALT_GEO_CODE", "CHARACTERISTIC_NAME", "C1_COUNT_TOTAL"]].copy()
        ch.columns = ["dauid", "char", "val"]
        ch = ch[ch["dauid"].isin(want)]
        ch["char"] = ch["char"].str.strip()
        ch = ch[ch["char"].isin(TARGETS)]
        if len(ch):
            parts.append(ch)
cen = pd.concat(parts, ignore_index=True)
cen["val"] = pd.to_numeric(cen["val"], errors="coerce")
cen = cen.drop_duplicates(["dauid", "char"], keep="first")
wide = cen.pivot(index="dauid", columns="char", values="val").rename(columns=TARGETS).reset_index().rename(columns={"dauid": "DAUID"})
wide = wide[wide["pop_total"] > 0]
wide["pct_under15"] = wide["age_0_14"] / wide["pop_total"]
wide["pct_65plus"] = wide["age_65p"] / wide["pop_total"]
wide["pct_no_qual"] = wide["no_qual"] / wide["pop_total"]
wide["pct_one_person"] = wide["one_person"] / wide["pop_total"]
cencols = ["pct_under15", "pct_65plus", "pct_no_qual", "pct_one_person"]
df = df.merge(wide[["DAUID"] + cencols], on="DAUID", how="left"); all_cols += cencols

# housing tertiles (recompute from cached files, as script 12)
tax = pd.read_csv(RAW / "housing/tax_report.csv", sep=";", dtype=str)
parc = pd.read_csv(RAW / "housing/parcel_points.csv", sep=";", dtype=str)
for c in ["current_land_value", "current_improvement_value"]:
    tax[c] = pd.to_numeric(tax[c], errors="coerce")
tax["total_value"] = tax["current_land_value"].fillna(0) + tax["current_improvement_value"].fillna(0)
tax = tax[tax["total_value"] > 0]
parc = parc.dropna(subset=["geo_point_2d", "tax_coord"]).drop_duplicates("tax_coord")
ll = parc["geo_point_2d"].str.split(",", expand=True)
parc["lat"] = pd.to_numeric(ll[0], errors="coerce"); parc["lon"] = pd.to_numeric(ll[1], errors="coerce")
m2 = tax.merge(parc[["tax_coord", "lat", "lon"]], left_on="land_coordinate", right_on="tax_coord", how="inner")
g2 = gpd.GeoDataFrame(m2, geometry=[Point(xy) for xy in zip(m2.lon, m2.lat)], crs="EPSG:4326")
j2 = gpd.sjoin(g2, da_g, how="inner", predicate="within")
med = j2.groupby("DAUID")["total_value"].median().reset_index()
q = med["total_value"].quantile([1/3, 2/3])
med["price_low"] = (med["total_value"] < q.iloc[0]).astype(int)
med["price_mid"] = ((med["total_value"] >= q.iloc[0]) & (med["total_value"] < q.iloc[1])).astype(int)
med["price_high"] = (med["total_value"] >= q.iloc[1]).astype(int)
df = df.merge(med[["DAUID", "price_low", "price_mid", "price_high"]], on="DAUID", how="left")
all_cols += ["price_low", "price_mid", "price_high"]

# temporal (VSB terms + BC holidays, as script 14)
BREAKS = [("2023-01-01","2023-01-02"),("2023-03-13","2023-03-24"),("2023-06-30","2023-09-04"),
          ("2023-12-25","2024-01-07"),("2024-03-18","2024-03-28"),("2024-06-28","2024-09-02"),
          ("2024-12-23","2025-01-03"),("2025-03-17","2025-03-28"),("2025-06-27","2025-09-01"),
          ("2025-12-22","2025-12-31")]
bdays = set()
for a, b in BREAKS:
    bdays.update(d.date() for d in pd.date_range(a, b))
bc = pyhol.country_holidays("CA", subdiv="BC", years=range(2023, 2026))
rows = []
for ms in pd.date_range("2023-01-01", "2025-12-01", freq="MS"):
    dim = monthrange(ms.year, ms.month)[1]
    days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0))
    rows.append({"month": f"{ms.year}-{ms.month:02d}",
                 "pct_school_break": sum(1 for d in days if d.date() in bdays) / dim,
                 "pct_holiday_days": sum(1 for d in days if d.date() in bc) / dim})
df = df.merge(pd.DataFrame(rows), on="month", how="left"); all_cols += ["pct_school_break", "pct_holiday_days"]

# Mobi (cached aggregate)
agg = pd.read_csv(RAW / "mobi/mobi_station_month.csv")
stations = pd.read_csv(RAW / "mobi/mobi_stations.csv")
norm = lambda s: (str(s).strip().lower().split(" ", 1)[1] if str(s).strip()[:4].isdigit() and " " in str(s) else str(s).strip().lower())
stations["key"] = stations["name"].map(norm); agg["key"] = agg["station"].map(norm)
mm = agg.merge(stations.drop_duplicates("key")[["key", "lat", "lon"]], on="key", how="inner")
gm = gpd.GeoDataFrame(mm, geometry=[Point(xy) for xy in zip(mm.lon, mm.lat)], crs="EPSG:4326")
jm = gpd.sjoin(gm, da_g, how="inner", predicate="within")
dyn = jm.groupby(["DAUID", "month"], as_index=False)["trips"].sum().rename(columns={"trips": "mobi_trips"})
df = df.merge(dyn, on=["DAUID", "month"], how="left"); df["mobi_trips"] = df["mobi_trips"].fillna(0)
all_cols += ["mobi_trips"]

# fusion: baseline vs everything (matched panel; drop rows missing any static layer)
need = [c for c in all_cols if c not in pcols + ["mobi_trips"]]
d = df.dropna(subset=BASE_FEATS + need + ["crime_count"])
tmv = months[-6:]
tr, te = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)].reset_index(drop=True)
print(f"panel: {d['DAUID'].nunique()} DAs | features: {len(BASE_FEATS)}+{len(all_cols)}")
r1 = RandomForestRegressor(**RF); r1.fit(tr[BASE_FEATS], tr["crime_count"])
pb = r1.predict(te[BASE_FEATS])
r2m = RandomForestRegressor(**RF); r2m.fit(tr[BASE_FEATS + all_cols], tr["crime_count"])
pf = r2m.predict(te[BASE_FEATS + all_cols])
y = te["crime_count"].values
b, f = r2_score(y, pb), r2_score(y, pf)
print(f"Baseline R2: {b:.4f}   FULL (all layers): {f:.4f}   Delta R2: {f-b:+.4f}")

# cluster bootstrap CI
codes, _ = pd.factorize(te["DAUID"].values); n = codes.max() + 1
gidx = [np.flatnonzero(codes == g) for g in range(n)]
rng = np.random.default_rng(42); dd = np.empty(1000)
for i in range(1000):
    idx = np.concatenate([gidx[g] for g in rng.integers(0, n, n)])
    dd[i] = r2_score(y[idx], pf[idx]) - r2_score(y[idx], pb[idx])
print(f"Delta CI: [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]")
