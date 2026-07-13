"""17_per_type_extended.py — Vancouver per-crime-type fusion, extended to the parity layers.

Script 11 covered CIMD/Weather/POI per type. This script adds the layers closed at
aggregate level in scripts 10 and 12-15, so Vancouver's per-type table matches London
Table 5.6 as far as its open data allows:
  Demographics (pop density + age structure), Education (pct no qualification),
  Household (pct one-person), Housing assessed-value tertiles, Temporal (VSB terms +
  BC holidays), Mobi bike-share, and the socio-structural block (education + household;
  no DA-level mental health exists, the gated layer).
Same harness as script 11 (8 features, RF 100/12/5/42, MIN>=12 crimes per DA), with
per-layer inner joins and per-layer baselines as in the Chicago per-type harness.
Merges the new columns into data/processed/vancouver/per_type_fusion.csv.
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
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver"
PROC = ROOT / "data/processed/vancouver"
RF = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)

da_g = gpd.read_file(RAW / "boundaries/vancouver_da_2021.geojson")[["DAUID", "LANDAREA", "geometry"]].to_crs("EPSG:4326")
da_g["LANDAREA"] = pd.to_numeric(da_g["LANDAREA"], errors="coerce")

# ── demographics: CIMD population / land area + census age structure ──
z = zipfile.ZipFile(RAW / "cimd/bc_scores_quintiles.zip")
name = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
pop_col = [c for c in cimd.columns if "Population" in c][0]
pop = cimd[[da_col, pop_col]].rename(columns={da_col: "DAUID", pop_col: "da_pop"})
pop["DAUID"] = pop["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
pop["da_pop"] = pd.to_numeric(pop["da_pop"], errors="coerce")
dens = pop.merge(da_g[["DAUID", "LANDAREA"]], on="DAUID", how="inner")
dens["pop_density"] = dens["da_pop"] / dens["LANDAREA"]

# ── census wide table (age / education / household), cached ──
wide_cache = PROC / "census_wide.csv"
if wide_cache.exists():
    wide = pd.read_csv(wide_cache, dtype={"DAUID": str})
else:
    z2 = zipfile.ZipFile(RAW / "census/census2021_da_bc.zip")
    csvs = [n for n in z2.namelist() if "data_BritishColumbia" in n and n.lower().endswith(".csv")]
    TARGETS = {"Population, 2021": "pop_total", "0 to 14 years": "age_0_14", "65 years and over": "age_65p",
               "No certificate, diploma or degree": "no_qual", "One-person households": "one_person"}
    want = set(da_g["DAUID"].astype(str).unique())
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
    wide.to_csv(wide_cache, index=False)
print(f"   census wide: {len(wide):,} DAs")

demo = dens[["DAUID", "pop_density"]].merge(wide[["DAUID", "pct_under15", "pct_65plus"]], on="DAUID", how="inner").dropna()

# ── housing tertiles (as scripts 12/16), cached ──
med_cache = PROC / "housing_tertiles.csv"
if med_cache.exists():
    med = pd.read_csv(med_cache, dtype={"DAUID": str})
else:
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
    j2 = gpd.sjoin(g2, da_g[["DAUID", "geometry"]], how="inner", predicate="within")
    med = j2.groupby("DAUID")["total_value"].median().reset_index()
    q = med["total_value"].quantile([1 / 3, 2 / 3])
    med["price_low"] = (med["total_value"] < q.iloc[0]).astype(int)
    med["price_mid"] = ((med["total_value"] >= q.iloc[0]) & (med["total_value"] < q.iloc[1])).astype(int)
    med["price_high"] = (med["total_value"] >= q.iloc[1]).astype(int)
    med.to_csv(med_cache, index=False)
print(f"   housing tertiles: {len(med):,} DAs")

# ── temporal: VSB terms (verified, script 14) + BC statutory holidays ──
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
tf = pd.DataFrame(rows)

# ── Mobi station-month trips per DA (as script 15) ──
agg = pd.read_csv(RAW / "mobi/mobi_station_month.csv")
stations = pd.read_csv(RAW / "mobi/mobi_stations.csv")
norm = lambda s: (str(s).strip().lower().split(" ", 1)[1] if str(s).strip()[:4].isdigit() and " " in str(s) else str(s).strip().lower())
stations["key"] = stations["name"].map(norm); agg["key"] = agg["station"].map(norm)
mm = agg.merge(stations.drop_duplicates("key")[["key", "lat", "lon"]], on="key", how="inner")
gm = gpd.GeoDataFrame(mm, geometry=[Point(xy) for xy in zip(mm.lon, mm.lat)], crs="EPSG:4326")
jm = gpd.sjoin(gm, da_g[["DAUID", "geometry"]], how="inner", predicate="within")
mobi = jm.groupby(["DAUID", "month"], as_index=False)["trips"].sum().rename(columns={"trips": "mobi_trips"})

LSPECS = {
    "Demo": ("static", demo, ["pop_density", "pct_under15", "pct_65plus"]),
    "Education": ("static", wide[["DAUID", "pct_no_qual"]].dropna(), ["pct_no_qual"]),
    "Household": ("static", wide[["DAUID", "pct_one_person"]].dropna(), ["pct_one_person"]),
    "Housing": ("static", med[["DAUID", "price_low", "price_mid", "price_high"]],
                ["price_low", "price_mid", "price_high"]),
    "Temporal": ("monthly", tf, ["pct_school_break", "pct_holiday_days"]),
    "Mobi": ("dynamic_fill0", mobi, ["mobi_trips"]),
    "SS": ("static", wide[["DAUID", "pct_no_qual", "pct_one_person"]].dropna(),
           ["pct_no_qual", "pct_one_person"]),
}

# ── crime, per type (as script 11) ──
crime = pd.read_csv(RAW / "crime/vancouver_crime_2023_2025.csv")
crime.columns = [c.lower() for c in crime.columns]
crime = crime[crime["x"].fillna(0) != 0].copy()
crime["month"] = crime["year"].astype(str) + "-" + crime["month"].astype(int).map(lambda m: f"{m:02d}")
pts = gpd.GeoDataFrame(crime, geometry=[Point(xy) for xy in zip(crime["x"], crime["y"])], crs="EPSG:32610")
pts = pts.to_crs(da_g.crs)
joined = gpd.sjoin(pts, da_g[["DAUID", "geometry"]], how="inner", predicate="within")
cnt = joined.groupby(["DAUID", "month", "type"]).size().reset_index(name="crime_count")

TYPES = ["Theft from Vehicle", "Other Theft", "Mischief", "Break and Enter Residential/Other",
         "Break and Enter Commercial", "Theft of Bicycle", "Theft of Vehicle"]

BASE = ["lag_1", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_12", "month_sin", "month_cos"]


def evals(d, feats, test_months):
    dd = d.dropna(subset=feats + ["crime_count"])
    tr, te = dd[~dd["month"].isin(test_months)], dd[dd["month"].isin(test_months)]
    if len(te) < 50:
        return None
    rf = RandomForestRegressor(**RF)
    rf.fit(tr[feats], tr["crime_count"])
    return r2_score(te["crime_count"], rf.predict(te[feats]))


rows_out = []
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

    row = {"type": t}
    for nm, (kind, tbl, cols) in LSPECS.items():
        if kind == "monthly":
            m = p.merge(tbl, on="month", how="inner")
        elif kind == "dynamic_fill0":
            m = p.merge(tbl, on=["DAUID", "month"], how="left")
            m[cols] = m[cols].fillna(0)
        else:
            m = p.merge(tbl, on="DAUID", how="inner")
        m = m.dropna(subset=cols)
        b = evals(m, BASE, tmv)
        f = evals(m, BASE + cols, tmv)
        row[nm] = (f - b) if (b is not None and f is not None) else np.nan
    rows_out.append(row)
    print(f"{t:36s} " + "  ".join(f"{k}={row[k]:+.4f}" for k in LSPECS), flush=True)

new = pd.DataFrame(rows_out).set_index("type")
old = pd.read_csv(PROC / "per_type_fusion.csv").set_index("type")
old = old.drop(columns=[c for c in new.columns if c in old.columns])
res = old.join(new)
res.to_csv(PROC / "per_type_fusion.csv")
pd.set_option("display.width", 250, "display.max_columns", 25)
print("\nFULL PER-TYPE TABLE (all layers):")
print(res.round(4).to_string())
print("\nBest layer per type:")
layer_cols = [c for c in res.columns if c not in ("base_R2", "n_das")]
for t in res.index:
    best = res.loc[t, layer_cols].astype(float).idxmax()
    print(f"   {t:36s} -> {best} ({res.loc[t, best]:+.4f})")
