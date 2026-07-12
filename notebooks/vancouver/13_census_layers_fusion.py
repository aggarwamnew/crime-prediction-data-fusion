"""13_census_layers_fusion.py — Vancouver's last closable open-data gaps (Session 21b).

Downloads the StatCan 2021 Census Profile comprehensive file for British Columbia at
DA level (download-telecharger GEONO=006, CSV) and extracts, per DA:
  - age structure: % aged 0-14, % aged 65+           (~ London TS007a)
  - education: % without certificate/diploma/degree  (~ London TS067)
  - household: % one-person households (if present)  (~ London TS003)
Then runs the standard per-layer fusion. The file is large (~GB unzipped); it is
stream-filtered in chunks to the characteristics and Vancouver-region DAs needed.
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests

from _fusion import run_fusion, load_panel_with_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/census"
RAW.mkdir(parents=True, exist_ok=True)
ZIP = RAW / "census2021_da_bc.zip"
URL = ("https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/"
       "download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=006")

print("=" * 70)
print("VANCOUVER CENSUS DA LAYERS — age / education / household")
print("=" * 70)

if not ZIP.exists():
    print("   downloading BC DA comprehensive profile (large)...", flush=True)
    with requests.get(URL, stream=True, timeout=1800,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"   saved {ZIP.stat().st_size/1e6:.0f} MB")

z = zipfile.ZipFile(ZIP)
name = [n for n in z.namelist() if n.lower().endswith(".csv") and "data" not in n.lower() or n.lower().endswith(".csv")][0]
csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
print(f"   archive members: {z.namelist()[:5]} ... using {csvs}")

TARGETS = {
    "Population, 2021": "pop_total",
    "0 to 14 years": "age_0_14",
    "65 years and over": "age_65p",
    "No certificate, diploma or degree": "no_qual",
    "One-person households": "one_person",
}

# the panel's DAUIDs (city of Vancouver, active) to filter early
panel, _ = load_panel_with_features()
want_das = set(panel["DAUID"].astype(str).unique())

parts = []
for csv_name in csvs:
    with z.open(csv_name) as fh:
        for chunk in pd.read_csv(fh, chunksize=1_000_000, encoding="latin-1",
                                 dtype=str, on_bad_lines="skip"):
            cols = {c.upper(): c for c in chunk.columns}
            geo = cols.get("ALT_GEO_CODE") or cols.get("DGUID")
            cname = next((c for u, c in cols.items() if "CHARACTERISTIC_NAME" in u), None)
            cval = next((c for u, c in cols.items() if u.startswith("C1_COUNT")), None)
            if not (geo and cname and cval):
                continue
            ch = chunk[[geo, cname, cval]].copy()
            ch.columns = ["dauid", "char", "val"]
            ch["dauid"] = ch["dauid"].str.replace(r"^2021S0512", "", regex=True)  # DGUID form
            ch = ch[ch["dauid"].isin(want_das)]
            ch["char"] = ch["char"].str.strip()
            ch = ch[ch["char"].isin(TARGETS)]
            if len(ch):
                parts.append(ch)
    break  # first (data) csv only

df = pd.concat(parts, ignore_index=True)
df["val"] = pd.to_numeric(df["val"], errors="coerce")
# first occurrence per (dauid, characteristic): profile repeats some names in sub-contexts
df = df.drop_duplicates(["dauid", "char"], keep="first")
wide = df.pivot(index="dauid", columns="char", values="val").rename(columns=TARGETS)
wide = wide.reset_index().rename(columns={"dauid": "DAUID"})
print(f"   DAs matched: {len(wide):,}  cols: {list(wide.columns)}")

wide = wide[wide.get("pop_total", pd.Series(dtype=float)) > 0]
feats = []
if {"age_0_14", "age_65p", "pop_total"}.issubset(wide.columns):
    wide["pct_under15"] = wide["age_0_14"] / wide["pop_total"]
    wide["pct_65plus"] = wide["age_65p"] / wide["pop_total"]
    feats += ["pct_under15", "pct_65plus"]
if "no_qual" in wide.columns:
    wide["pct_no_qual"] = wide["no_qual"] / wide["pop_total"]
    feats.append("pct_no_qual")
if "one_person" in wide.columns:
    wide["pct_one_person"] = wide["one_person"] / wide["pop_total"]
    feats.append("pct_one_person")
print(f"   features built: {feats}")

if {"pct_under15", "pct_65plus"}.issubset(wide.columns):
    run_fusion("Age structure (Census DA)", ["pct_under15", "pct_65plus"],
               static_by_da=wide[["DAUID", "pct_under15", "pct_65plus"]].dropna())
if "pct_no_qual" in wide.columns:
    run_fusion("Education (Census DA)", ["pct_no_qual"],
               static_by_da=wide[["DAUID", "pct_no_qual"]].dropna())
if "pct_one_person" in wide.columns:
    run_fusion("Household (Census DA)", ["pct_one_person"],
               static_by_da=wide[["DAUID", "pct_one_person"]].dropna())
