"""
Layer: DEPRIVATION for Vancouver = StatCan Canadian Index of Multiple Deprivation (CIMD)
2021, BC file, at Dissemination Area level. The direct IMD analogue (clean DAUID join).
Features: the four CIMD dimension scores (residential instability, economic dependency,
ethno-cultural composition, situational vulnerability).
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from _fusion import run_fusion

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/vancouver/cimd"
RAW.mkdir(parents=True, exist_ok=True)
ZIP = RAW / "bc_scores_quintiles.zip"
URL = "https://www150.statcan.gc.ca/n1/pub/45-20-0001/2023001/csv/bc_scores_quintiles_csv-eng.zip"
H = {"User-Agent": "Mozilla/5.0 thesis-crime-research"}

print("=" * 70)
print("VANCOUVER DEPRIVATION LAYER — StatCan CIMD 2021 (BC, DA)")
print("=" * 70)

if not ZIP.exists():
    r = requests.get(URL, headers=H, timeout=180)
    r.raise_for_status()
    ZIP.write_bytes(r.content)
    print(f"   downloaded -> {ZIP.stat().st_size/1e6:.2f} MB")

z = zipfile.ZipFile(ZIP)
name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
cimd = pd.read_csv(z.open(name), encoding="latin-1")
# identify DAUID col + score cols
da_col = [c for c in cimd.columns if "Dissemination Area" in c][0]
score_cols = [c for c in cimd.columns if c.strip().endswith("Scores")]
print(f"   DAUID col: {da_col}")
print(f"   score cols: {score_cols}")

cimd = cimd[[da_col] + score_cols].rename(columns={da_col: "DAUID"})
cimd["DAUID"] = cimd["DAUID"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in score_cols:
    cimd[c] = pd.to_numeric(cimd[c], errors="coerce")
cimd = cimd.dropna(subset=score_cols, how="all")
print(f"   CIMD DAs (BC): {len(cimd):,}")

run_fusion("CIMD", score_cols, static_by_da=cimd)
