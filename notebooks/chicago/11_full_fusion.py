"""
Chicago FULL FUSION — mirror of London script 20.

Combines the contextual + socio-structural layers (SVI, demographics, mental health,
weather) into one model and compares to baseline on the matched tract set. Tests whether
the layers are subadditive (London: full < context+SS) and confirms aggregate ceiling.
Transport (sparse) is excluded from the main full-fusion, exactly as London kept
PTAL/station/bike as separate diagnostics.
"""
from pathlib import Path

import pandas as pd

from _fusion import load_panel_with_features, _train_eval, BASE_FEATS
import layers

panel, months = load_panel_with_features()
test_months = months[-6:]

print("=" * 70)
print("CHICAGO FULL FUSION (contextual + socio-structural)  [mirror London 20]")
print("=" * 70)

# static layers
static_specs = [layers.svi(), layers.demographics(), layers.mental_health()]
poi = layers.pois()
if poi:
    static_specs.append(poi)
    print("   (POI cache found -> included)")
else:
    print("   (POI cache not ready -> excluded for now)")

df = panel.copy()
all_layer_cols = []
for kind, tbl, cols in static_specs:
    df = df.merge(tbl, on="GEOID", how="inner")
    all_layer_cols += cols

# weather (join on month)
_, wagg, wcols = layers.weather()
df = df.merge(wagg, on="month", how="inner")
all_layer_cols += wcols

df = df.dropna(subset=all_layer_cols)
n_tracts = df["GEOID"].nunique()

base_r2, base_mae, ntr, nte = _train_eval(df, BASE_FEATS, test_months)
full_r2, full_mae, _, _ = _train_eval(df, BASE_FEATS + all_layer_cols, test_months)

print(f"\n   Tracts matched: {n_tracts} | train {ntr:,}/test {nte:,} | features {len(BASE_FEATS)+len(all_layer_cols)}")
print(f"   Baseline R2:  {base_r2:.4f}  (MAE {base_mae:.3f})")
print(f"   FULL fusion:  {full_r2:.4f}  (MAE {full_mae:.3f})")
print(f"   Delta R2:     {full_r2 - base_r2:+.4f}")
print(f"\n   London full fusion was +0.0020 (subadditive). Chicago:")
