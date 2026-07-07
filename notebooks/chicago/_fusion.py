"""
Shared Chicago fusion harness — mirrors the London per-layer fusion pattern.

Loads the baseline tract-month panel, rebuilds the 11 baseline features, then for a
given supplementary layer:
  - inner-joins the layer (restricts to tracts/months that have it),
  - trains baseline (11 feats) and fused (11 + layer feats) on the SAME rows/split,
  - returns baseline R2, fused R2, Delta R2.
Same RF(200, depth=15, leaf=5, seed=42) and last-6-months test split as London.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data/processed/chicago"
BASE_FEATS = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
              "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
              "month_sin", "month_cos", "time_idx"]


def load_panel_with_features():
    """Return the tract-month panel with the 11 baseline features built."""
    panel = pd.read_parquet(PROC / "tract_month_panel.parquet").sort_values(["GEOID", "month"])
    for lag in [1, 2, 3, 6, 12]:
        panel[f"lag_{lag}"] = panel.groupby("GEOID")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        panel[f"rolling_mean_{w}"] = panel.groupby("GEOID")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    panel["month_num"] = panel["month"].str[-2:].astype(int)
    panel["month_sin"] = np.sin(2 * np.pi * panel["month_num"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month_num"] / 12)
    months = sorted(panel["month"].unique())
    panel["time_idx"] = panel["month"].map({m: i for i, m in enumerate(months)})
    return panel, months


def _train_eval(df, feature_cols, test_months):
    d = df.dropna(subset=feature_cols + ["crime_count"]).copy()
    tr = d[~d["month"].isin(test_months)]
    te = d[d["month"].isin(test_months)]
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(tr[feature_cols], tr["crime_count"])
    pred = rf.predict(te[feature_cols])
    return r2_score(te["crime_count"], pred), mean_absolute_error(te["crime_count"], pred), len(tr), len(te)


def run_fusion(layer_name, layer_cols, static_by_geoid=None, dynamic_panel=None):
    """
    Fuse one layer and report baseline vs fused on the matched set.
      static_by_geoid: DataFrame with GEOID + layer_cols (time-invariant), OR
      dynamic_panel:   DataFrame with GEOID, month + layer_cols (time-varying).
    """
    panel, months = load_panel_with_features()
    test_months = months[-6:]

    if static_by_geoid is not None:
        merged = panel.merge(static_by_geoid, on="GEOID", how="inner")
    else:
        merged = panel.merge(dynamic_panel, on=["GEOID", "month"], how="inner")

    # restrict to rows where layer values are present
    merged = merged.dropna(subset=layer_cols)
    n_tracts = merged["GEOID"].nunique()

    base_r2, base_mae, ntr, nte = _train_eval(merged, BASE_FEATS, test_months)
    fused_r2, fused_mae, _, _ = _train_eval(merged, BASE_FEATS + layer_cols, test_months)

    print(f"\n--- {layer_name} ---")
    print(f"   Tracts matched: {n_tracts}  | train {ntr:,} / test {nte:,}")
    print(f"   Baseline R2:  {base_r2:.4f}  (MAE {base_mae:.3f})")
    print(f"   + {layer_name}: {fused_r2:.4f}  (MAE {fused_mae:.3f})")
    print(f"   Delta R2:     {fused_r2 - base_r2:+.4f}")
    return {"layer": layer_name, "tracts": n_tracts,
            "base_r2": base_r2, "fused_r2": fused_r2, "delta_r2": fused_r2 - base_r2}
