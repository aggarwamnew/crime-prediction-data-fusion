"""Vancouver fusion harness — mirror of the Chicago one, keyed on DAUID."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed/vancouver"
BASE_FEATS = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
              "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
              "month_sin", "month_cos", "time_idx"]


def load_panel_with_features():
    panel = pd.read_parquet(PROC / "da_month_panel.parquet").sort_values(["DAUID", "month"])
    for lag in [1, 2, 3, 6, 12]:
        panel[f"lag_{lag}"] = panel.groupby("DAUID")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        panel[f"rolling_mean_{w}"] = panel.groupby("DAUID")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    mn = panel["month"].str[-2:].astype(int)
    panel["month_sin"] = np.sin(2 * np.pi * mn / 12)
    panel["month_cos"] = np.cos(2 * np.pi * mn / 12)
    months = sorted(panel["month"].unique())
    panel["time_idx"] = panel["month"].map({m: i for i, m in enumerate(months)})
    return panel, months


def _train_eval(df, cols, test_months):
    d = df.dropna(subset=cols + ["crime_count"])
    tr = d[~d["month"].isin(test_months)]
    te = d[d["month"].isin(test_months)]
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(tr[cols], tr["crime_count"])
    pred = rf.predict(te[cols])
    return r2_score(te["crime_count"], pred), mean_absolute_error(te["crime_count"], pred), len(tr), len(te)


def run_fusion(name, layer_cols, static_by_da=None, dynamic_panel=None, month_only=None):
    panel, months = load_panel_with_features()
    test_months = months[-6:]
    if static_by_da is not None:
        m = panel.merge(static_by_da, on="DAUID", how="inner")
    elif month_only is not None:
        m = panel.merge(month_only, on="month", how="inner")
    else:
        m = panel.merge(dynamic_panel, on=["DAUID", "month"], how="inner")
    m = m.dropna(subset=layer_cols)
    b_r2, b_mae, ntr, nte = _train_eval(m, BASE_FEATS, test_months)
    f_r2, f_mae, _, _ = _train_eval(m, BASE_FEATS + layer_cols, test_months)
    print(f"\n--- {name} ---")
    print(f"   DAs matched: {m['DAUID'].nunique()} | train {ntr:,}/test {nte:,}")
    print(f"   Baseline R2: {b_r2:.4f}  + {name}: {f_r2:.4f}  Delta R2: {f_r2-b_r2:+.4f}")
    return {"layer": name, "delta_r2": f_r2 - b_r2}
