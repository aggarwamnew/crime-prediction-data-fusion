"""14_uq_decomposition.py — Chicago replication of the London uncertainty-decomposition
test (London script 36): can pre-fusion diagnostics rank where fusion will pay?

Same two diagnostics (reducible-surplus share over an aleatoric floor phi*mu with
phi from raw within-tract dispersion; RF ensemble-spread share), compared against the
realised best single-layer Delta R2 per crime type from data/processed/chicago/
per_type_fusion.csv. London found rule-out valid, rule-in null (rho = +0.05).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

RF = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
ROOT = Path(__file__).resolve().parents[2]
BASE = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month_sin", "month_cos", "time_idx"]

cnt = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_type.parquet")
active = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_panel.parquet")["GEOID"].unique()
months = sorted(cnt["month"].unique())
midx = {m: i for i, m in enumerate(months)}

pt = pd.read_csv(ROOT / "data/processed/chicago/per_type_fusion.csv", index_col=0)
layer_cols = [c for c in pt.columns if c != "base_R2"]
realised = pt[layer_cols].max(axis=1)          # best single-layer delta per type

rows = []
for ct in realised.index:
    sub = cnt[cnt["primary_type"] == ct]
    grid = pd.MultiIndex.from_product([active, months], names=["GEOID", "month"])
    p = (sub.groupby(["GEOID", "month"])["n"].sum().reindex(grid, fill_value=0)
         .reset_index(name="crime_count")).sort_values(["GEOID", "month"])
    for lag in [1, 2, 3, 6, 12]:
        p[f"lag_{lag}"] = p.groupby("GEOID")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        p[f"rolling_mean_{w}"] = p.groupby("GEOID")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    mn = p["month"].str[-2:].astype(int)
    p["month_sin"] = np.sin(2 * np.pi * mn / 12)
    p["month_cos"] = np.cos(2 * np.pi * mn / 12)
    p["time_idx"] = p["month"].map(midx)

    g = p.groupby("GEOID")["crime_count"]
    phi = float((g.var() / g.mean()).replace([np.inf, -np.inf], np.nan).dropna().median())

    d = p.dropna(subset=BASE + ["crime_count"])
    tmv = months[-6:]
    tr, te = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)].reset_index(drop=True)
    rf = RandomForestRegressor(**RF)
    rf.fit(tr[BASE], tr["crime_count"])
    mu = rf.predict(te[BASE])
    y = te["crime_count"].values
    A = phi * np.clip(mu, 1e-6, None)
    mse = float(np.mean((y - mu) ** 2))
    surplus = max(0.0, mse - float(A.mean())) / mse
    tree_preds = np.stack([t.predict(te[BASE].values) for t in rf.estimators_])
    E = tree_preds.var(axis=0)
    spread = float(E.mean()) / (float(E.mean()) + float(A.mean()))
    rows.append({"crime_type": ct, "phi": phi, "r2_base": r2_score(y, mu),
                 "surplus_share": surplus, "spread_share": spread,
                 "realised_best_delta": float(realised[ct])})
    print(f"{ct:24s} phi={phi:5.2f}  R2={rows[-1]['r2_base']:+.3f}  "
          f"surplus={surplus:.3f}  spread={spread:.3f}  realised={realised[ct]:+.3f}", flush=True)

res = pd.DataFrame(rows)
rho_s, p_s = spearmanr(res["surplus_share"], res["realised_best_delta"])
rho_e, p_e = spearmanr(res["spread_share"], res["realised_best_delta"])
print(f"\nSpearman(surplus, realised) = {rho_s:+.3f} (p={p_s:.3f})")
print(f"Spearman(spread,  realised) = {rho_e:+.3f} (p={p_e:.3f})")
res.to_csv(ROOT / "data/processed/chicago/uncertainty_decomposition.csv", index=False)
print("Saved -> data/processed/chicago/uncertainty_decomposition.csv")
