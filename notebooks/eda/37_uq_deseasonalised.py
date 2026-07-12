"""37_uq_deseasonalised.py — Repairing the uncertainty diagnostic (London + Chicago).

The Chicago replication (chicago/14) falsified the naive rule-out reading of the
uncertainty diagnostic: a floor estimated from RAW within-unit dispersion absorbs
predictable seasonal variance, so seasonal knowledge gaps (Chicago weapons <-> weather)
masquerade as noise. This script re-estimates the aleatoric floor from the SEASONAL
DIFFERENCE of each unit's series: if y_t = s(t) + eps_t with Var(eps) = phi*mu, then
Var(y_t - y_{t-12}) = 2*phi*mu, so
    phi_deseasonalised = median_units[ Var(y_t - y_{t-12}) / (2 * mean(y)) ].
Level and seasonality cancel in the difference, so predictable structure no longer
inflates the floor. The rule-out test is then re-run in BOTH cities.
"""
import sys
sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

RF_LON = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
RF_CHI = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)

REALISED_LON = {
    'Possession of weapons': 0.049, 'Drugs': 0.021, 'Theft from the person': 0.014,
    'Bicycle theft': 0.013, 'Burglary': 0.008, 'Criminal damage and arson': 0.007,
    'Public order': 0.006, 'Robbery': 0.006, 'Vehicle crime': 0.005,
    'Shoplifting': 0.004, 'Violence and sexual offences': 0.003,
    'Other theft': 0.002, 'Anti-social behaviour': 0.001, 'Other crime': 0.001,
}


def phi_deseason(panel, unit_col):
    d12 = panel.sort_values([unit_col, 'month']).groupby(unit_col)['crime_count'].diff(12)
    tmp = panel.assign(_d=d12)
    var_d = tmp.groupby(unit_col)['_d'].var()
    mean_y = panel.groupby(unit_col)['crime_count'].mean()
    ratio = (var_d / (2 * mean_y)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median())


def surplus(panel, months, feats, unit_col, rf_params):
    d = panel.dropna(subset=feats + ['crime_count'])
    tmv = months[-6:]
    tr, te = d[~d['month'].isin(tmv)], d[d['month'].isin(tmv)].reset_index(drop=True)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(tr[feats], tr['crime_count'])
    mu = rf.predict(te[feats])
    y = te['crime_count'].values
    phi = phi_deseason(panel, unit_col)
    A = phi * np.clip(mu, 1e-6, None)
    mse = float(np.mean((y - mu) ** 2))
    return phi, max(0.0, mse - float(A.mean())) / mse, r2_score(y, mu)


print("=" * 100)
print("DESEASONALISED-FLOOR UNCERTAINTY DIAGNOSTIC — London + Chicago rule-out retest")
print("=" * 100)

# ── London ──
db = ThesisDB()
rows_l = []
for ct, real in REALISED_LON.items():
    q = ct.replace("'", "''")
    cm = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{q}' GROUP BY lsoa_code, month")
    totals = cm.groupby('lsoa_code')['crime_count'].sum()
    act = totals[totals >= 12].index
    cm = cm[cm['lsoa_code'].isin(act)]
    months = sorted(cm['month'].unique())
    grid = pd.MultiIndex.from_product([sorted(act), months], names=['lsoa_code', 'month'])
    p = cm.set_index(['lsoa_code', 'month'])['crime_count'].reindex(grid, fill_value=0).reset_index()
    p = p.sort_values(['lsoa_code', 'month'])
    for lag in [1, 3, 6, 12]:
        p[f'lag_{lag}'] = p.groupby('lsoa_code')['crime_count'].shift(lag)
    p['rolling_mean_3'] = p.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    p['rolling_mean_12'] = p.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    mn = pd.to_datetime(p['month']).dt.month
    p['month_sin'] = np.sin(2 * np.pi * mn / 12)
    p['month_cos'] = np.cos(2 * np.pi * mn / 12)
    feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_12', 'month_sin', 'month_cos']
    phi, s, r2b = surplus(p, months, feats, 'lsoa_code', RF_LON)
    rows_l.append({'crime_type': ct, 'phi_d': phi, 'surplus_d': s, 'realised': real})
    print(f"LON {ct:32s} phi_d={phi:5.2f}  surplus_d={s:.3f}  realised={real:+.3f}", flush=True)
db.close()
rl = pd.DataFrame(rows_l)
rho_l, p_l = spearmanr(rl['surplus_d'], rl['realised'])

# ── Chicago ──
ROOT = PROJECT_ROOT
cnt = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_type.parquet")
active = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_panel.parquet")["GEOID"].unique()
months_c = sorted(cnt["month"].unique())
midx = {m: i for i, m in enumerate(months_c)}
pt = pd.read_csv(ROOT / "data/processed/chicago/per_type_fusion.csv", index_col=0)
realised_c = pt[[c for c in pt.columns if c != "base_R2"]].max(axis=1)

rows_c = []
for ct in realised_c.index:
    sub = cnt[cnt["primary_type"] == ct]
    grid = pd.MultiIndex.from_product([active, months_c], names=["GEOID", "month"])
    p = (sub.groupby(["GEOID", "month"])["n"].sum().reindex(grid, fill_value=0)
         .reset_index(name="crime_count")).sort_values(["GEOID", "month"])
    p = p.rename(columns={"GEOID": "unit"})
    for lag in [1, 2, 3, 6, 12]:
        p[f"lag_{lag}"] = p.groupby("unit")["crime_count"].shift(lag)
    for w in [3, 6, 12]:
        p[f"rolling_mean_{w}"] = p.groupby("unit")["crime_count"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    mn = p["month"].str[-2:].astype(int)
    p["month_sin"] = np.sin(2 * np.pi * mn / 12)
    p["month_cos"] = np.cos(2 * np.pi * mn / 12)
    p["time_idx"] = p["month"].map(midx)
    feats = [f"lag_{l}" for l in [1, 2, 3, 6, 12]] + ["rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
                                                      "month_sin", "month_cos", "time_idx"]
    phi, s, r2b = surplus(p, months_c, feats, "unit", RF_CHI)
    rows_c.append({'crime_type': ct, 'phi_d': phi, 'surplus_d': s, 'realised': float(realised_c[ct])})
    print(f"CHI {ct:32s} phi_d={phi:5.2f}  surplus_d={s:.3f}  realised={realised_c[ct]:+.3f}", flush=True)

rc = pd.DataFrame(rows_c)
rho_c, p_c = spearmanr(rc['surplus_d'], rc['realised'])

print("\n" + "=" * 100)
print(f"LONDON : Spearman(deseasonalised surplus, realised) = {rho_l:+.3f} (p={p_l:.3f})   [raw floor was +0.05]")
print(f"CHICAGO: Spearman(deseasonalised surplus, realised) = {rho_c:+.3f} (p={p_c:.3f})   [raw floor was -0.46]")
rl.assign(city='London').pipe(lambda a: pd.concat([a, rc.assign(city='Chicago')])).to_csv(
    ROOT / "data/processed/london/uq_deseasonalised.csv", index=False)
print("Saved -> data/processed/london/uq_deseasonalised.csv")
