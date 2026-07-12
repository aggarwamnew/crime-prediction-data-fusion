"""36_uncertainty_decomposition.py — Uncertainty decomposition as a layer-selection guide.

Tests whether the baseline model's own uncertainty signature, computed BEFORE any data
fusion, could have predicted which crime types would benefit from supplementary layers
(and which would not), replacing an exhaustive 14-type x 7-layer ablation with a single
diagnostic pass. Direction suggested by external review feedback: use uncertainty
quantification to identify what KIND of knowledge gap a model has, and let that guide
the search for additional layers.

Two pre-fusion diagnostics per crime type (both use only crime history — no layers):

  1. REDUCIBLE-SURPLUS SHARE (residual-based).
     Aleatoric variance floor per test row: A_i = phi_type * mu_hat_i, where phi_type is
     the within-LSOA variance-to-mean ratio of the RAW counts (model-independent
     overdispersion) and mu_hat_i is the baseline prediction. Any test MSE above the mean
     floor is in-principle reducible (epistemic):
         surplus_share = max(0, MSE - mean(A)) / MSE.

  2. ENSEMBLE-SPREAD SHARE (model-internal).
     Epistemic proxy E_i = variance of per-tree predictions in the Random Forest;
         spread_share = mean(E) / (mean(E) + mean(A)).
     Cruder (bagged trees are correlated) but computable from the fitted model alone.

Validation: Spearman rank-correlation of each diagnostic against the REALISED best
single-layer Delta R2 per crime type from the ablation study (corrected Table 5.6).
A strong positive correlation validates uncertainty-guided layer selection
retrospectively on this study's own data.

Outputs: data/processed/london/uncertainty_decomposition.csv
         reports/figures/uncertainty/01_diagnostic_vs_realised.png
         reports/figures/uncertainty/02_epistemic_share_by_type.png
"""
import sys
sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

db = ThesisDB()
FIG_DIR = PROJECT_ROOT / "reports/figures/uncertainty"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Realised best single-layer Delta R2 per crime type (corrected Table 5.6, thesis Results).
REALISED = {
    'Possession of weapons': 0.049, 'Drugs': 0.021, 'Theft from the person': 0.014,
    'Bicycle theft': 0.013, 'Burglary': 0.008, 'Criminal damage and arson': 0.007,
    'Public order': 0.006, 'Robbery': 0.006, 'Vehicle crime': 0.005,
    'Shoplifting': 0.004, 'Violence and sexual offences': 0.003,
    'Other theft': 0.002, 'Anti-social behaviour': 0.001, 'Other crime': 0.001,
}

RF_PT = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
RF_AGG = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)


def build_panel(monthly, min_crimes, per_type):
    """Panel + baseline features, matching the ablation harnesses exactly."""
    totals = monthly.groupby('lsoa_code')['crime_count'].sum()
    active = totals[totals >= min_crimes].index
    m = monthly[monthly['lsoa_code'].isin(active)]
    months = sorted(m['month'].unique())
    lsoas = sorted(m['lsoa_code'].unique())
    grid = pd.MultiIndex.from_product([lsoas, months], names=['lsoa_code', 'month'])
    df = m.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    lags = [1, 3, 6, 12] if per_type else [1, 2, 3, 6, 12]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    windows = [3, 12] if per_type else [3, 6, 12]
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    feats = [f'lag_{l}' for l in lags] + [f'rolling_mean_{w}' for w in windows] + ['month_sin', 'month_cos']
    if not per_type:
        df['time_idx'] = df['month'].map({mm: i for i, mm in enumerate(months)})
        feats.append('time_idx')
    return df, months, feats


def phi_from_raw(df):
    """Model-independent overdispersion: median within-LSOA var/mean of raw counts."""
    g = df.groupby('lsoa_code')['crime_count']
    ratio = (g.var() / g.mean()).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median())


def diagnose(df, months, feats, rf_params):
    """Fit baseline once; return the two pre-fusion uncertainty diagnostics."""
    d = df.dropna(subset=feats + ['crime_count'])
    test_months = months[-6:]
    train = d[~d['month'].isin(test_months)]
    test = d[d['month'].isin(test_months)].reset_index(drop=True)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(train[feats], train['crime_count'])
    mu = rf.predict(test[feats])
    y = test['crime_count'].values

    phi = phi_from_raw(df)
    A = phi * np.clip(mu, 1e-6, None)          # aleatoric variance floor per row
    mse = float(np.mean((y - mu) ** 2))
    surplus_share = max(0.0, mse - float(A.mean())) / mse

    tree_preds = np.stack([t.predict(test[feats].values) for t in rf.estimators_])
    E = tree_preds.var(axis=0)                  # ensemble spread per row
    spread_share = float(E.mean()) / (float(E.mean()) + float(A.mean()))

    return {'phi': phi, 'r2_base': r2_score(y, mu), 'mse': mse,
            'aleatoric_mean': float(A.mean()), 'surplus_share': surplus_share,
            'spread_mean': float(E.mean()), 'spread_share': spread_share,
            'n_test': len(test)}


print("=" * 100)
print("UNCERTAINTY DECOMPOSITION — pre-fusion diagnostics vs realised ablation gains")
print("=" * 100)

# ── aggregate model (context row) ──
monthly_all = db.query("SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean GROUP BY lsoa_code, month")
df, months, feats = build_panel(monthly_all, 36, per_type=False)
agg = diagnose(df, months, feats, RF_AGG)
print(f"\nAGGREGATE: phi={agg['phi']:.2f}  R2={agg['r2_base']:.4f}  "
      f"surplus_share={agg['surplus_share']:.3f}  spread_share={agg['spread_share']:.3f}")

# ── per-type diagnostics ──
rows = []
for ct in REALISED:
    q = ct.replace("'", "''")
    cm = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{q}' GROUP BY lsoa_code, month")
    df, months, feats = build_panel(cm, 12, per_type=True)
    res = diagnose(df, months, feats, RF_PT)
    res['crime_type'] = ct
    res['realised_best_delta'] = REALISED[ct]
    rows.append(res)
    print(f"{ct:32s} phi={res['phi']:5.2f}  R2={res['r2_base']:+.3f}  "
          f"surplus={res['surplus_share']:.3f}  spread={res['spread_share']:.3f}  realised={REALISED[ct]:+.3f}", flush=True)

db.close()
res = pd.DataFrame(rows)

rho_s, p_s = spearmanr(res['surplus_share'], res['realised_best_delta'])
rho_e, p_e = spearmanr(res['spread_share'], res['realised_best_delta'])
print("\n" + "=" * 100)
print(f"Spearman(surplus_share, realised best Delta R2) = {rho_s:+.3f}  (p = {p_s:.4f})")
print(f"Spearman(spread_share,  realised best Delta R2) = {rho_e:+.3f}  (p = {p_e:.4f})")

out = PROJECT_ROOT / "data/processed/london/uncertainty_decomposition.csv"
res.to_csv(out, index=False)
print(f"Saved -> {out}")

# ── figures ──
SHORT = {'Possession of weapons': 'Weapons', 'Theft from the person': 'Theft person',
         'Criminal damage and arson': 'Crim. damage', 'Violence and sexual offences': 'Violence',
         'Anti-social behaviour': 'ASB'}
res['label'] = res['crime_type'].map(lambda c: SHORT.get(c, c))

fig, ax = plt.subplots(figsize=(9.5, 7))
ax.scatter(res['surplus_share'], res['realised_best_delta'], s=70, color='#3b82f6', alpha=0.9, zorder=3)
for _, r in res.iterrows():
    ax.annotate(r['label'], (r['surplus_share'], r['realised_best_delta']),
                textcoords='offset points', xytext=(7, 4), fontsize=9)
ax.set_xlabel('Pre-fusion diagnostic: reducible-surplus share of test MSE')
ax.set_ylabel('Realised best single-layer $\\Delta R^2$ (ablation)')
ax.set_title(f'Pre-fusion reducible-error share versus realised fusion gain\n'
             f'Spearman $\\rho$ = {rho_s:+.2f} (p = {p_s:.3f})', loc='left')
ax.grid(alpha=0.3)
fig.tight_layout()
f1 = FIG_DIR / "01_diagnostic_vs_realised.png"
fig.savefig(f1, dpi=150, bbox_inches='tight')
print(f"saved {f1}")

order = res.sort_values('surplus_share', ascending=True)
helped = order['realised_best_delta'] >= 0.005
fig2, ax = plt.subplots(figsize=(9.5, 7))
colors = ['#8b5cf6' if h else '#cbd5e1' for h in helped]
ax.barh(range(len(order)), order['surplus_share'], color=colors, alpha=0.95)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order['label'], fontsize=10)
ax.set_xlabel('Reducible-surplus share of test MSE (pre-fusion)')
ax.set_title('Epistemic headroom by crime type, computed before any fusion\n'
             '(purple = types where the ablation later found $\\Delta R^2 \\geq 0.005$)', loc='left')
ax.grid(axis='x', alpha=0.3)
fig2.tight_layout()
f2 = FIG_DIR / "02_epistemic_share_by_type.png"
fig2.savefig(f2, dpi=150, bbox_inches='tight')
print(f"saved {f2}")
