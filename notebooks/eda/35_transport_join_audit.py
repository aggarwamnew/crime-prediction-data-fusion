"""35_transport_join_audit.py — Audit of the per-type transport join (scripts 30/31).

The bootstrap CI work (script 33) found that the headline weapons + station-taps lift
(+0.1371) could not be reproduced with a direct join of the monthly ridership table.
Code inspection shows why: scripts 30 and 31 build their per-type ridership lookup from
`ct_merged`, which contains only the (LSOA, month) pairs where the crime type OCCURRED.
After reindexing to the full LSOA x month grid, all zero-crime months receive ridership 0,
even at major stations, so the transport feature encodes "was there >=1 crime this month"
(target leakage). For sparse crime types this inflates Delta R2 dramatically.

This script runs the A/B for both transport sources across the five crime types quoted
in the thesis: (a) LEAKY join (verbatim reproduction of scripts 30/31) and (b) CORRECT
join (ridership taken from the full monthly table for every month, zero-filled only for
LSOAs genuinely without stations). Correct-join deltas also get cluster-bootstrap CIs.

Output: data/processed/london/transport_join_audit.csv
"""
import sys
sys.path.insert(0, '/Users/mohitaggarwalpty/Documents/AIProjects/Thesis')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT

B = 1000
db = ThesisDB()

ride = pd.read_csv(PROJECT_ROOT / "data/processed/london/station_ridership_monthly.csv")
bike = pd.read_csv(PROJECT_ROOT / "data/processed/london/santander_monthly.csv")
SOURCES = {
    'station_taps': (ride, ['ridership_total', 'station_count', 'ridership_per_station']),
    'santander': (bike, ['bike_total', 'bike_stations', 'bike_per_station']),
}
CRIME_TYPES = ['Possession of weapons', 'Bicycle theft', 'Drugs', 'Burglary', 'Criminal damage and arson']
RF = dict(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
LAGS = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_12', 'month_sin', 'month_cos']


def build_grid(ct_merged, months, lsoas):
    grid = pd.MultiIndex.from_product([lsoas, months], names=['lsoa_code', 'month'])
    df = ct_merged[['lsoa_code', 'month', 'crime_count']].set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
    df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
    df['rolling_mean_3'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_12'] = df.groupby('lsoa_code')['crime_count'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
    return df


def fit_delta(train, test, feats):
    rf_b = RandomForestRegressor(**RF)
    rf_b.fit(train[LAGS], train['crime_count'])
    pb = rf_b.predict(test[LAGS])
    rf_f = RandomForestRegressor(**RF)
    rf_f.fit(train[LAGS + feats], train['crime_count'])
    pf = rf_f.predict(test[LAGS + feats])
    return pb, pf


def cluster_delta_ci(test, pb, pf, seed=42):
    y = test['crime_count'].values
    codes, _ = pd.factorize(test['lsoa_code'].values)
    n = codes.max() + 1
    gidx = [np.flatnonzero(codes == g) for g in range(n)]
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for b in range(B):
        idx = np.concatenate([gidx[g] for g in rng.integers(0, n, n)])
        d[b] = r2_score(y[idx], pf[idx]) - r2_score(y[idx], pb[idx])
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


rows = []
for src_name, (tbl, feats) in SOURCES.items():
    for ct in CRIME_TYPES:
        q = ct.replace("'", "''")
        cm = db.query(f"SELECT lsoa_code, month, COUNT(*) as crime_count FROM crime_clean WHERE crime_type = '{q}' GROUP BY lsoa_code, month")
        ct_merged = cm.merge(tbl[['lsoa_code', 'month'] + feats], on=['lsoa_code', 'month'], how='left')
        ct_merged[feats] = ct_merged[feats].fillna(0)
        totals = ct_merged.groupby('lsoa_code')['crime_count'].sum()
        active = totals[totals >= 12].index
        ct_merged = ct_merged[ct_merged['lsoa_code'].isin(active)]
        months = sorted(ct_merged['month'].unique())
        lsoas = sorted(ct_merged['lsoa_code'].unique())
        base = build_grid(ct_merged, months, lsoas)

        # (a) LEAKY join: lookup built from crime-month rows only (verbatim scripts 30/31)
        leaky_lk = ct_merged[['lsoa_code', 'month'] + feats].drop_duplicates(['lsoa_code', 'month']).set_index(['lsoa_code', 'month'])
        df_l = base.join(leaky_lk, on=['lsoa_code', 'month'], how='left')
        df_l[feats] = df_l[feats].fillna(0)

        # (b) CORRECT join: full monthly table for every month
        df_c = base.merge(tbl[['lsoa_code', 'month'] + feats], on=['lsoa_code', 'month'], how='left')
        df_c[feats] = df_c[feats].fillna(0)

        out = {'source': src_name, 'crime_type': ct, 'n_lsoas': len(active)}
        for tag, d in [('leaky', df_l), ('correct', df_c)]:
            dm = d.dropna()
            test_months = months[-6:]
            train = dm[~dm['month'].isin(test_months)]
            test = dm[dm['month'].isin(test_months)].reset_index(drop=True)
            pb, pf = fit_delta(train, test, feats)
            r2b = r2_score(test['crime_count'], pb)
            delta = r2_score(test['crime_count'], pf) - r2b
            out[f'r2_base_{tag}'] = r2b
            out[f'delta_{tag}'] = delta
            if tag == 'correct':
                lo, hi = cluster_delta_ci(test, pb, pf)
                out['delta_correct_lo'] = lo
                out['delta_correct_hi'] = hi
        # diagnostics: how often the transport feature is non-zero in each variant
        out['nz_leaky'] = float((df_l[feats[0]] > 0).mean())
        out['nz_correct'] = float((df_c[feats[0]] > 0).mean())
        rows.append(out)
        print(f"{src_name:13s} {ct:28s} leaky={out['delta_leaky']:+.4f}  correct={out['delta_correct']:+.4f} "
              f"[{out['delta_correct_lo']:+.4f},{out['delta_correct_hi']:+.4f}]  nz(leaky)={out['nz_leaky']:.2f} nz(correct)={out['nz_correct']:.2f}", flush=True)

db.close()
res = pd.DataFrame(rows)
out_path = PROJECT_ROOT / "data/processed/london/transport_join_audit.csv"
res.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
