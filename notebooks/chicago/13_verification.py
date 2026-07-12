"""13_verification.py — Chicago verification parity with London (Session 21).

1. Cluster-bootstrap 95% CIs (by tract, B=1000) for the quoted Chicago numbers:
   baseline R2, full-fusion Delta R2 (SVI+demo+mental+weather, the configuration the
   thesis quotes: +0.0021), per-type robbery +CTA (+0.025) and weapons +weather (+0.092).
2. Leak assertion for the CTA join: in the leaky pattern (London scripts 30/31) the
   dynamic feature is zero whenever the month has no crime; in a clean join many
   zero-crime tract-months still carry ridership. Reports the share.
3. Static-only: SVI-only model (no crime history), Chicago analogue of London script 34.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from _fusion import load_panel_with_features, BASE_FEATS
import layers as L

B = 1000
RF = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
ROOT = Path(__file__).resolve().parents[2]


def cluster_ci(test_df, pb, pf=None, seed=42):
    y = test_df["crime_count"].values
    codes, _ = pd.factorize(test_df["GEOID"].values)
    n = codes.max() + 1
    gidx = [np.flatnonzero(codes == g) for g in range(n)]
    rng = np.random.default_rng(seed)
    r2b = np.empty(B)
    d = np.empty(B) if pf is not None else None
    for b in range(B):
        idx = np.concatenate([gidx[g] for g in rng.integers(0, n, n)])
        r2b[b] = r2_score(y[idx], pb[idx])
        if pf is not None:
            d[b] = r2_score(y[idx], pf[idx]) - r2b[b]
    out = {"r2": r2_score(y, pb), "r2_lo": np.percentile(r2b, 2.5), "r2_hi": np.percentile(r2b, 97.5)}
    if pf is not None:
        out.update({"delta": r2_score(y, pf) - out["r2"],
                    "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5)})
    return out


def fit_pair(df, months, base, extra):
    d = df.dropna(subset=base + extra + ["crime_count"])
    tmv = months[-6:]
    tr, te = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)].reset_index(drop=True)
    rf1 = RandomForestRegressor(**RF); rf1.fit(tr[base], tr["crime_count"])
    pb = rf1.predict(te[base])
    pf = None
    if extra:
        rf2 = RandomForestRegressor(**RF); rf2.fit(tr[base + extra], tr["crime_count"])
        pf = rf2.predict(te[base + extra])
    return te, pb, pf


print("=" * 96)
print(f"CHICAGO VERIFICATION (cluster bootstrap by tract, B={B})")
print("=" * 96)

panel, months = load_panel_with_features()

# 1a. baseline
te, pb, _ = fit_pair(panel, months, BASE_FEATS, [])
ci = cluster_ci(te, pb)
print(f"Baseline R2 = {ci['r2']:.4f}  [{ci['r2_lo']:.4f}, {ci['r2_hi']:.4f}]")

# 1b. full fusion (quoted spec: SVI + demographics + mental health + weather, no POI)
df = panel.copy(); cols = []
for kind, tbl, c in [L.svi(), L.demographics(), L.mental_health()]:
    df = df.merge(tbl, on="GEOID", how="inner"); cols += c
_, wagg, wcols = L.weather()
df = df.merge(wagg, on="month", how="inner"); cols += wcols
df = df.dropna(subset=cols)
te, pb, pf = fit_pair(df, months, BASE_FEATS, cols)
ci = cluster_ci(te, pb, pf)
print(f"Full fusion  dR2 = {ci['delta']:+.4f}  [{ci['d_lo']:+.4f}, {ci['d_hi']:+.4f}]  (base {ci['r2']:.4f})")

# static-only: SVI features alone, no history
d = df.dropna(subset=cols + ["crime_count"])
tmv = months[-6:]
tr, te2 = d[~d["month"].isin(tmv)], d[d["month"].isin(tmv)]
svi_only = [c for c in L.svi()[2]]
rf = RandomForestRegressor(**RF); rf.fit(tr[svi_only], tr["crime_count"])
print(f"SVI-only (no history) R2 = {r2_score(te2['crime_count'], rf.predict(te2[svi_only])):.4f}"
      f"   | history baseline on same panel = {ci['r2']:.4f}")

# 2. per-type CIs (exact 12_per_type spec: full 11 features, active-tract grid)
cnt = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_type.parquet")
active = pd.read_parquet(ROOT / "data/processed/chicago/tract_month_panel.parquet")["GEOID"].unique()
all_months = sorted(cnt["month"].unique())
midx = {m: i for i, m in enumerate(all_months)}


def pt_panel(ct):
    sub = cnt[cnt["primary_type"] == ct]
    grid = pd.MultiIndex.from_product([active, all_months], names=["GEOID", "month"])
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
    return p


# robbery + CTA (inner join on tract-months with CTA data, as in 12_per_type)
_, cta, cta_cols = L.cta()
p = pt_panel("ROBBERY").merge(cta, on=["GEOID", "month"], how="inner").dropna(subset=cta_cols)
te, pb, pf = fit_pair(p, all_months, BASE_FEATS, cta_cols)
ci = cluster_ci(te, pb, pf)
print(f"Robbery +CTA dR2 = {ci['delta']:+.4f}  [{ci['d_lo']:+.4f}, {ci['d_hi']:+.4f}]  (base {ci['r2']:.3f})")

# weapons + weather
p = pt_panel("WEAPONS VIOLATION").merge(wagg, on="month", how="inner").dropna(subset=wcols)
te, pb, pf = fit_pair(p, all_months, BASE_FEATS, wcols)
ci = cluster_ci(te, pb, pf)
print(f"Weapons +Weather dR2 = {ci['delta']:+.4f}  [{ci['d_lo']:+.4f}, {ci['d_hi']:+.4f}]  (base {ci['r2']:.3f})")

# 3. leak assertion on the CTA join used above
pr = pt_panel("ROBBERY").merge(cta, on=["GEOID", "month"], how="inner")
zero_crime = pr[pr["crime_count"] == 0]
share_nonzero_feature = float((zero_crime["cta_rides"] > 0).mean())
print(f"\nLEAK ASSERTION: among zero-robbery tract-months in the joined panel, "
      f"{share_nonzero_feature:.1%} carry non-zero CTA ridership")
print("(a leaky join, as in London scripts 30/31, would show 0.0% here)")
