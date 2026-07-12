"""
Cross-city comparison figures for the thesis (London / Chicago / Vancouver).
Outputs to reports/figures/cross_city/. Numbers from the committed replication results
(notebooks/chicago/RESULTS.md, notebooks/vancouver, data/cross_city_feasibility.md).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "reports/figures/cross_city"
OUT.mkdir(parents=True, exist_ok=True)

CITY_COLORS = {"London": "#3b82f6", "Chicago": "#8b5cf6", "Vancouver": "#10b981"}
plt.rcParams.update({"font.size": 12, "axes.spreadsheet": False} if False else {"font.size": 12})

# ---------------------------------------------------------------- Figure 1: baseline R2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.5]})

bl_labels = ["London\n(LSOA)", "Chicago\n(tract)", "Vancouver\n(DA, property)", "Vancouver\n(24 nbhd, all)"]
bl_vals = [0.943, 0.902, 0.925, 0.971]
bl_colors = ["#3b82f6", "#8b5cf6", "#10b981", "#34d399"]
bars = ax1.bar(range(4), bl_vals, color=bl_colors, alpha=0.9, width=0.65)
ax1.set_ylim(0.85, 1.0)
ax1.set_ylabel("Test $R^2$ (crime-only baseline)")
ax1.set_title("(a) Baseline prediction replicates across cities", fontsize=12, loc="left")
ax1.set_xticks(range(4)); ax1.set_xticklabels(bl_labels, fontsize=9)
for b, v in zip(bars, bl_vals):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------- Figure 1b: Delta R2 by layer x city
layers = ["Deprivation", "Demographics", "Mental\nhealth", "Weather", "POIs", "Education", "Household", "Housing", "Temporal", "Full\nfusion"]
london = [0.0005, 0.0007, 0.0003, 0.0025, 0.0004, 0.0003, 0.0005, 0.0006, 0.0001, 0.0020]
chicago = [0.0005, -0.0002, 0.0001, 0.0022, 0.0004, 0.0005, 0.0003, 0.0002, 0.0008, 0.0021]
vancouver = [-0.0001, 0.0001, np.nan, -0.0001, -0.0000, np.nan, np.nan, -0.0004, -0.0001, -0.0001]  # nan = not available

x = np.arange(len(layers)); w = 0.27
ax2.bar(x - w, london, w, label="London", color=CITY_COLORS["London"], alpha=0.9)
ax2.bar(x, chicago, w, label="Chicago", color=CITY_COLORS["Chicago"], alpha=0.9)
# plot vancouver, marking NaNs with a hatch-less gap + 'n/a'
van_plot = [0 if np.isnan(v) else v for v in vancouver]
ax2.bar(x + w, van_plot, w, label="Vancouver", color=CITY_COLORS["Vancouver"], alpha=0.9)
for i, v in enumerate(vancouver):
    if np.isnan(v):
        ax2.text(x[i] + w, 0.00005, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="gray")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Aggregate $\\Delta R^2$ vs baseline")
ax2.set_title("(b) Fusion lift is negligible everywhere; full fusion near-identical", fontsize=12, loc="left")
ax2.set_xticks(x); ax2.set_xticklabels(layers, fontsize=9)
ax2.legend(frameon=False)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
f1 = OUT / "01_crosscity_baseline_and_fusion.png"
fig.savefig(f1, dpi=150, bbox_inches="tight")
print(f"saved {f1}")

# ---------------------------------------------------------------- Figure 2: Chicago per-type best layer
import pandas as pd
pt_path = Path(__file__).resolve().parents[1] / "data/processed/chicago/per_type_fusion.csv"
if pt_path.exists():
    df = pd.read_csv(pt_path, index_col=0)
    layer_cols = [c for c in df.columns if c != "base_R2"]
    best_layer = df[layer_cols].idxmax(axis=1)
    best_val = df[layer_cols].max(axis=1)
    order = best_val.sort_values(ascending=True).index
    fig2, ax = plt.subplots(figsize=(11, 7))
    palette = {"SVI": "#ef4444", "Weather": "#f59e0b", "Demo": "#84cc16", "MentHlth": "#06b6d4",
               "CTA": "#8b5cf6", "Divvy": "#ec4899", "POI": "#3b82f6"}
    colors = [palette.get(best_layer[t], "#64748b") for t in order]
    ax.barh(range(len(order)), [best_val[t] for t in order], color=colors, alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t.title()}" for t in order], fontsize=10)
    for i, t in enumerate(order):
        ax.text(best_val[t] + 0.001, i, f"{best_layer[t]} ({best_val[t]:+.3f})", va="center", fontsize=9)
    ax.set_xlabel("Best single-layer $\\Delta R^2$ for the crime type")
    ax.set_title("Chicago: best supplementary layer per crime type\n(layers concentrate on distinct crime types, as in London)", loc="left", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(best_val) * 1.25)
    fig2.tight_layout()
    f2 = OUT / "02_chicago_pertype_best_layer.png"
    fig2.savefig(f2, dpi=150, bbox_inches="tight")
    print(f"saved {f2}")
else:
    print("per_type_fusion.csv not found; skipping figure 2")
