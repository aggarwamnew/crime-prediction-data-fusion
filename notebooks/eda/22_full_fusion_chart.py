"""22_full_fusion_chart.py — Updated per-crime-type chart with 4-tier comparison."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Results from script 21 (per-crime-type full fusion, housing=tertile bands)
data = [
    ('Possession of weapons', 0.0367, +0.0407, +0.0379, +0.0007),
    ('Drugs', 0.3581, +0.0231, +0.0261, +0.0041),
    ('Criminal damage & arson', 0.2797, +0.0108, +0.0084, -0.0011),
    ('Public order', 0.5854, +0.0080, +0.0085, +0.0001),
    ('Shoplifting', 0.7501, +0.0080, +0.0076, -0.0017),
    ('Bicycle theft', 0.3937, +0.0050, +0.0046, -0.0015),
    ('Robbery', 0.6219, +0.0062, +0.0050, -0.0007),
    ('Burglary', 0.3157, +0.0047, +0.0042, +0.0002),
    ('Violence & sexual', 0.7601, +0.0037, +0.0037, -0.0001),
    ('Vehicle crime', 0.2812, +0.0037, +0.0034, -0.0004),
    ('Other crime', 0.0698, +0.0070, +0.0015, -0.0018),
    ('Other theft', 0.9128, +0.0018, +0.0017, +0.0003),
    ('Anti-social behaviour', 0.6838, -0.0018, -0.0015, +0.0000),
    ('Theft from person', 0.7832, -0.0312, -0.0242, +0.0036),
]

names = [d[0] for d in data]
d_ctx = [d[2] for d in data]
d_full = [d[3] for d in data]
d_ss_marginal = [d[4] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})

# Left: Contextual vs Full fusion
y = np.arange(len(names))
w = 0.35
bars1 = ax1.barh(y - w/2, d_ctx, w, label='+ Contextual only', color='#3b82f6', alpha=0.85)
bars2 = ax1.barh(y + w/2, d_full, w, label='+ Full (all tiers)', color='#8b5cf6', alpha=0.85)
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=10)
ax1.set_xlabel('Δ R² (improvement over crime-only baseline)', fontsize=11)
ax1.set_title('Per-Crime-Type: Contextual vs Full Fusion', fontweight='bold', fontsize=13)
ax1.legend(loc='lower right', fontsize=10)
ax1.invert_yaxis()
ax1.set_xlim(-0.035, 0.055)

# Right: Marginal SS contribution
colors = ['#10b981' if v > 0.001 else '#ef4444' if v < -0.001 else '#9ca3af' for v in d_ss_marginal]
ax2.barh(y, d_ss_marginal, 0.6, color=colors, alpha=0.85)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel('Δ R² (SS marginal)', fontsize=11)
ax2.set_title('Socio-Structural\nMarginal Lift', fontweight='bold', fontsize=12)
ax2.invert_yaxis()

plt.tight_layout()
outpath = str(Path(__file__).resolve().parents[2] / 'reports' / 'figures' / 'per_type' / '04_full_fusion_by_type.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {outpath}")
