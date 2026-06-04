from pathlib import Path
"""23_per_layer_chart.py — Per-crime-type chart with all 7 data layers (IMD, Weather, Demo, POI, Housing, Temporal, SS)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 5.4 (per-layer per-type Δ R²)
# Order: IMD, Weather, Demo, POI, Housing, Temporal, SS
crime_types = [
    'Possession of weapons',
    'Drugs',
    'Bicycle theft',
    'Burglary',
    'Criminal damage',
    'Public order',
    'Robbery',
    'Shoplifting',
    'Theft from person',
    'Vehicle crime',
    'Violence',
    'Other theft',
    'ASB',
    'Other crime',
]

data = {
    'IMD':     [+.018, +.020, +.008, +.006, +.006, +.004, +.004, +.003, -.003, +.003, +.002, +.002, +.001, +.000],
    'Weather': [-.005, +.011, +.013, +.002, +.002, +.003, -.002, +.003, -.002, +.002, -.000, +.002, -.002, +.001],
    'Demo':    [+.021, +.013, +.011, +.008, +.003, +.005, +.006, +.004, -.004, +.005, +.001, +.002, +.001, +.000],
    'POI':     [+.049, +.021, +.001, +.007, +.007, +.006, +.005, +.002, +.003, -.004, +.001, +.001, +.001, +.000],
    'Housing': [+.005, +.001, -.001, +.000, +.000, +.001, -.001, +.002, +.004, -.000, +.000, -.000, +.001, -.002],
    'Temporal':[+.004, +.003, +.001, -.002, +.000, +.001, +.002, +.001, +.014, +.001, -.000, +.001, +.000, -.001],
    'SS':      [-.006, +.016, +.006, +.004, +.004, +.003, +.004, -.000, +.001, -.000, +.003, +.001, -.001, -.000],
}

layers = list(data.keys())
colors = ['#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1']
n_types = len(crime_types)
n_layers = len(layers)

fig, ax = plt.subplots(figsize=(16, 10))

bar_height = 0.11
y_positions = np.arange(n_types)

for i, (layer, color) in enumerate(zip(layers, colors)):
    offsets = y_positions - (n_layers - 1) * bar_height / 2 + i * bar_height
    ax.barh(offsets, data[layer], bar_height, label=f'+ {layer}', color=color, alpha=0.85, edgecolor='white', linewidth=0.3)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticks(y_positions)
ax.set_yticklabels(crime_types, fontsize=11)
ax.set_xlabel('Δ R² (improvement over crime-only baseline)', fontsize=12)
ax.set_title('Data Fusion Impact by Crime Type:\nIMD vs Weather vs Demographics vs POIs vs Housing vs Temporal vs Socio-Structural',
             fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.invert_yaxis()
ax.set_xlim(-0.012, 0.055)
ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
outpath = str(Path(__file__).resolve().parents[2] / 'reports' / 'figures' / 'per_type' / '06_all_layers_with_ss.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {outpath}")
