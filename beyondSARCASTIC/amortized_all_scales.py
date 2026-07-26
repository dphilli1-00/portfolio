"""
Combine the two most recent plots: the amortized-cost-vs-n-collects curve
(from amortized_per_collect.py) run at EVERY footprint scale from the
single 4m trihedral up through the full city, all overlaid on the same
axes -- same color per scale, solid = Ansys (flat), dashed = mine
(descending toward its runtime floor). Trihedral-vs-trihedral, city-vs-
city, laid out together so the pairs are directly comparable, and so it's
visually obvious that every pair crosses at the same n* (~6 collects),
regardless of scale.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0
P = 4497  # pulses/collect, locked-in real params

scenes = [
    ("4m trihedral (N=1)", 1),
    ("Target cluster (N=5)", 5),
    ("City block (N=25)", 25),
    ("Small district (N=125)", 125),
    ("District (N=625)", 625),
    ("Large district (N=3,125)", 3125),
    ("Full city (N=20,000)", 20000),
]

n_plot = np.array([1, 1.5, 2, 3, 5, 6, 8, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000])
colors = cm.viridis(np.linspace(0.05, 0.9, len(scenes)))

fig, ax = plt.subplots(figsize=(11, 8))

for (label, N), color in zip(scenes, colors):
    ansys_per_collect = P * (T_FIXED + N * R_ANSYS)
    mine_floor = P * (T_FIXED + N * R_ASC)
    build_cost = N * B_WORST
    mine_curve = build_cost / n_plot + mine_floor

    ax.plot(n_plot, [ansys_per_collect] * len(n_plot), color=color, linewidth=2,
            linestyle='-', label=f'{label} -- Ansys')
    ax.plot(n_plot, mine_curve, color=color, linewidth=2, linestyle='--', marker='o',
            markersize=3, label=f'{label} -- mine')

n_star = B_WORST / (P * (R_ANSYS - R_ASC))
ax.axvline(n_star, color='black', linestyle=':', linewidth=1.5, alpha=0.6,
           label=f'n* = {n_star:.2f} (same for every scale)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of collects run against the same built library (log)')
ax.set_ylabel('Amortized compute time PER COLLECT (sec, log)')
ax.set_title('Amortized cost per collect vs number of experiments, all footprint scales\n'
             'solid = Ansys (flat per scale), dashed = mine (descends to runtime floor)\n'
             'real params: 2m res, X-band, satellite @ 600km')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, which='both', alpha=0.25)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/amortized_all_scales.png", dpi=150)
print("Saved amortized_all_scales.png")
