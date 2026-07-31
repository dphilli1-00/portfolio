"""Display order1 / order2 / order3 side by side, SBR (ray-traced ground
truth) on top, box-projected point-ASC on the bottom. Run
run_order3_levels.py first to produce order3_levels_images.npz.
"""
import numpy as np
import matplotlib.pyplot as plt

d = np.load('order3_levels_images.npz')
grid = d['grid']
extent = [grid.min(), grid.max(), grid.min(), grid.max()]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
rows = [('sbr', 'SBR (ray-traced)'), ('asc', 'point-ASC (box-projected)')]
cols = ['order1', 'order2', 'order3']

for r, (prefix, row_label) in enumerate(rows):
    for c, level in enumerate(cols):
        img = d[f'{prefix}_{level}']
        mag = np.abs(img)
        mag_db = 20 * np.log10(mag / mag.max() + 1e-12) if mag.max() > 0 else np.full_like(mag, -30.0)
        ax = axes[r, c]
        im = ax.imshow(mag_db, extent=extent, origin='lower', vmin=-30, vmax=0, cmap='viridis')
        ax.set_title(f'{row_label}\n{level}')
        ax.set_xlabel('cross-range x (m)')
        ax.set_ylabel('range y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('order3_levels.png', dpi=120)
print("saved order3_levels.png")
plt.show()
