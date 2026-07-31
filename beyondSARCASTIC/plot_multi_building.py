"""Plot the multi-building validation images (run run_multi_building_validation.py
first to produce multi_building_images.npz)."""
import numpy as np
import matplotlib.pyplot as plt

d = np.load('multi_building_images.npz')
grid = d['grid']
extent = [grid.min(), grid.max(), grid.min(), grid.max()]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
pairs = [('SBR (order2 only)', d['img_sbr2']), ('point-ASC (order2 only)', d['img_asc2']),
         ('SBR (order1+order2)', d['img_sbr_full']), ('point-ASC (order1+order2)', d['img_asc_full'])]
for ax, (name, img) in zip(axes.flat, pairs):
    mag = np.abs(img)
    mag_db = 20 * np.log10(mag / mag.max() + 1e-12)
    im = ax.imshow(mag_db, extent=extent, origin='lower', vmin=-30, vmax=0, cmap='viridis')
    ax.set_title(name); ax.set_xlabel('cross-range x (m)'); ax.set_ylabel('range y (m)')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('multi_building_images.png', dpi=120)
print("saved multi_building_images.png")
plt.show()
