"""One-off diagnostic: decompose the closed-form ASC result into leg1
(direct) / leg2 (ground bounce) / leg3 (if enabled) and backproject each
SEPARATELY, at the SAME full resolution (Nyquist pulse count, proper
n_freq) as run_direct_to_facet_validation.py -- to find which leg
produces the full-width horizontal streak artifacts seen in the combined
500m/density-200 image (direct_to_facet_validation_result.png).

A small 150m/4-building test at full resolution showed NO streak in
either leg -- so this is checking whether it's leg-specific at the
scene size that DOES show it, not assuming leg2 is guilty.

Usage:
    python3 diagnose_streak_artifact.py --gpu
    python3 diagnose_streak_artifact.py --gpu --footprint 300 --density 200   # smaller/faster first pass
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import backproject, to_numpy, compute_ref_ranges, required_freq_samples

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--footprint', type=float, default=500.0)
ap.add_argument('--density', type=float, default=200.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dr-max', type=float, default=2.0)
ap.add_argument('--delta-cr', type=float, default=2.0)
ap.add_argument('--standoff', type=float, default=8000.0)
ap.add_argument('--altitude', type=float, default=3000.0)
ap.add_argument('--fc', type=float, default=10e9)
ap.add_argument('--bandwidth', type=float, default=600e6)
ap.add_argument('--az-res', type=float, default=2.0)
ap.add_argument('--img-size', type=int, default=200)
args = ap.parse_args()

xp, on_gpu = get_backend(args.gpu)
print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

standoff, altitude = args.standoff, args.altitude
o_broadside = np.array([0.0, -standoff, altitude])
wavelength = C / args.fc

facets_b = make_multi_building_scene_adaptive(
    xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
    dR_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, args.footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

d_theta = wavelength / (2.0 * args.az_res)
squint_len = d_theta * standoff
n_pulses = int(np.ceil(standoff * wavelength / (args.az_res ** 2)))
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = xp.asarray(np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

min_freq, maxdr, needed_window = required_freq_samples(
    to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
n_freq = int(np.ceil(min_freq * 1.05))
freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, n_freq))
print(f"n_pulses={n_pulses}, n_freq={n_freq} (same sizing run_direct_to_facet_validation.py uses)")

t0 = time.perf_counter()
s, stats = run_asc_box_projected_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                              return_components=True, progress=True)
print(f"ASC: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
img1 = backproject(xp, stats['s_by_leg']['leg1'], plat, freqs, grid, grid, args.fc, R_ref)
img2 = backproject(xp, stats['s_by_leg']['leg2'], plat, freqs, grid, grid, args.fc, R_ref)
img_sum = backproject(xp, s, plat, freqs, grid, grid, args.fc, R_ref)
print(f"backprojection: {time.perf_counter()-t0:.1f}s")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, img, title in [(axes[0], img1, 'leg1 (direct)'),
                        (axes[1], img2, 'leg2 (ground bounce)'),
                        (axes[2], img_sum, 'leg1+leg2 (combined)')]:
    db = 20 * np.log10(np.abs(to_numpy(img)) + 1e-12)
    db -= db.max()
    ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower',
              extent=[-args.footprint/2, args.footprint/2, -args.footprint/2, args.footprint/2])
    ax.set_title(title)
fig.suptitle(f"leg decomposition: {facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets, "
             f"footprint={args.footprint}m, density={args.density}/km2")
fig.tight_layout()
fig.savefig('diagnose_streak_artifact.png', dpi=130)
print("saved diagnose_streak_artifact.png")

# Also report which specific FACETS have valid leg2 ground-bounce points
# whose y-coordinate clusters near any suspiciously-common value -- if
# many different facets' G.y (leg2 ground hit point) collapse near the
# same value regardless of the facet's own position, that's a strong
# signal of a real bug in the ground-reflection geometry rather than a
# resolution/sampling artifact.
Cb, Nb = facets_b['center'], facets_b['normal']
o_bs = xp.asarray(o_broadside)
look = Cb - o_bs[None, :]
R_asc = xp.linalg.norm(look, axis=1)
d_in = look / R_asc[:, None]
from box_projected_multibounce import _reflect_and_intersect_scene, _building_boxes_from_facets
box_min, box_max = _building_boxes_from_facets(xp, facets_b)
hit = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, float(facets_g.get('footprint_half_extent', 50.0)),
                                    box_min, box_max, facets_b['facet_building_id'])
G, valid = hit['G'], hit['valid']
Gy_valid = to_numpy(G[valid, 1])
print(f"\nleg2 ground/box hit points at broadside: {valid.sum()}/{len(valid)} valid, "
      f"G.y range [{Gy_valid.min():.1f}, {Gy_valid.max():.1f}]")
hist, edges = np.histogram(Gy_valid, bins=40)
top = np.argsort(hist)[-5:][::-1]
print("busiest G.y bins (potential clustering):")
for i in top:
    print(f"  y in [{edges[i]:.1f}, {edges[i+1]:.1f}]: {hist[i]} facets")
