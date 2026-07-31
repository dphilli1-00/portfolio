"""Run SBR (max_bounces=3) and box-projected point-ASC (include_order3=True)
on a scene dense enough to actually have triple-bounce paths, and save
each of the 3 bounce levels separately (not just summed) so they can be
displayed side by side -- order1 (direct wall), order2 (wall-ground or
wall-wall double bounce), order3 (triple bounce).

density_per_km2=1500 / seed=1 is the smallest scene in this project that
was confirmed (via check_leg3_denser.py) to have nonzero order3 energy at
all -- the default 350/km2 4-building scene never does (every order-2
bounce there lands on flat ground or a building ROOF, and bouncing off a
horizontal roof sends the ray skyward, a dead end for a third bounce).

Tune PULSES / AIM_GRID_N down if this is too slow, up for a cleaner
picture -- see run_multi_building_validation.py's docstring for the same
pulses-vs-footprint and ray-density-vs-order2-fidelity tradeoffs, which
apply here too (and are worse for order3: a third bounce needs a cast
ray to survive three sequential "landed in the right discrete cell"
tests, so SBR's order3 term needs much denser rays than order2 to be a
statistically fair ground truth -- see the docstring note in
run_asc_box_projected_multibounce for the honest caveat: at moderate ray
counts SBR under-discovers order3 paths badly, so treat any SBR-vs-ASC
order3 mismatch as possibly a sampling-density artifact, not necessarily
a model bug, until ray density is pushed much higher).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import make_ground_facet, get_backend, make_aim_grid
from multibounce_demo import run_multibounce_sbr
from validation_tier2_compare import backproject, compute_ref_ranges, to_numpy, required_freq_samples
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import run_asc_box_projected_multibounce

t_start = time.perf_counter()
xp, on_gpu = get_backend(False)

SCENE_FOOTPRINT_M = 100.0
IMG_WINDOW_M = 100.0
GROUND_MATERIAL = 'dry_soil'
DENSITY_PER_KM2 = 1500.0
SEED = 1
standoff, altitude = 8000.0, 3000.0
fc, bandwidth = 9.6e9, 6.0e8
PULSES = 60          # bump for finer azimuth resolution / less aliasing
AIM_GRID_N = 150      # bump for denser SBR ray sampling (order3 needs a LOT more than order2)
IMG_SIZE = 100
squint_len = 400.0

o_broadside = np.array([0.0, -standoff, altitude])
facets_b = make_multi_building_scene_adaptive(
    xp, SCENE_FOOTPRINT_M, density_per_km2=DENSITY_PER_KM2, seed=SEED, o=o_broadside, dR_max=3.0,
    min_wh=(8.0, 8.0, 6.0), max_wh=(18.0, 18.0, 16.0))
facets_g = make_ground_facet(xp, SCENE_FOOTPRINT_M, material=GROUND_MATERIAL)
print(f"n_facets building={facets_b['n_facets']}  ground={facets_g['center'].shape[0]}", flush=True)

u = np.linspace(-squint_len / 2, squint_len / 2, PULSES)
plat = xp.asarray(np.column_stack([u, np.full(PULSES, -standoff), np.full(PULSES, altitude)]))
aim_pts, margin = make_aim_grid(xp, SCENE_FOOTPRINT_M, AIM_GRID_N, standoff, altitude, max_height_m=45.0)
print(f"aim grid: {aim_pts.shape[0]} rays over {SCENE_FOOTPRINT_M}m scene", flush=True)

ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-IMG_WINDOW_M / 2, IMG_WINDOW_M / 2, IMG_SIZE))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
n_freq = int(min_freq * 1.05)
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
print(f"required_freq_samples: min_freq={min_freq} -> n_freq={n_freq}", flush=True)
print(f"setup: {time.perf_counter() - t_start:.1f}s", flush=True)

t0 = time.perf_counter()
s_sbr, stats_sbr = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                        max_bounces=3, return_components=True)
print(f"[1/2] SBR (max_bounces=3): {time.perf_counter() - t0:.1f}s  counts={stats_sbr['counts']}")

t0 = time.perf_counter()
s_asc, stats_asc = run_asc_box_projected_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                                       ground_material=GROUND_MATERIAL, return_components=True,
                                                       include_order3=True)
print(f"[2/2] point-ASC (leg1+leg2+leg3): {time.perf_counter() - t0:.1f}s")

# --- backproject each of the 3 levels SEPARATELY, both sides ---
levels_sbr = {
    'order1': stats_sbr['s_by_order']['order1'],
    'order2': stats_sbr['s_by_order']['order2'],
    'order3': stats_sbr['s_by_order']['order3'],
}
levels_asc = {
    'order1': stats_asc['s_by_leg']['leg1'],
    'order2': stats_asc['s_by_leg']['leg2'],
    'order3': stats_asc['s_by_leg']['leg3'],
}

imgs = {}
for side, levels in [('sbr', levels_sbr), ('asc', levels_asc)]:
    for name, s_level in levels.items():
        imgs[f'{side}_{name}'] = to_numpy(backproject(xp, s_level, plat, freqs, grid, grid, fc, R_ref))

np.savez('order3_levels_images.npz', grid=to_numpy(grid), **imgs)
print(f"\nsaved order3_levels_images.npz  (keys: {sorted(imgs.keys())})")
print(f"total: {time.perf_counter() - t_start:.1f}s")
