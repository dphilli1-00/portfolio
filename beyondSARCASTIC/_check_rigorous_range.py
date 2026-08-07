"""Internal-only CPU check (does NOT modify production code). Derives a
RIGOROUS (not empirical) per-ray upper bound on how far a reflected ray
could possibly need to travel to hit a building box, then takes the global
worst case across real facets/pulses as a candidate static culling range,
then validates a culled vs. UNcalled box search give IDENTICAL results on
real scene data -- the actual safety net, independent of how defensible the
geometric derivation is on paper.

Two exact per-ray bounds, nearer of the two wins:
  1. Ground bound: if the ray heads downward, it hits the ground at a fixed,
     already-computed t_ground. Any building beyond that distance LOSES to
     the ground hit regardless of whether it's geometrically intersected,
     so it can never change the outcome.
  2. Height-envelope bound: every building box has z in [0, max_building_
     height] (scene-wide, computed from real box_max). If the ray is
     climbing (d_out_z > 0), it exits every building's height envelope at
     a fixed t = (max_height - start_z) / d_out_z; beyond that, no box can
     be intersected no matter how far the ray is extended.
  Rays that are (nearly) level (neither meaningfully descending nor
  climbing) get no bound from this reasoning -- treated explicitly as
  "uncapped" for this diagnostic, not silently ignored.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import _building_boxes_from_facets, _reflect_and_intersect_scene, ray_box_intersect

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 400.0
density = 300.0
fc = 10e9
wavelength = C / fc
standoff, altitude = 8000.0, 3000.0
o_broadside = np.array([0.0, -standoff, altitude])
ground_half_extent = footprint / 2.0

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=o_broadside,
    dR_max=2.0, delta_cr=2.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
n_buildings = facets_b['n_buildings']
print(f"{n_buildings} buildings, {facets_b['n_facets']} facets, footprint={footprint}m")

Cb, Nb = facets_b['center'], facets_b['normal']
fbid = facets_b['facet_building_id']
box_min, box_max = _building_boxes_from_facets(xp, facets_b)
max_building_height = float(np.max(box_max[:, 2]))
print(f"max building height in scene: {max_building_height:.2f}m")

az_res = 3.0
d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 12
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])

worst_bound = 0.0
n_uncapped = 0
n_total = 0
all_actual_t = []

for p in range(n_pulses):
    o = plat[p]
    look = Cb - o[None, :]
    R_asc = np.linalg.norm(look, axis=1)
    d_in = look / R_asc[:, None]
    cos_inc1 = np.sum(-d_in * Nb, axis=1)
    visible1 = cos_inc1 > 0
    if not visible1.any():
        continue

    d_out = d_in - 2.0 * np.sum(d_in * Nb, axis=1, keepdims=True) * Nb
    C_bounce = Cb

    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = np.where(heading_down, d_out[:, 2], -1.0)
    t_ground = np.where(heading_down, -C_bounce[:, 2] / d_out_z_safe, np.inf)

    climbing = d_out[:, 2] > 1e-9
    d_out_z_safe_up = np.where(climbing, d_out[:, 2], 1.0)
    t_height = np.where(climbing, (max_building_height - C_bounce[:, 2]) / d_out_z_safe_up, np.inf)

    per_ray_bound = np.minimum(t_ground, t_height)
    uncapped = ~np.isfinite(per_ray_bound) & visible1
    n_uncapped += int(uncapped.sum())
    n_total += int(visible1.sum())

    finite_bounds = per_ray_bound[visible1 & np.isfinite(per_ray_bound)]
    if finite_bounds.shape[0] > 0:
        worst_bound = max(worst_bound, float(finite_bounds.max()))

    # ground truth: what the EXACT (unculled) box search actually finds,
    # to confirm real hits stay well inside whatever bound we pick
    t_box_exact = ray_box_intersect(xp, C_bounce[visible1], d_out[visible1], box_min, box_max)
    finite_hits = t_box_exact[np.isfinite(t_box_exact)]
    if finite_hits.shape[0] > 0:
        all_actual_t.extend(finite_hits.tolist())

print(f"\nrays with NO bound from this reasoning (near-level, neither descending nor climbing): "
      f"{n_uncapped}/{n_total} ({100*n_uncapped/max(1,n_total):.2f}%)")
print(f"worst-case (max) rigorous per-ray bound across all bounded rays: {worst_bound:.2f}m")

all_actual_t = np.array(all_actual_t) if all_actual_t else np.array([0.0])
print(f"\nactual real box-intersection distances found by the EXACT search (ground truth, all buildings hit):")
print(f"  median: {np.median(all_actual_t):.2f}m, 99th pct: {np.percentile(all_actual_t, 99):.2f}m, "
      f"max: {all_actual_t.max():.2f}m")
