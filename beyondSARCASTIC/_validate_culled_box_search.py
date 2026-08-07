"""Internal-only CPU validation (does NOT modify production code). Prototype
the per-building culled box search and confirm it gives BIT-IDENTICAL
results to the existing exact (unculled) _reflect_and_intersect_scene on
real scene data, using the rigorously-derived global range (not an
empirical guess) validated in _check_rigorous_range.py.

Safety refinement beyond the earlier check: candidate-building membership
here is tested building-CENTER to building-CENTER, but a facet can sit
anywhere on its own building's footprint and the target can be hit anywhere
on ITS footprint too -- so the center-to-center distance test is inflated by
BOTH buildings' own footprint radii (half-diagonal) before comparing to the
rigorous range, so no pair whose closest possible extents could be within
range gets excluded just because their centers happen to be far apart.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import (_building_boxes_from_facets, _reflect_and_intersect_scene,
                                        ray_box_intersect, box_face_normal, to_numpy)

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
fbid_np = to_numpy(xp, fbid)
box_min, box_max = _building_boxes_from_facets(xp, facets_b)
box_min_np, box_max_np = to_numpy(xp, box_min), to_numpy(xp, box_max)
max_building_height = float(np.max(box_max_np[:, 2]))
F = Cb.shape[0]

az_res = 3.0
d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 12
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])

# ---- pre-pass: rigorous global range (same derivation as _check_rigorous_range.py) ----
worst_bound = 0.0
for p in range(n_pulses):
    o = plat[p]
    look = Cb - o[None, :]
    R_asc = np.linalg.norm(look, axis=1)
    d_in = look / R_asc[:, None]
    cos_inc1 = np.sum(-d_in * Nb, axis=1)
    visible1 = cos_inc1 > 0
    d_out = d_in - 2.0 * np.sum(d_in * Nb, axis=1, keepdims=True) * Nb

    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = np.where(heading_down, d_out[:, 2], -1.0)
    t_ground = np.where(heading_down, -Cb[:, 2] / d_out_z_safe, np.inf)

    climbing = d_out[:, 2] > 1e-9
    d_out_z_safe_up = np.where(climbing, d_out[:, 2], 1.0)
    t_height = np.where(climbing, (max_building_height - Cb[:, 2]) / d_out_z_safe_up, np.inf)

    per_ray_bound = np.minimum(t_ground, t_height)
    finite_bounds = per_ray_bound[visible1 & np.isfinite(per_ray_bound)]
    if finite_bounds.shape[0] > 0:
        worst_bound = max(worst_bound, float(finite_bounds.max()))

max_range = worst_bound * 1.05  # small safety margin on top of the rigorous value itself
print(f"rigorous global range: {worst_bound:.2f}m -> using {max_range:.2f}m (5% margin)")

# ---- precompute (ONCE, static, pulse-independent): candidate building
# lists, inflated by both buildings' own footprint radii ----
centers_xy = 0.5 * (box_min_np[:, :2] + box_max_np[:, :2])
extents_xy = box_max_np[:, :2] - box_min_np[:, :2]
radii = 0.5 * np.linalg.norm(extents_xy, axis=1)   # footprint half-diagonal per building
dxy = centers_xy[:, None, :] - centers_xy[None, :, :]
center_dist = np.linalg.norm(dxy, axis=2)
inflated_dist = center_dist - radii[:, None] - radii[None, :]
building_candidates = [np.nonzero(inflated_dist[b] < max_range)[0] for b in range(n_buildings)]
cand_counts = [c.shape[0] for c in building_candidates]
print(f"candidate buildings per building: median {int(np.median(cand_counts))}, "
      f"mean {np.mean(cand_counts):.1f}, max {max(cand_counts)} (out of {n_buildings})")

facet_indices_by_building = [np.nonzero(fbid_np == b)[0] for b in range(n_buildings)]


def culled_reflect_and_intersect(C_bounce, N_bounce, d_in):
    """Prototype of the culled path -- same output contract as
    _reflect_and_intersect_scene, built from the SAME primitives
    (ray_box_intersect, box_face_normal), just grouped per building using
    the precomputed static candidate lists instead of testing every
    building for every facet."""
    N_rays = C_bounce.shape[0]
    d_out = d_in - 2.0 * np.sum(d_in * N_bounce, axis=1, keepdims=True) * N_bounce

    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = np.where(heading_down, d_out[:, 2], -1.0)
    t_ground = np.where(heading_down, -C_bounce[:, 2] / d_out_z_safe, np.inf)
    G_ground = C_bounce + np.where(np.isfinite(t_ground), t_ground, 0.0)[:, None] * d_out
    in_bounds = (np.abs(G_ground[:, 0]) <= ground_half_extent) & (np.abs(G_ground[:, 1]) <= ground_half_extent)
    ground_valid = heading_down & (t_ground > 1e-9) & in_bounds
    t_ground = np.where(ground_valid, t_ground, np.inf)

    best_t = np.full(N_rays, np.inf)
    best_building = np.full(N_rays, -1, dtype=np.int64)
    for b in range(n_buildings):
        idx = facet_indices_by_building[b]
        if idx.shape[0] == 0:
            continue
        cand = building_candidates[b]
        if cand.shape[0] == 0:
            continue
        t_sub = ray_box_intersect(xp, C_bounce[idx], d_out[idx], box_min[cand], box_max[cand])  # (n_b, n_cand)
        self_mask = cand == b
        if self_mask.any():
            t_sub[:, self_mask] = np.inf
        sub_win = np.argmin(t_sub, axis=1)
        sub_t = np.take_along_axis(t_sub, sub_win[:, None], axis=1)[:, 0]
        best_t[idx] = sub_t
        best_building[idx] = cand[sub_win]

    all_t = np.concatenate([t_ground[:, None], best_t[:, None]], axis=1)
    winner_idx = np.argmin(all_t, axis=1)
    t_win = np.take_along_axis(all_t, winner_idx[:, None], axis=1)[:, 0]
    valid = np.isfinite(t_win)
    G = C_bounce + np.where(valid, t_win, 0.0)[:, None] * d_out
    is_ground = winner_idx == 0
    building_hit = np.where(is_ground, -1, best_building)

    ground_normal = np.tile(np.array([[0.0, 0.0, 1.0]]), (N_rays, 1))
    bh_clamped = np.clip(building_hit, 0, box_min.shape[0] - 1)
    bld_normal, bld_face = box_face_normal(xp, G, box_min[bh_clamped], box_max[bh_clamped])
    surf_normal = np.where(is_ground[:, None], ground_normal, bld_normal)
    is_roof_hit = (~is_ground) & (bld_face == 4)
    return dict(G=G, valid=valid, d_out=d_out, building_hit=building_hit,
                is_ground=is_ground, is_roof_hit=is_roof_hit, surf_normal=surf_normal)


max_pos_diff = 0.0
max_t_diff = 0.0
n_mismatched_building = 0
n_mismatched_valid = 0
n_mismatched_ground = 0
n_total_rays = 0

for p in range(n_pulses):
    o = plat[p]
    look = Cb - o[None, :]
    R_asc = np.linalg.norm(look, axis=1)
    d_in = look / R_asc[:, None]

    exact = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, ground_half_extent, box_min, box_max, fbid)
    culled = culled_reflect_and_intersect(Cb, Nb, d_in)

    n_total_rays += F
    valid_mismatch = to_numpy(xp, exact['valid']) != to_numpy(xp, culled['valid'])
    n_mismatched_valid += int(valid_mismatch.sum())
    ground_mismatch = to_numpy(xp, exact['is_ground']) != to_numpy(xp, culled['is_ground'])
    n_mismatched_ground += int(ground_mismatch.sum())
    bh_mismatch = (to_numpy(xp, exact['building_hit']) != to_numpy(xp, culled['building_hit'])) & to_numpy(xp, exact['valid'])
    n_mismatched_building += int(bh_mismatch.sum())

    both_valid = to_numpy(xp, exact['valid']) & to_numpy(xp, culled['valid'])
    if both_valid.any():
        pos_diff = np.linalg.norm(to_numpy(xp, exact['G'])[both_valid] - to_numpy(xp, culled['G'])[both_valid], axis=1)
        max_pos_diff = max(max_pos_diff, float(pos_diff.max()))

print(f"\n=== correctness check across {n_pulses} pulses, {n_total_rays} total rays ===")
print(f"mismatched valid flag:        {n_mismatched_valid}")
print(f"mismatched is_ground flag:    {n_mismatched_ground}")
print(f"mismatched building_hit (where both valid): {n_mismatched_building}")
print(f"max position difference (where both valid): {max_pos_diff:.6f}m ({max_pos_diff/wavelength:.4f} wavelengths)")
print(f"\n-> {'PASS: bit-identical (to floating-point tolerance)' if (n_mismatched_valid==0 and n_mismatched_ground==0 and n_mismatched_building==0 and max_pos_diff < 1e-6) else 'FAIL: culled path diverges from exact -- range is not safe as computed'}")
