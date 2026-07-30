"""Same diagnostic as _diagnose_candidate_gap.py, but swept across several
platform positions spanning the actual 400m aperture used by the
validation run, not just broadside -- the single-position test showed ASC
MORE permissive (31 vs 18), the opposite of the aggregate 20-pulse result
(424 vs 1070), so whatever's driving the aggregate gap must vary across
the aperture in a way one position doesn't capture."""
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, ray_facet_intersect, get_backend, make_aim_grid, C
import multibounce_demo as mb

xp, on_gpu = get_backend(False)

footprint, density, rays = 300.0, 200.0, 150
standoff, altitude = 8000.0, 3000.0

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')
facets_combined = mb.concat_facets(xp, facets_b, facets_g)
Cb, Nb, Ab = facets_b['center'], facets_b['normal'], facets_b['amp']
Cc, Nc = facets_combined['center'], facets_combined['normal']
half_extent_g = float(facets_g['footprint_half_extent'])

aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)
eps = 1e-3

print(f"{facets_b['n_buildings']} buildings\n")
print(f"{'u (m)':>8} {'ray_valid':>10} {'asc_geom':>10} {'A: missed by center':>20} {'B: occl blocked':>17}")

for u in [-200, -100, -50, 0, 50, 100, 200]:
    o = xp.asarray([float(u), -standoff, altitude])

    d1 = aim_pts - o[None, :]
    d1 = d1 / xp.linalg.norm(d1, axis=1, keepdims=True)
    hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(xp, o, d1, facets_b)
    hit_normal1 = Nb[idx1]
    d2 = d1 - 2.0 * xp.sum(d1 * hit_normal1, axis=1, keepdims=True) * hit_normal1
    o2 = hit_pt1 + eps * hit_normal1
    hit_pt2, hit_amp2, cos_inc2, hit_mask2_raw, idx2 = ray_facet_intersect(xp, o2, d2, facets_combined)
    hit_mask2 = hit_mask2_raw & hit_mask1
    hit_normal2 = Nc[idx2]
    clear2 = mb._return_visible(xp, hit_pt2, hit_normal2, o, facets_combined, eps)
    valid2 = hit_mask2 & clear2
    ray_valid_idx1 = np.unique(mb.to_numpy(idx1[valid2]))

    look = Cb - o[None, :]
    R_asc = xp.linalg.norm(look, axis=1)
    d_in = look / R_asc[:, None]
    cos_inc1_asc = xp.sum(-d_in * Nb, axis=1)
    visible1 = cos_inc1_asc > 0
    G, valid_geom2, d_out = mb._reflect_and_intersect_ground_plane(xp, Cb, Nb, d_in, half_extent_g)
    valid_geom_asc = visible1 & valid_geom2
    asc_geomvalid_idx1 = np.nonzero(mb.to_numpy(valid_geom_asc))[0]

    missed_by_center = np.setdiff1d(ray_valid_idx1, asc_geomvalid_idx1)
    both_geom_valid = np.intersect1d(ray_valid_idx1, asc_geomvalid_idx1)
    n_blocked = 0
    if both_geom_valid.shape[0] > 0:
        idx_xp = xp.asarray(both_geom_valid)
        o_rep = xp.broadcast_to(o[None, :], (idx_xp.shape[0], 3))
        b1 = mb._segment_occluded_by_facets(xp, o_rep, Cb[idx_xp], facets_b)
        b2 = mb._segment_occluded_by_facets(xp, Cb[idx_xp], G[idx_xp], facets_b)
        b3 = mb._segment_occluded_by_facets(xp, G[idx_xp], o_rep, facets_b)
        n_blocked = int(mb.to_numpy(b1 | b2 | b3).sum())

    print(f"{u:>8} {ray_valid_idx1.shape[0]:>10} {asc_geomvalid_idx1.shape[0]:>10} "
          f"{missed_by_center.shape[0]:>20} {n_blocked:>17}")
