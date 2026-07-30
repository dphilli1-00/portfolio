"""Diagnostic: with the taper now isolating the amplitude question, what's
actually behind the remaining ~40% ray-traced-vs-ASC candidate-count gap?

Two live hypotheses, checked independently at a single platform position
(cheap, no need for a full multi-pulse run):

  A) single-point (facet-CENTER-only) sampling in ASC's valid_geom2 test
     misses walls where the center's own reflection just misses ground
     bounds, even though other points on that same (few-meters-wide)
     facet would reflect in-bounds -- exactly the mechanism flagged (but
     not fixed) earlier this session.
  B) occlusion filtering is removing real, unobstructed paths (a false
     positive block) rather than the geometric reach test being the
     bottleneck.

Measured separately by comparing, for the SAME set of wall facets ray
tracing found at least one valid order-2 path through:
  - how many ASC's center-based valid_geom2 (backface + reach, BEFORE
    occlusion) already excludes -- isolates (A)
  - of the ones valid_geom2 DOES accept, how many occlusion then blocks
    -- isolates (B)
"""
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, ray_facet_intersect, get_backend, make_aim_grid, C
import multibounce_demo as mb

xp, on_gpu = get_backend(False)

footprint, density, rays = 300.0, 200.0, 150
standoff, altitude = 8000.0, 3000.0

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

facets_combined = mb.concat_facets(xp, facets_b, facets_g)
F_b = facets_b['center'].shape[0]
F_c = facets_combined['center'].shape[0]

o = xp.asarray([0.0, -standoff, altitude])   # single platform position, broadside
aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)

Cb, Nb, Ab = facets_b['center'], facets_b['normal'], facets_b['amp']
Cc, Nc = facets_combined['center'], facets_combined['normal']

# ---- ray-traced order-2: which wall facets (idx1) got at least one valid path? ----
d1 = aim_pts - o[None, :]
d1 = d1 / xp.linalg.norm(d1, axis=1, keepdims=True)
hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(xp, o, d1, facets_b)
hit_normal1 = Nb[idx1]

d2 = d1 - 2.0 * xp.sum(d1 * hit_normal1, axis=1, keepdims=True) * hit_normal1
eps = 1e-3
o2 = hit_pt1 + eps * hit_normal1
hit_pt2, hit_amp2, cos_inc2, hit_mask2_raw, idx2 = ray_facet_intersect(xp, o2, d2, facets_combined)
hit_mask2 = hit_mask2_raw & hit_mask1
hit_normal2 = Nc[idx2]
clear2 = mb._return_visible(xp, hit_pt2, hit_normal2, o, facets_combined, eps)
valid2 = hit_mask2 & clear2

ray_valid_idx1 = np.unique(mb.to_numpy(idx1[valid2]))
print(f"\nRay-traced: {ray_valid_idx1.shape[0]} distinct wall facets produced >=1 valid order-2 path")

# ---- ASC's own per-facet checks at the SAME platform position ----
look = Cb - o[None, :]
R_asc = xp.linalg.norm(look, axis=1)
d_in = look / R_asc[:, None]
cos_inc1_asc = xp.sum(-d_in * Nb, axis=1)
visible1 = cos_inc1_asc > 0

half_extent_g = float(facets_g['footprint_half_extent'])
G, valid_geom2, d_out = mb._reflect_and_intersect_ground_plane(xp, Cb, Nb, d_in, half_extent_g)
valid_geom_asc = visible1 & valid_geom2   # BEFORE occlusion

asc_geomvalid_idx1 = np.nonzero(mb.to_numpy(valid_geom_asc))[0]
print(f"ASC (center-based, pre-occlusion): {asc_geomvalid_idx1.shape[0]} wall facets pass backface+reach")

# hypothesis A: of the ray-confirmed-valid walls, how many does ASC's
# CENTER-only reach test already reject, before occlusion even runs?
missed_by_center_test = np.setdiff1d(ray_valid_idx1, asc_geomvalid_idx1)
print(f"\n[Hypothesis A] Ray-valid walls REJECTED by ASC's center-only reach test "
      f"(before occlusion): {missed_by_center_test.shape[0]} / {ray_valid_idx1.shape[0]} "
      f"({100.0*missed_by_center_test.shape[0]/max(ray_valid_idx1.shape[0],1):.1f}%)")

# hypothesis B: of the walls BOTH agree pass the free geometric check,
# how many does occlusion then block?
both_geom_valid = np.intersect1d(ray_valid_idx1, asc_geomvalid_idx1)
if both_geom_valid.shape[0] > 0:
    idx_xp = xp.asarray(both_geom_valid)
    o_rep = xp.broadcast_to(o[None, :], (idx_xp.shape[0], 3))
    blocked1 = mb._segment_occluded_by_facets(xp, o_rep, Cb[idx_xp], facets_b)
    blocked2 = mb._segment_occluded_by_facets(xp, Cb[idx_xp], G[idx_xp], facets_b)
    blocked3 = mb._segment_occluded_by_facets(xp, G[idx_xp], o_rep, facets_b)
    blocked = mb.to_numpy(blocked1 | blocked2 | blocked3)
    print(f"[Hypothesis B] Of the {both_geom_valid.shape[0]} walls BOTH sides agree geometrically reach: "
          f"occlusion blocks {int(blocked.sum())} ({100.0*blocked.sum()/both_geom_valid.shape[0]:.1f}%)")
else:
    print("[Hypothesis B] no overlap to test")
