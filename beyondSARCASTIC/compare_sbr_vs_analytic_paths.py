"""compare_occlusion_all.py proved the two occlusion FUNCTIONS
(_segment_occluded_by_facets vs _return_visible) agree on every
candidate for this scene -- 30/30. So the earlier finding (disabling
ASC's occlusion check improves agreement with SBR) can't be explained by
the occlusion logic itself disagreeing.

Next hypothesis: real SBR's ray-traced order-2 path never actually tests
occlusion at ASC's continuous, analytically-computed ground point G --
it tests occlusion (via _return_visible, inside run_multibounce_sbr) at
wherever the REFLECTED RAY actually landed, which is constrained to one
of the ground's discrete tiled patches (make_ground_facet). Same nominal
wall facet, two different literal 3D points being tested for occlusion --
that could flip the verdict even though the occlusion FUNCTIONS agree
when fed the same point.

This script runs the real run_multibounce_sbr pipeline for a single
(broadside) pulse, extracts which wall facets (idx1) survive into valid
order-2 paths, and cross-references that against the analytic
"ASC says blocked" set for the same scene. If any of ASC's blocked
facets show up as VALID in SBR's real output, that confirms the
ground-point-discretization explanation.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import make_ground_facet, get_backend, make_aim_grid, compute_layover_margin
from multibounce_demo import (run_multibounce_sbr, _reflect_and_intersect_ground_plane,
                               _segment_occluded_by_facets, to_numpy)
from scene_multi_building_adaptive import make_multi_building_scene_adaptive

xp, on_gpu = get_backend(False)
standoff, altitude = 8000.0, 3000.0
o_broadside = np.array([0.0, -standoff, altitude])

FOOTPRINT_M = 100.0
DENSITY_PER_KM2 = 350.0
SEED = 1
DR_MAX = 3.0
MAX_WH = (18.0, 18.0, 16.0)
fc, bandwidth = 9.6e9, 6.0e8

facets_b = make_multi_building_scene_adaptive(
    xp, FOOTPRINT_M, density_per_km2=DENSITY_PER_KM2, seed=SEED, o=o_broadside, dR_max=DR_MAX,
    min_wh=(8.0, 8.0, 6.0), max_wh=MAX_WH)
facets_g = make_ground_facet(xp, FOOTPRINT_M, material='dry_soil')

# --- analytic ASC-side occlusion set (same as compare_occlusion_all.py) ---
Cb = facets_b['center']; Nb = facets_b['normal']
o = xp.asarray(o_broadside)
look = Cb - o[None, :]
R = xp.linalg.norm(look, axis=1)
d_in = look / R[:, None]
half_extent_g = FOOTPRINT_M / 2.0
G, valid_geom, d_out = _reflect_and_intersect_ground_plane(xp, Cb, Nb, d_in, half_extent_g)
cos_inc1 = xp.sum(-d_in * Nb, axis=1)
visible1 = cos_inc1 > 0
valid = visible1 & valid_geom
cand_idx = to_numpy(xp.nonzero(valid)[0])
hit_pts = G[xp.nonzero(valid)[0]]
o_rep = xp.broadcast_to(o[None, :], (cand_idx.shape[0], 3))
blocked_asc = to_numpy(_segment_occluded_by_facets(xp, hit_pts, o_rep, facets_b))
asc_blocked_facets = set(cand_idx[blocked_asc].tolist())
asc_valid_facets = set(cand_idx[~blocked_asc].tolist())
print(f"ASC analytic: {len(asc_valid_facets)} valid, {len(asc_blocked_facets)} blocked -> {sorted(asc_blocked_facets)}")

# --- real SBR ray-traced result, single broadside pulse ---
plat = xp.asarray([[0.0, -standoff, altitude]])
margin = compute_layover_margin(standoff, altitude, 45.0)
half = 15.0 + margin
g = np.linspace(-FOOTPRINT_M / 2 - margin, FOOTPRINT_M / 2 + margin, 200)
gx, gy = np.meshgrid(g, g)
aim_pts = xp.asarray(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))
freqs = xp.asarray([fc])
ref_pos = xp.zeros(3)

s_sbr, stats_sbr = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                        max_bounces=2, return_components=True)
print(f"SBR counts: {stats_sbr['counts']}")

# NOTE: run_multibounce_sbr doesn't currently expose which idx1 values
# survived into valid order-2 paths directly -- if stats_sbr has a
# 'valid_wall_facets_order2' or similar key inspect it here; otherwise
# this print of the raw stats dict keys will show what's available to
# extract, and multibounce_demo.py's run_multibounce_sbr source (around
# the order-2 bounce block, ~line 625-660) shows exactly where idx1/idx2
# and the valid2 mask live if it needs a small return_components addition
# to expose them directly.
print("stats_sbr keys:", list(stats_sbr.keys()))
