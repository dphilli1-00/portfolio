"""Does point-ASC's exact segment-vs-facet occlusion check
(_segment_occluded_by_facets) agree with SBR's own return-path
visibility check (_return_visible), on the SAME candidate wall-ground
paths for the SAME multi-building scene?

Background: with occlusion_check=True, point-ASC and adaptive-mesh SBR
diverged (order2 SSIM ~0.76-0.80). Disabling ASC's occlusion check
improved agreement (SSIM ~0.90), suggesting ASC's occlusion filter was
discarding paths SBR's own result includes. A hand-checked single
candidate (facet 8) actually showed the two checks AGREEING (both said
"blocked"), and a first pass over all 30 geometrically-valid candidates
in a small 4-building scene found 30/30 agreement too -- meaning
whatever's actually causing the SBR-vs-ASC divergence isn't captured by
this direct analytic comparison, at least not on that scene size. This
script is the tool to keep digging: run it on a bigger/denser scene (or
plug in the same scene run_multi_building_validation.py uses with more
buildings/rays) to see whether disagreements show up at scale, or
whether the real explanation is somewhere else entirely (e.g. SBR's
own ray density still not resolving every candidate the analytic check
enumerates, independent of the occlusion question).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import make_ground_facet, get_backend
from multibounce_demo import (_reflect_and_intersect_ground_plane, _segment_occluded_by_facets,
                               _return_visible, to_numpy)
from scene_multi_building_adaptive import make_multi_building_scene_adaptive

xp, on_gpu = get_backend(False)
standoff, altitude = 8000.0, 3000.0
o_broadside = np.array([0.0, -standoff, altitude])

# same scene-generation knobs as run_multi_building_validation.py -- bump
# density_per_km2 / footprint / max_wh here to stress-test at a bigger scale
FOOTPRINT_M = 100.0
DENSITY_PER_KM2 = 350.0
SEED = 1
DR_MAX = 3.0
MAX_WH = (18.0, 18.0, 16.0)

facets_b = make_multi_building_scene_adaptive(
    xp, FOOTPRINT_M, density_per_km2=DENSITY_PER_KM2, seed=SEED, o=o_broadside, dR_max=DR_MAX,
    min_wh=(8.0, 8.0, 6.0), max_wh=MAX_WH)
facets_g = make_ground_facet(xp, FOOTPRINT_M, material='dry_soil')


def concat_facets(a, b):
    return {k: xp.concatenate([a[k], b[k]], axis=0) for k in
            ['center', 'u_hat', 'v_hat', 'normal', 'half_u', 'half_v', 'amp']}


facets_combined = concat_facets(facets_b, facets_g)

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
cand_idx = xp.nonzero(valid)[0]
n = cand_idx.shape[0]
print(f"{n} geometrically-valid candidates (out of {Cb.shape[0]} building facets)")

hit_pts = G[cand_idx]
ground_normal = xp.tile(xp.asarray([[0.0, 0.0, 1.0]]), (n, 1))
o_rep = xp.broadcast_to(o[None, :], (n, 3))
eps = 1e-3

# ASC's occlusion test: buildings only, eps offset along the SEGMENT direction
blocked_asc = to_numpy(_segment_occluded_by_facets(xp, hit_pts, o_rep, facets_b))
# SBR's own return-visibility test: buildings+ground combined, eps offset
# along the bounce point's SURFACE NORMAL (not the segment direction)
clear_sbr = to_numpy(_return_visible(xp, hit_pts, ground_normal, o, facets_combined, eps))
blocked_sbr = ~clear_sbr

agree = blocked_asc == blocked_sbr
print(f"agree: {agree.sum()}/{n}   disagree: {(~agree).sum()}/{n}")
print(f"ASC says blocked: {blocked_asc.sum()}   SBR-style says blocked: {blocked_sbr.sum()}")

idxs_np = to_numpy(cand_idx)
for i in range(n):
    if not agree[i]:
        print(f"  DISAGREE at facet {idxs_np[i]}: ASC_blocked={blocked_asc[i]}  SBR_blocked={blocked_sbr[i]}  "
              f"G={to_numpy(hit_pts[i])}")
