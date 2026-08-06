"""Internal-only diagnostic (CPU, small scene) -- does NOT modify
multibounce_demo.py. Companion to _diagnose_g_facet_gap.py, which showed
box_projected_multibounce's G (continuous box-intersection point) lands
2.6-6.8m (median 87 wavelengths) from the nearest facet center on the
target building, for facets whose partitioning is coarse because
adaptive_facet_partition sizes resolution relative to the SENSOR, not
isotropically.

This script checks the OTHER side: SBR's own building-target order2
scoring ALSO uses a facet-center approximation (Cc[idx2] in _score_paths,
unless ground_pos_override applies -- which it currently does NOT for
building targets). ray_facet_intersect already computes the literal,
continuous ray/facet intersection point (hit_pt2) for every hit, exactly
the same way it does for ground before ground_pos_override rescues it --
it's just discarded in favor of the facet center for building targets.
_decode_unique_paths_with_rep already tracks a representative ray per
unique (idx1,idx2) path (built for exactly this purpose, currently only
used for ground and order3-chained paths).

This measures, using the REAL production ray-tracing functions (not a
reimplementation), how much RANGE/PHASE difference exists between scoring
a real SBR-discovered building-target order2 path at Cc[idx2] (current
convention) vs hit_pt2 (literal traced point, already computed and
already available via rep_ray_idx) -- i.e., quantifies how much of SBR's
OWN internal building-target position error this would fix, independent
of any comparison to ASC.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_facet, ray_facet_intersect, concat_facets
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import _building_aabbs, _return_visible, _decode_unique_paths_with_rep

xp, on_gpu = get_backend(False)

footprint = 200.0
density = 400.0
fc = 10e9
wavelength = C / fc
standoff, altitude = 8000.0, 3000.0
o = np.array([50.0, -standoff, altitude])

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=np.array([0.0, -standoff, altitude]),
    dR_max=5.0, delta_cr=5.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

facets_combined = concat_facets(xp, facets_b, facets_g)
F_b = facets_b['center'].shape[0]
F_c = facets_combined['center'].shape[0]
ground_idx = F_b
Cb, Nb = facets_b['center'], facets_b['normal']
Cc = facets_combined['center']
box_min, box_max, bid_b = _building_aabbs(xp, facets_b)
bid_c = np.concatenate([bid_b, np.full(F_c - F_b, -1, dtype=bid_b.dtype)])

# bounce 1: sensor -> every building facet directly (same convention as
# run_multibounce_sbr's aim_pts=facets_buildings['center'])
d1 = Cb - o[None, :]
d1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(
    xp, o, d1, facets_b, box_min=box_min, box_max=box_max, building_id_of_facet=bid_b)

# bounce 2: reflect off bounce-1 normal, trace against buildings+ground
hit_normal1 = Nb[idx1]
d2 = d1 - 2.0 * np.sum(d1 * hit_normal1, axis=1, keepdims=True) * hit_normal1
eps = 1e-3
o2 = hit_pt1 + eps * hit_normal1
hit_pt2, hit_amp2, cos_inc2, hit_mask2_raw, idx2 = ray_facet_intersect(
    xp, o2, d2, facets_combined, box_min=box_min, box_max=box_max, building_id_of_facet=bid_c)
hit_mask2 = hit_mask2_raw & hit_mask1
hit_normal2 = facets_combined['normal'][idx2]

clear2 = _return_visible(xp, hit_pt2, hit_normal2, o, facets_combined, eps, box_min=box_min, box_max=box_max, building_id_of_facet=bid_c)
valid2 = hit_mask2 & clear2

is_ground2 = idx2 >= ground_idx
building_target = valid2 & (~is_ground2)
print(f"building-target REAL SBR order2 hits (this pulse): {int(building_target.sum())}")

idx_bt = np.nonzero(building_target)[0]
if idx_bt.shape[0] == 0:
    print("no building-target hits at this pulse -- try a different o")
    sys.exit(0)

# For these REAL discovered paths, compare R_equiv2 using Cc[idx2] (facet
# center, current _score_paths convention for building targets) vs
# hit_pt2 (literal traced point, already computed here -- exactly what
# ground_pos_override already does for ground rows via rep_ray_idx).
R_asc = np.linalg.norm(hit_pt1[idx_bt] - o[None, :], axis=1)
target_center = Cc[idx2[idx_bt]]
target_literal = hit_pt2[idx_bt]

L_center = (np.linalg.norm(o[None, :] - hit_pt1[idx_bt], axis=1)
            + np.linalg.norm(hit_pt1[idx_bt] - target_center, axis=1)
            + np.linalg.norm(o[None, :] - target_center, axis=1))
L_literal = (np.linalg.norm(o[None, :] - hit_pt1[idx_bt], axis=1)
             + np.linalg.norm(hit_pt1[idx_bt] - target_literal, axis=1)
             + np.linalg.norm(o[None, :] - target_literal, axis=1))
R_equiv_center = L_center / 2.0
R_equiv_literal = L_literal / 2.0
dR = np.abs(R_equiv_center - R_equiv_literal)

pos_dist = np.linalg.norm(target_center - target_literal, axis=1)
phase_error_rad = 4.0 * np.pi * dR / wavelength
frac_over_quarter_wave = (dR > wavelength / 4.0).mean()

print(f"\nfacet-center vs literal-hit-point comparison for these {len(idx_bt)} real SBR paths:")
print(f"  position distance:  median {np.median(pos_dist):.3f}m, max {pos_dist.max():.3f}m")
print(f"  R_equiv2 range diff (this is what actually hits the phase term): median {np.median(dR):.4f}m ({np.median(dR)/wavelength:.2f} wavelengths)")
print(f"  max range diff: {dR.max():.4f}m ({dR.max()/wavelength:.2f} wavelengths)")
print(f"  fraction exceeding lambda/4 range error (>90deg two-way phase error): {frac_over_quarter_wave*100:.1f}%")
print(f"\n-> if this fraction is high, SBR's OWN current facet-center convention for building")
print(f"   targets is already introducing large, likely near-random phase error internally --")
print(f"   independent of any comparison to ASC -- and using hit_pt2 (already computed, just")
print(f"   discarded today) instead of Cc[idx2] should be a real, principled fix on SBR's side.")
