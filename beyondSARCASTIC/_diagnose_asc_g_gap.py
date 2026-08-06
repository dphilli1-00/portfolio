"""Internal-only CPU diagnostic (does NOT modify production code). Follow-up
to the full-scale GPU result showing the hit_pt2 fix (SBR-side, now
correctly tapered) REGRESSED leg2_building coherence vs the original facet-
center convention (0.6289 -> 0.3122). Working hypothesis: the original
"agreement" was two independently-approximate systems (SBR's coarse facet-
center, ASC's idealized box-surface G) coincidentally correlating with each
other better than either does with true physics -- making SBR's own leg2
scoring MORE accurate (hit_pt2) moved it AWAY from ASC's G, not toward it,
which points at ASC's G being the thing that's actually imprecise.

This tests that directly: for the SAME real source facet (idx1) and SAME
platform position, compute BOTH systems' second-bounce target position for
the SAME outgoing direction --
  SBR:  ray-trace d_out against the REAL faceted scene (ray_facet_intersect)
        -> hit_pt2 (literal reflection point on whatever real facet's plane
        it actually lands on)
  ASC:  _reflect_and_intersect_scene (production function from
        box_projected_multibounce.py) against the IDEALIZED building
        bounding boxes -> G

Both use the identical d_in = (Cb[idx1]-o)/|...| and identical specular
reflection law, so d_out should be numerically identical between the two --
any difference in the resulting target point comes purely from "real facets"
vs "idealized box" as the SECOND-bounce target representation. Also checks
whether ASC's nearest-box search even agrees with SBR's ray tracing on WHICH
building gets hit at all (a more fundamural target-selection question, not
just position precision).
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet, ray_facet_intersect, concat_facets
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import _building_aabbs, _return_visible
from box_projected_multibounce import _reflect_and_intersect_scene, _building_boxes_from_facets

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 200.0
density = 400.0
fc = 10e9
standoff, altitude = 8000.0, 3000.0
wavelength = C / fc
o_broadside = np.array([0.0, -standoff, altitude])
ground_half_extent = footprint / 2.0

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=o_broadside,
    dR_max=5.0, delta_cr=5.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

facets_combined = concat_facets(xp, facets_b, facets_g)
F_b = facets_b['center'].shape[0]
F_c = facets_combined['center'].shape[0]
ground_idx = F_b
Cb, Nb = facets_b['center'], facets_b['normal']
fbid = facets_b['facet_building_id']
box_min, box_max, bid_b = _building_aabbs(xp, facets_b)
bid_c = np.concatenate([bid_b, np.full(F_c - F_b, -1, dtype=bid_b.dtype)])

# ASC's own building boxes (built from metadata, not facets -- see that
# function's docstring for why it differs from _building_aabbs)
asc_box_min, asc_box_max = _building_boxes_from_facets(xp, facets_b)

az_res = 4.0
d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 10
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])

all_pos_dist = []
all_dR = []
all_same_building = []
n_bt_total = 0
n_asc_missed = 0
n_asc_ground_sbr_building = 0

for p in range(n_pulses):
    o = plat[p]
    d1 = Cb - o[None, :]
    R1 = np.linalg.norm(d1, axis=1)
    d1 = d1 / R1[:, None]
    hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(
        xp, o, d1, facets_b, box_min=box_min, box_max=box_max, building_id_of_facet=bid_b)

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
    idx_bt = np.nonzero(building_target)[0]
    n_bt_total += idx_bt.shape[0]
    if idx_bt.shape[0] == 0:
        continue

    # ASC's own second-bounce target for the SAME source facet (idx1) and
    # SAME d_in, using the production function unmodified.
    idx1_bt = idx1[idx_bt]
    C_bounce = Cb[idx1_bt]
    N_bounce = Nb[idx1_bt]
    d_in_bt = d1[idx_bt]
    hit_asc = _reflect_and_intersect_scene(xp, C_bounce, N_bounce, d_in_bt, ground_half_extent,
                                            asc_box_min, asc_box_max, fbid[idx1_bt])
    G = hit_asc['G']; valid_asc = hit_asc['valid']; is_ground_asc = hit_asc['is_ground']
    building_hit_asc = hit_asc['building_hit']

    sbr_target_building = facets_combined['facet_building_id'][idx2[idx_bt]] if 'facet_building_id' in facets_combined else fbid[idx2[idx_bt]]
    # facets_combined doesn't carry facet_building_id past F_b (ground has none) --
    # idx2[idx_bt] is always < ground_idx here (building_target mask), so fbid indexing is safe
    sbr_target_building = fbid[idx2[idx_bt]]

    for k in range(idx_bt.shape[0]):
        if not bool(valid_asc[k]):
            n_asc_missed += 1
            continue
        if bool(is_ground_asc[k]):
            n_asc_ground_sbr_building += 1
            continue
        same_b = int(building_hit_asc[k]) == int(sbr_target_building[k])
        all_same_building.append(same_b)
        if not same_b:
            continue
        pos_d = float(np.linalg.norm(G[k] - hit_pt2[idx_bt[k]]))
        all_pos_dist.append(pos_d)
        # range-equivalent error (what actually hits the phase term)
        o_pt = o
        L_hit = (np.linalg.norm(o_pt - C_bounce[k]) + np.linalg.norm(C_bounce[k] - hit_pt2[idx_bt[k]])
                 + np.linalg.norm(o_pt - hit_pt2[idx_bt[k]]))
        L_G = (np.linalg.norm(o_pt - C_bounce[k]) + np.linalg.norm(C_bounce[k] - G[k])
               + np.linalg.norm(o_pt - G[k]))
        dR = abs((L_hit - L_G) / 2.0)
        all_dR.append(dR)

print(f"\ntotal real SBR building-target order2 hits across {n_pulses} pulses: {n_bt_total}")
print(f"ASC found no valid target at all for: {n_asc_missed} of those (nearest-box search missed where SBR's real ray hit)")
print(f"ASC's nearest-box search picked GROUND instead of a building for: {n_asc_ground_sbr_building}")
print(f"of the rest, ASC agreed with SBR on WHICH building: {sum(all_same_building)}/{len(all_same_building)} "
      f"({100*sum(all_same_building)/max(1,len(all_same_building)):.1f}%)")

if all_pos_dist:
    all_pos_dist = np.array(all_pos_dist)
    all_dR = np.array(all_dR)
    print(f"\nfor the {len(all_pos_dist)} cases where ASC's G and SBR's hit_pt2 target the SAME building:")
    print(f"  |G - hit_pt2| position distance: median {np.median(all_pos_dist):.3f}m ({np.median(all_pos_dist)/wavelength:.1f} wavelengths), "
          f"max {all_pos_dist.max():.3f}m ({all_pos_dist.max()/wavelength:.1f} wavelengths)")
    print(f"  range-equivalent error (hits the phase term directly): median {np.median(all_dR):.4f}m ({np.median(all_dR)/wavelength:.2f} wavelengths)")
    frac_over_q = (all_dR > wavelength / 4.0).mean()
    print(f"  fraction exceeding lambda/4 (>90deg two-way phase error): {frac_over_q*100:.1f}%")
else:
    print("\nno same-building agreement cases found at this pulse/seed sample -- try more pulses")
