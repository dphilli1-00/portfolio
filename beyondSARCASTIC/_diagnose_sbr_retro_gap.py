"""Internal-only CPU diagnostic (does NOT modify production code). Follow-up
to the G-vs-hit_pt2 test, which found ASC's G tracks SBR's literal hit_pt2
almost exactly at the median (0.001m) -- position precision is NOT the
driver of the coherence gap. Set-size asymmetry pointed elsewhere: ASC's
retro-gate keeps only ~37% as many building-target contributions as SBR's
own ray-traced population at full scale (73,804 vs 196,633).

New hypothesis: SBR's own ray tracing NEVER checks whether bounce-2's own
law of reflection actually sends the ray back toward the sensor. It only
checks (a) does bounce-1's specular direction hit SOME real facet
(bounce-2), and (b) is a STRAIGHT LINE from that landing point back to the
sensor unoccluded (_return_visible -- pure line-of-sight, not a specular-law
check). Physically, for a genuine monostatic double-bounce return, bounce-2's
OWN specular reflection (incoming direction mirrored off bounce-2's own
normal) must point back toward the sensor -- if it doesn't, no real photon
travels that exact path back to a monostatic receiver, regardless of whether
the straight-line sightline happens to be clear. _score_paths never checks
this (only cos-incidence-weighted reflectivity), so SBR's own
order2_building population may include a large fraction of geometrically-
found-but-not-actually-retroreflective bounces that get scored as if fully
coherent anyway -- the same gap ASC's retro-gate exists to close on ITS side,
just never applied to SBR's reference population.

This measures, for SBR's own real, ray-traced, occlusion-clear building-
target hits: what fraction ALSO satisfy the same retro-alignment test
leg2_retroreflection_check applies on the ASC side (reflect bounce-2's
incoming direction off bounce-2's OWN normal, compare to the actual
direction from hit_pt2 back to the sensor, gate on misalignment angle vs a
diffraction-limited beamwidth tolerance)."""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet, ray_facet_intersect, concat_facets
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import _building_aabbs, _return_visible

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 200.0
density = 400.0
fc = 10e9
standoff, altitude = 8000.0, 3000.0
wavelength = C / fc
o_broadside = np.array([0.0, -standoff, altitude])
retro_beamwidth_mult = 3.0

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
box_min, box_max, bid_b = _building_aabbs(xp, facets_b)
bid_c = np.concatenate([bid_b, np.full(F_c - F_b, -1, dtype=bid_b.dtype)])
half_u_c = facets_combined['half_u']

az_res = 4.0
d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 10
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])

n_bt_total = 0
n_retro_pass = 0
all_retro_angle_deg = []

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

    # SBR's own bounce-2 specular reflection law -- what does the physics
    # ACTUALLY say the outgoing direction from bounce-2 is, given the real
    # incoming direction (d2) and the real facet's own normal (hit_normal2)?
    d2_bt = d2[idx_bt]
    n2_bt = hit_normal2[idx_bt]
    retro_dir = d2_bt - 2.0 * np.sum(d2_bt * n2_bt, axis=1, keepdims=True) * n2_bt

    return_vec = o[None, :] - hit_pt2[idx_bt]
    return_len = np.linalg.norm(return_vec, axis=1)
    return_len_safe = np.where(return_len > 1e-9, return_len, 1.0)
    to_sensor_dir = return_vec / return_len_safe[:, None]

    retro_cos = np.clip(np.sum(retro_dir * to_sensor_dir, axis=1), -1.0, 1.0)
    retro_angle = np.arccos(retro_cos)
    all_retro_angle_deg.extend(np.degrees(retro_angle).tolist())

    L_wall2 = 2.0 * half_u_c[idx2[idx_bt]]
    beamwidth = wavelength / np.maximum(L_wall2, wavelength)
    retro_pass = retro_angle < (retro_beamwidth_mult * beamwidth)
    n_retro_pass += int(retro_pass.sum())

print(f"\nSBR's own real, ray-traced, occlusion-clear building-target hits: {n_bt_total}")
print(f"of those, fraction that ALSO pass the same retro-alignment test ASC's leg2_retroreflection_check applies: "
      f"{n_retro_pass}/{n_bt_total} ({100*n_retro_pass/max(1,n_bt_total):.1f}%)")

all_retro_angle_deg = np.array(all_retro_angle_deg)
print(f"\nretro-misalignment angle distribution over SBR's real building-target hits (degrees):")
print(f"  median: {np.median(all_retro_angle_deg):.1f}, mean: {all_retro_angle_deg.mean():.1f}, "
      f"90th pct: {np.percentile(all_retro_angle_deg, 90):.1f}, max: {all_retro_angle_deg.max():.1f}")
print(f"  fraction < 5deg:  {(all_retro_angle_deg < 5).mean()*100:.1f}%")
print(f"  fraction < 15deg: {(all_retro_angle_deg < 15).mean()*100:.1f}%")
print(f"  fraction < 45deg: {(all_retro_angle_deg < 45).mean()*100:.1f}%")
