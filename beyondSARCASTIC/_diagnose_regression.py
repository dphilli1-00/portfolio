"""Internal-only CPU diagnostic (does NOT modify production code). The real
GPU run just came back showing the hit_pt2 fix REGRESSED leg2_building
coherence (0.6289 -> 0.1431 apples-to-apples), the opposite of what the
tiny 6-pulse CPU A/B suggested. This isolates two candidate explanations by
testing three position/taper combinations against ASC's own leg2_building,
all on the SAME real paths:

  OLD:            Cc[idx2] (facet center) position, taper ENABLED  (pre-fix)
  NEW-AS-SHIPPED: hit_pt2 position,          taper DISABLED (this session's fix)
  NEW-TAPERED:    hit_pt2 position,          taper ENABLED  (isolates: is it the
                   position change or the taper-disable decision that hurts?)

Hypothesis: building_pos_override forcing noisy=True (taper off) was copied
from order3's pos_override_unconditional precedent, where a representative
ray's IDENTITY can genuinely jump discretely between different real rays
pulse-to-pulse (multi-candidate dedup). But order2 here has exactly ONE aim
ray per building facet per pulse (direct-to-facet aiming) -- no multi-
candidate dedup ambiguity -- so for a persistent (idx1,idx2) path, hit_pt2
should vary SMOOTHLY with platform position (continuous ray/plane
intersection, not a discrete jump), meaning the taper-disable justification
may not actually transfer to this case. If NEW-TAPERED coherence is close to
or better than OLD, taper-disable was the wrong call, not hit_pt2 itself.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet, ray_facet_intersect, concat_facets
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import (_building_aabbs, _return_visible, _decode_unique_paths_with_rep,
                               _score_paths, _reflect_and_intersect_ground_plane, _azimuth_sinc_taper)
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import to_numpy, required_freq_samples

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 200.0
density = 400.0
fc = 10e9
bandwidth = 200e6
az_res = 4.0
standoff, altitude = 8000.0, 3000.0
wavelength = C / fc
o_broadside = np.array([0.0, -standoff, altitude])
ground_material = 'dry_soil'
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
Cb, Nb, Ab = facets_b['center'], facets_b['normal'], facets_b['amp']
Cc, Nc, Ac = facets_combined['center'], facets_combined['normal'], facets_combined['amp']
Ub, HUb = facets_b['u_hat'], facets_b['half_u']
Uc, HUc = facets_combined['u_hat'], facets_combined['half_u']
box_min, box_max, bid_b = _building_aabbs(xp, facets_b)
bid_c = np.concatenate([bid_b, np.full(F_c - F_b, -1, dtype=bid_b.dtype)])

d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 10
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])
ref_pos = np.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, 40))
min_freq, maxdr, needed_window = required_freq_samples(plat, ref_pos, to_numpy(grid), bandwidth)
n_freq = int(np.ceil(min_freq * 1.05))
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
K = freqs.shape[0]
print(f"n_pulses={n_pulses}, n_freq={n_freq}")

order2_building_OLD = np.zeros((n_pulses, K), dtype=np.complex128)
order2_building_NEW_NOTAPER = np.zeros((n_pulses, K), dtype=np.complex128)
order2_building_NEW_TAPERED = np.zeros((n_pulses, K), dtype=np.complex128)

# track hit_pt2 for a few persistent (idx1,idx2) keys across pulses, to check smoothness
tracked_positions = {}

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

    key2 = idx1.astype(np.int64) * F_c + idx2.astype(np.int64)
    lvl2, rep2 = _decode_unique_paths_with_rep(xp, key2, valid2, 2, F_c)

    idx2_u = lvl2[1]
    is_ground2_u = idx2_u >= ground_idx
    idx1_u = lvl2[0]
    hit_pt2_u = hit_pt2[rep2]

    # track a couple of persistent building keys for smoothness check
    for k in range(min(3, idx1_u.shape[0])):
        if not bool(is_ground2_u[k]):
            key = (int(idx1_u[k]), int(idx2_u[k]))
            tracked_positions.setdefault(key, []).append((p, hit_pt2_u[k].copy()))

    C1_u = Cb[idx1_u]; N1_u = Nb[idx1_u]
    d_in_u = C1_u - o[None, :]
    d_in_u = d_in_u / np.linalg.norm(d_in_u, axis=1, keepdims=True)
    G_analytic, _valid_geom_u, _ = _reflect_and_intersect_ground_plane(xp, C1_u, N1_u, d_in_u, ground_half_extent)

    o_x = xp.asarray(o)
    ref_pos_x = xp.asarray(ref_pos)

    g_np = to_numpy(is_ground2_u)
    if not bool((~g_np).any()):
        continue
    not_ground2_u = ~is_ground2_u
    lvl2_b = [lvl2[0][not_ground2_u], lvl2[1][not_ground2_u]]

    # OLD: facet center, taper on (default _score_paths behavior, no override)
    c_old_b, _ = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2_b, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                               ground_idx=ground_idx, ground_material=ground_material, wavelength=wavelength,
                               level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
    # NEW-AS-SHIPPED: hit_pt2, taper forced off via building_pos_override
    c_new_notaper, _ = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2_b, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                     ground_idx=ground_idx, ground_material=ground_material, wavelength=wavelength,
                                     building_pos_override=[None, xp.asarray(hit_pt2_u[not_ground2_u])],
                                     level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
    order2_building_OLD[p, :] = to_numpy(c_old_b)
    order2_building_NEW_NOTAPER[p, :] = to_numpy(c_new_notaper)

    # NEW-TAPERED: hit_pt2 position, but taper computed same as OLD (own facet u_hat/half_u).
    # Reimplemented locally (not modifying _score_paths) using the SAME
    # _azimuth_sinc_taper the production function calls internally.
    idx1_b = lvl2_b[0]; idx2_b = lvl2_b[1]
    pos1 = Cb[idx1_b]
    pos2_hit = hit_pt2_u[not_ground2_u]
    o_rep = np.broadcast_to(o, (idx1_b.shape[0], 3))
    L_total = (np.linalg.norm(pos1 - o_rep, axis=1) + np.linalg.norm(pos2_hit - pos1, axis=1)
               + np.linalg.norm(o_rep - pos2_hit, axis=1))
    R_equiv = L_total / 2.0
    incoming1 = pos1 - o_rep
    incoming1_n = incoming1 / np.linalg.norm(incoming1, axis=1, keepdims=True)
    cos_i1 = np.abs(np.sum(-incoming1_n * Nb[idx1_b], axis=1))
    incoming2 = pos2_hit - pos1
    incoming2_n = incoming2 / np.linalg.norm(incoming2, axis=1, keepdims=True)
    norm2 = Cc[idx2_b]  # facet normal lookup only (orientation), position is hit_pt2
    normal2_vec = facets_combined['normal'][idx2_b]
    cos_i2 = np.abs(np.sum(-incoming2_n * normal2_vec, axis=1))
    refl1 = Ab[idx1_b]; refl2 = Ac[idx2_b]
    L1 = 2.0 * HUb[idx1_b]; illum1 = -incoming1_n; u1 = Ub[idx1_b]
    taper1 = _azimuth_sinc_taper(xp, wavelength, L1, Nb[idx1_b], u1, illum1)
    L2 = 2.0 * HUc[idx2_b]; illum2 = -incoming2_n; u2 = Uc[idx2_b]
    taper2 = _azimuth_sinc_taper(xp, wavelength, L2, normal2_vec, u2, illum2)
    amp_eff = refl1 * cos_i1 * taper1 * refl2 * cos_i2 * taper2
    R_ref_pl = np.linalg.norm(o - ref_pos)
    dR = R_equiv - R_ref_pl
    phase = np.exp(-4j * np.pi * to_numpy(freqs)[:, None] * dR[None, :] / C)
    c_new_tapered = np.sum(amp_eff[None, :] * phase, axis=1)
    order2_building_NEW_TAPERED[p, :] = c_new_tapered

print("running ASC box-projected closed form (retro-gated, split) at same scene/plat...")
s_asc, asc_stats = run_asc_box_projected_multibounce(
    xp, on_gpu, facets_b, facets_g, xp.asarray(plat), freqs, xp.asarray(ref_pos),
    ground_material=ground_material, return_components=True, include_order3=False,
    progress=False, leg2_occlusion_check=False, split_leg2_by_target=True,
    leg2_retroreflection_check=True, retro_beamwidth_mult=3.0,
    leg2_building_enabled=True, leg2_retro_taper=False)
leg2_building_asc = to_numpy(asc_stats['s_by_leg']['leg2_building'])
print(f"ASC counts: {asc_stats['counts']}")


def raw_coherence(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(np.vdot(a.ravel(), b.ravel())) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


print(f"\n=== position vs taper isolation, {n_pulses} pulses ===")
print(f"OLD (Cc[idx2], taper ON, pre-fix):             coherence = {raw_coherence(order2_building_OLD, leg2_building_asc):.4f}")
print(f"NEW-AS-SHIPPED (hit_pt2, taper OFF, this fix): coherence = {raw_coherence(order2_building_NEW_NOTAPER, leg2_building_asc):.4f}")
print(f"NEW-TAPERED (hit_pt2, taper ON):                coherence = {raw_coherence(order2_building_NEW_TAPERED, leg2_building_asc):.4f}")

print(f"\n=== hit_pt2 pulse-to-pulse smoothness check (persistent (idx1,idx2) keys) ===")
for key, pts in tracked_positions.items():
    if len(pts) < 2:
        continue
    ps = [p for p, _ in pts]
    coords = np.array([pt for _, pt in pts])
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    print(f"  key {key}: pulses {ps}, per-step position deltas (m) = {np.round(diffs, 4).tolist()}")
