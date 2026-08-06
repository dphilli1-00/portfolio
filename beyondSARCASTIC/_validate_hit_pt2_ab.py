"""Internal-only CPU A/B test (does NOT modify production code): calls the
REAL _score_paths (production function, now with building_pos_override) TWICE
on the SAME real ray-traced building-target order2 paths -- once with
building_pos_override (new fix) and once without (old facet-center
convention, i.e. exactly what the code did before this session's fix) --
and compares each against ASC's own leg2_building output. This isolates the
effect of the fix at a fixed small scale, instead of comparing across two
different scene configs (which the first smoke test conflated).
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet, ray_facet_intersect, concat_facets
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import (_building_aabbs, _return_visible, _decode_unique_paths_with_rep,
                               _score_paths, _reflect_and_intersect_ground_plane)
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
n_pulses = 6
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
order2_building_NEW = np.zeros((n_pulses, K), dtype=np.complex128)
order2_ground_check = np.zeros((n_pulses, K), dtype=np.complex128)  # unaffected-path regression check
n_bt_total = 0

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
    if not bool(is_ground2_u.any()) and not bool((~is_ground2_u).any()):
        continue

    idx1_u = lvl2[0]
    C1_u = Cb[idx1_u]; N1_u = Nb[idx1_u]
    d_in_u = C1_u - o[None, :]
    d_in_u = d_in_u / np.linalg.norm(d_in_u, axis=1, keepdims=True)
    G_analytic, _valid_geom_u, _ = _reflect_and_intersect_ground_plane(xp, C1_u, N1_u, d_in_u, ground_half_extent)
    hit_pt2_u = hit_pt2[rep2]

    o_x = xp.asarray(o)
    ref_pos_x = xp.asarray(ref_pos)

    # NEW (fixed): building rows scored at the literal traced hit point, ground rows unaffected
    contrib_new, n_new = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                       ground_idx=ground_idx, ground_material=ground_material,
                                       wavelength=wavelength,
                                       ground_pos_override=[None, xp.asarray(G_analytic)],
                                       building_pos_override=[None, xp.asarray(hit_pt2_u)],
                                       level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
    # OLD (pre-fix): building rows scored at Cc[idx2] (facet center), no override at all for them
    contrib_old, n_old = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                       ground_idx=ground_idx, ground_material=ground_material,
                                       wavelength=wavelength,
                                       ground_pos_override=[None, xp.asarray(G_analytic)],
                                       level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])

    building_target = valid2 & (~(idx2 >= ground_idx))
    n_bt_total += int(building_target.sum())

    g_np = to_numpy(is_ground2_u)
    if bool((~g_np).any()):
        not_ground2_u = ~is_ground2_u
        lvl2_b = [lvl2[0][not_ground2_u], lvl2[1][not_ground2_u]]
        c_new_b, _ = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2_b, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                   ground_idx=ground_idx, ground_material=ground_material, wavelength=wavelength,
                                   building_pos_override=[None, xp.asarray(hit_pt2_u[not_ground2_u])],
                                   level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
        c_old_b, _ = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2_b, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                   ground_idx=ground_idx, ground_material=ground_material, wavelength=wavelength,
                                   level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
        order2_building_NEW[p, :] = to_numpy(c_new_b)
        order2_building_OLD[p, :] = to_numpy(c_old_b)
    if bool(g_np.any()):
        lvl2_g = [lvl2[0][is_ground2_u], lvl2[1][is_ground2_u]]
        c_g, _ = _score_paths(xp, o_x, ref_pos_x, freqs, lvl2_g, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                               ground_idx=ground_idx, ground_material=ground_material, wavelength=wavelength,
                               ground_pos_override=[None, xp.asarray(G_analytic[g_np])],
                               level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
        order2_ground_check[p, :] = to_numpy(c_g)

print(f"total real building-target order2 hits across {n_pulses} pulses: {n_bt_total}")

print("running ASC box-projected closed form (retro-gated, split) at same scene/plat...")
s_asc, asc_stats = run_asc_box_projected_multibounce(
    xp, on_gpu, facets_b, facets_g, xp.asarray(plat), freqs, ref_pos_x,
    ground_material=ground_material, return_components=True, include_order3=False,
    progress=False, leg2_occlusion_check=False, split_leg2_by_target=True,
    leg2_retroreflection_check=True, retro_beamwidth_mult=3.0,
    leg2_building_enabled=True, leg2_retro_taper=False)
leg2_ground_asc = to_numpy(asc_stats['s_by_leg']['leg2_ground'])
leg2_building_asc = to_numpy(asc_stats['s_by_leg']['leg2_building'])
print(f"ASC counts: {asc_stats['counts']}")


def raw_coherence(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(np.vdot(a.ravel(), b.ravel())) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


coh_ground_check = raw_coherence(order2_ground_check, leg2_ground_asc)
coh_building_old = raw_coherence(order2_building_OLD, leg2_building_asc)
coh_building_new = raw_coherence(order2_building_NEW, leg2_building_asc)

print(f"\n=== A/B result, SAME real paths, SAME scene/pulses, only building-row position convention differs ===")
print(f"leg2_ground regression check (unaffected path, should be unchanged by this fix): {coh_ground_check:.4f}")
print(f"leg2_building vs OLD (facet-center Cc[idx2], pre-fix):     coherence = {coh_building_old:.4f}")
print(f"leg2_building vs NEW (literal hit_pt2, this session's fix): coherence = {coh_building_new:.4f}")
delta = coh_building_new - coh_building_old
print(f"delta: {delta:+.4f}  ({'IMPROVEMENT' if delta > 0.01 else ('regression' if delta < -0.01 else 'no meaningful change')})")
