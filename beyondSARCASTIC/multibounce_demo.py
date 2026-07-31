"""
Multi-bounce dense SBR: extends dense_sbr_demo.py's single-bounce ray
tracer with recursive bounce order 2 (double bounce, e.g. the classic
wall-ground dihedral) and order 3 (triple bounce), instead of stopping at
the first hit.

Why this exists as a SEPARATE script rather than a flag on
dense_sbr_demo.run_dense_sbr / sbr_vs_asc_compare.run_dense_sbr_timed:
those two functions produced the actual measured numbers behind the
deck's Tier 2 validation slides (SSIM/coherence/818x speedup). Keeping
multi-bounce tracing in its own file means adding it can't silently
change those already-scored results -- this imports make_building_scene,
ray_facet_intersect etc. from dense_sbr_demo.py but never touches
run_dense_sbr/run_dense_sbr_timed themselves.

Two real findings from actually building this, not assumed up front:

1. make_building_scene()'s buildings are isolated, non-adjacent,
   axis-aligned boxes with no re-entrant (concave) corners -- every
   building-only corner is convex and scatters away, not back toward the
   sensor. There is no wall-wall trihedral trap anywhere in that scene by
   construction. And there was no ray-traceable ground at all -- the
   existing ground layer (make_ground_clutter) is a diffuse point-
   scatterer background with no facet/normal/occlusion behavior, so nothing
   for a reflected ray to bounce off. Both true bottlenecks are fixed here
   via make_ground_facet() (dense_sbr_demo.py) -- a single large specular
   ground plane, kept entirely separate from the diffuse clutter layer so
   the existing single-bounce Tier 2 numbers are untouched.

2. Bounce order 1 still only ray-traces against BUILDING facets (exactly
   the existing dense_sbr_demo/sbr_vs_asc_compare convention) -- the
   ground facet only enters starting at bounce 2, when a ray reflected off
   a wall is traced onward. This is deliberate: it's what produces the
   physically expected wall-then-ground dihedral order (a downward-looking
   ray hits a wall first, reflects down toward the ground, reflects again
   back up toward the sensor) rather than degenerate ground-only "bounces".

Path scoring convention (see _score_paths): once ray tracing has
determined a bounce path (idx1, idx2[, idx3]) is BOTH geometrically valid
(each bounce lands in-bounds, front-face) AND has an unobstructed return
to the platform (a shadow ray fired from the final bounce point back to
the sensor, tested against the combined building+ground facet set), the
actual scattering contribution is evaluated analytically at the
PATH'S FACET CENTERS, not the literal per-ray hit points -- the same
"ray tracing decides visibility, closed-form geometry decides amplitude/
phase" convention sbr_vs_asc_compare.run_dense_sbr_timed already uses for
single-bounce, so multi-bounce stays an apples-to-apples extension of it
rather than a different modeling convention bolted on.

Amplitude per path is the product of (facet reflectivity x cos-incidence)
at each bounce -- consistent with the deck's Step 3 slide language
("each [bounce] multiplies by Fresnel reflectivity < 1"), using this
project's existing scalar per-facet reflectivity in place of full Fresnel
R_s/R_p (same simplification the rest of this codebase makes; see
materials.py for the "matrix product across bounces" polarimetric
upgrade this does NOT implement).

Round-trip range convention: existing single-bounce code assumes the
outbound and return paths are the same one-way distance R and uses
phase = exp(-j*4*pi*f*(R-R_ref)/c) (4*pi = 2*(2*pi), i.e. round trip =
2R). A multi-bounce path's true round-trip length L_total = sum of every
bounce-to-bounce segment, INCLUDING the return leg, is generally NOT
2x any single one-way distance. To stay compatible with the existing
R_ref-referenced dR convention (rather than inventing a second phase
formula), this script defines an equivalent one-way range
R_equiv = L_total / 2 and reuses the same exp(-j*4*pi*f*(R_equiv-R_ref)/c)
formula -- algebraically identical to using L_total directly with a
2*pi factor, just expressed in the codebase's existing units.

Usage:
    python3 multibounce_demo.py --footprint 200 --density 200 --rays 60 \
        --pulses 20 --freq 32 --max-bounces 3
"""

import argparse
import time
import json
import numpy as np

from dense_sbr_demo import (make_building_scene, make_ground_facet, concat_facets,
                             ray_facet_intersect, get_backend, make_aim_grid, C)
from materials import effective_specular_reflectivity


def to_numpy(a):
    if hasattr(a, 'get'):
        return a.get()
    return np.asarray(a)


def _return_visible(xp, hit_pt, hit_normal, o, facets_combined, eps):
    """Fire a shadow ray from each bounce point back toward the platform;
    a path is only a valid scattering contributor if nothing else in the
    scene (another building, or the ground) sits between the bounce point
    and the sensor."""
    dir_ret = o[None, :] - hit_pt
    dist_ret = xp.linalg.norm(dir_ret, axis=1)
    dist_ret_safe = xp.where(dist_ret > 0, dist_ret, 1.0)
    dir_ret_n = dir_ret / dist_ret_safe[:, None]
    o_ret = hit_pt + eps * hit_normal
    hit_pt_r, _, _, hit_mask_r, _ = ray_facet_intersect(xp, o_ret, dir_ret_n, facets_combined)
    t_block = xp.where(hit_mask_r, xp.linalg.norm(hit_pt_r - o_ret, axis=1), xp.inf)
    clear = t_block >= (dist_ret - 2.0 * eps)
    return clear


def _decode_unique_paths(xp, keys_valid, n_levels, F_c):
    """Dedupe a per-ray path key (base-F_c digit encoding of the bounce
    index sequence) down to the set of geometrically-distinct paths, and
    decode each unique key back into its (idx1, idx2[, idx3]) components.
    Fully vectorized -- no per-path Python loop."""
    uniq = xp.unique(keys_valid) if keys_valid.shape[0] > 0 else keys_valid
    levels = []
    rem = uniq
    for _ in range(n_levels):
        levels.append(rem % F_c)
        rem = rem // F_c
    levels.reverse()   # levels[0] = idx1, levels[1] = idx2, [levels[2] = idx3]
    return levels


def _score_paths(xp, o, ref_pos, freqs, level_idxs, level_centers, level_normals, level_amps,
                  ground_idx=None, ground_material=None, wavelength=None):
    """Given deduped path index arrays (one (U,) array per bounce level),
    evaluate the analytic per-path contribution at each level's FACET
    CENTER (not the literal ray-traced hit point) and sum coherently.
    Returns (K,) contribution and the path count U.

    Reflectivity per bounce is normally the facet's own static amp value
    (a building wall's pre-drawn reflectivity, unchanged from single-
    bounce). For any bounce landing on the ground facet (level_idxs[i] ==
    ground_idx), that static value is a meaningless placeholder (see
    make_ground_facet) -- it's replaced here with
    materials.effective_specular_reflectivity(ground_material, theta_i,
    wavelength), evaluated at THIS PATH'S OWN local incidence angle
    (already computed as cos_i below, not a scene-wide constant). A
    ground bounce at a rough-relative-to-wavelength incidence angle is
    physically suppressed toward zero here rather than contributing the
    same fixed number every wall-ground path would otherwise get."""
    U = int(level_idxs[0].shape[0])
    K = freqs.shape[0]
    if U == 0:
        return xp.zeros(K, dtype=xp.complex128), 0

    pts = [level_centers[i][level_idxs[i]] for i in range(len(level_idxs))]     # each (U,3)
    norms = [level_normals[i][level_idxs[i]] for i in range(len(level_idxs))]
    amps = [level_amps[i][level_idxs[i]] for i in range(len(level_idxs))]

    o_rep = xp.broadcast_to(o[None, :], (U, 3))
    path_pts = [o_rep] + pts + [o_rep]   # platform -> bounce1 -> ... -> bounceN -> platform

    L_total = xp.zeros(U)
    for i in range(len(path_pts) - 1):
        L_total = L_total + xp.linalg.norm(path_pts[i + 1] - path_pts[i], axis=1)

    amp_eff = xp.ones(U)
    for i in range(len(pts)):
        incoming = path_pts[i + 1] - path_pts[i]         # vector arriving at this bounce
        incoming_n = incoming / xp.linalg.norm(incoming, axis=1, keepdims=True)
        cos_i = xp.abs(xp.sum(-incoming_n * norms[i], axis=1))

        refl_i = amps[i]
        if ground_idx is not None:
            is_ground = level_idxs[i] == ground_idx
            if bool(to_numpy(is_ground).any()):
                theta_i = xp.arccos(xp.clip(cos_i, 0.0, 1.0))
                theta_i_np = to_numpy(theta_i)
                ground_refl_np = effective_specular_reflectivity(ground_material, theta_i_np, wavelength)
                ground_refl = xp.asarray(ground_refl_np)
                refl_i = xp.where(is_ground, ground_refl, refl_i)

        amp_eff = amp_eff * refl_i * cos_i

    R_ref = xp.linalg.norm(o - ref_pos)
    R_equiv = L_total / 2.0      # see module docstring: matches the existing 4*pi/round-trip=2R convention
    dR = R_equiv - R_ref
    phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)   # (K,U)
    contrib = (amp_eff[None, :] * phase).sum(axis=1)
    return contrib, U


def run_multibounce_sbr(xp, on_gpu, facets_buildings, facets_ground, plat, aim_pts, freqs, ref_pos,
                         max_bounces=3, eps=1e-3):
    n_pulses = plat.shape[0]
    K = freqs.shape[0]

    facets_combined = concat_facets(xp, facets_buildings, facets_ground)
    F_b = facets_buildings['center'].shape[0]
    F_c = facets_combined['center'].shape[0]
    ground_idx = F_c - 1   # ground is always the single facet appended last by concat_facets
    ground_material = facets_ground.get('material', 'dry_soil')
    wavelength = C / float(to_numpy(freqs).mean())   # fractional bandwidth here is modest enough
    # that treating wavelength as ~constant across it (rather than per-frequency-sample) is fine
    # for the roughness/specular check -- it's a slowly-varying geometric factor, not the fine
    # range-resolution phase term itself.

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    Cc, Nc, Ac = facets_combined['center'], facets_combined['normal'], facets_combined['amp']

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    counts = dict(order1=0, order2=0, order3=0)
    t_total = 0.0

    for p in range(n_pulses):
        o = plat[p]
        d1 = aim_pts - o[None, :]
        d1 = d1 / xp.linalg.norm(d1, axis=1, keepdims=True)
        R_ref = xp.linalg.norm(o - ref_pos)

        t0 = time.perf_counter()

        # ---- bounce 1: buildings only -- identical convention to
        # sbr_vs_asc_compare.run_dense_sbr_timed's single-bounce path ----
        hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(xp, o, d1, facets_buildings)
        hit_normal1 = Nb[idx1]

        hit_facets1 = idx1[hit_mask1]
        visible1 = xp.unique(hit_facets1) if hit_facets1.shape[0] > 0 else hit_facets1
        if visible1.shape[0] > 0:
            vpos = Cb[visible1]; vnorm = Nb[visible1]; vamp = Ab[visible1]
            look = vpos - o[None, :]
            look = look / xp.linalg.norm(look, axis=1, keepdims=True)
            cos_v = xp.abs(xp.sum(-look * vnorm, axis=1))
            R_v = xp.linalg.norm(vpos - o[None, :], axis=1)
            dR1 = R_v - R_ref
            amp_eff1 = vamp * cos_v
            phase1 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR1) / C)
            s[p, :] += (amp_eff1[None, :] * phase1).sum(axis=1)
            counts['order1'] += int(to_numpy(visible1).shape[0])

        # ---- bounce 2: reflect off bounce-1 hit, trace against buildings+ground ----
        d2 = d1 - 2.0 * xp.sum(d1 * hit_normal1, axis=1, keepdims=True) * hit_normal1
        o2 = hit_pt1 + eps * hit_normal1
        hit_pt2, hit_amp2, cos_inc2, hit_mask2_raw, idx2 = ray_facet_intersect(xp, o2, d2, facets_combined)
        hit_mask2 = hit_mask2_raw & hit_mask1
        hit_normal2 = Nc[idx2]

        clear2 = _return_visible(xp, hit_pt2, hit_normal2, o, facets_combined, eps)
        valid2 = hit_mask2 & clear2

        key2 = idx1.astype(xp.int64) * F_c + idx2.astype(xp.int64)
        key2_valid = key2[valid2]
        lvl2 = _decode_unique_paths(xp, key2_valid, 2, F_c)
        contrib2, n2 = _score_paths(xp, o, ref_pos, freqs, lvl2, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                     ground_idx=ground_idx, ground_material=ground_material,
                                     wavelength=wavelength)
        s[p, :] += contrib2
        counts['order2'] += n2

        if max_bounces >= 3:
            # ---- bounce 3: continue tracing from EVERY bounce-2 hit
            # point (whether or not its own order-2 return was clear --
            # the physical ray keeps bouncing regardless), trace against
            # buildings+ground again ----
            d3 = d2 - 2.0 * xp.sum(d2 * hit_normal2, axis=1, keepdims=True) * hit_normal2
            o3 = hit_pt2 + eps * hit_normal2
            hit_pt3, hit_amp3, cos_inc3, hit_mask3_raw, idx3 = ray_facet_intersect(xp, o3, d3, facets_combined)
            hit_mask3 = hit_mask3_raw & hit_mask2
            hit_normal3 = Nc[idx3]

            clear3 = _return_visible(xp, hit_pt3, hit_normal3, o, facets_combined, eps)
            valid3 = hit_mask3 & clear3

            key3 = key2 * F_c + idx3.astype(xp.int64)
            key3_valid = key3[valid3]
            lvl3 = _decode_unique_paths(xp, key3_valid, 3, F_c)
            contrib3, n3 = _score_paths(xp, o, ref_pos, freqs, lvl3, [Cb, Cc, Cc], [Nb, Nc, Nc], [Ab, Ac, Ac],
                                         ground_idx=ground_idx, ground_material=ground_material,
                                         wavelength=wavelength)
            s[p, :] += contrib3
            counts['order3'] += n3

        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t_total += (time.perf_counter() - t0)

    return s, dict(counts=counts, t_total_s=t_total,
                    t_per_pulse_ms=t_total / n_pulses * 1000.0,
                    n_facets_buildings=F_b, n_facets_combined=F_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=200.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--rays', type=int, default=60, help='ray grid side (rays^2/pulse)')
    ap.add_argument('--pulses', type=int, default=20)
    ap.add_argument('--freq', type=int, default=32)
    ap.add_argument('--standoff', type=float, default=8000.0)
    ap.add_argument('--altitude', type=float, default=3000.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    ap.add_argument('--ground-material', type=str, default='dry_soil',
                     choices=['dry_soil', 'concrete', 'metal'],
                     help='material assigned to the ray-traceable ground facet -- drives its '
                          'per-bounce reflectivity via materials.effective_specular_reflectivity '
                          '(Fresnel reflectivity x roughness-dependent specular/forward-scatter '
                          'factor, evaluated at each bounce\'s own local incidence angle), not a '
                          'fixed constant')
    ap.add_argument('--max-bounces', type=int, default=3, choices=[1, 2, 3])
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")
    print(f"Scene: {args.footprint}m x {args.footprint}m, {args.density}/km^2, "
          f"ground_material={args.ground_material}, max_bounces={args.max_bounces}")

    facets_buildings = make_building_scene(xp, args.footprint, args.density, seed=0)
    facets_ground = make_ground_facet(xp, args.footprint, material=args.ground_material)
    print(f"{facets_buildings['n_buildings']} buildings, {facets_buildings['n_facets']} building facets "
          f"+ 1 ground facet ({args.ground_material})")

    wavelength = C / args.fc
    sample_theta_deg = [10, 30, 50, 70, 85]   # incidence angle FROM NORMAL (0=straight down)
    rho_str = ", ".join(
        f"theta={t}deg->R_eff={float(effective_specular_reflectivity(args.ground_material, np.radians(t), wavelength)):.3f}"
        for t in sample_theta_deg
    )
    print(f"Ground effective specular reflectivity vs. incidence angle from normal "
          f"(wavelength={wavelength*100:.2f}cm): {rho_str}")

    squint_len = 400.0
    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([
        u, np.full(args.pulses, -args.standoff), np.full(args.pulses, args.altitude)
    ]))

    aim_pts, margin = make_aim_grid(xp, args.footprint, args.rays, args.standoff, args.altitude,
                                     max_height_m=40.0)
    if margin > 0.01 * args.footprint:
        print(f"aim grid padded +/-{margin:.1f}m beyond footprint for roof layover")

    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, args.freq))
    ref_pos = xp.zeros(3)

    print("Running multi-bounce dense SBR...")
    s, stats = run_multibounce_sbr(xp, on_gpu, facets_buildings, facets_ground, plat, aim_pts, freqs,
                                    ref_pos, max_bounces=args.max_bounces)

    print(json.dumps(stats, indent=2))
    print(f"\n=== Bounce path counts (summed across {args.pulses} pulses, deduped per pulse) ===")
    print(f"Order 1 (single bounce, buildings only):  {stats['counts']['order1']}")
    print(f"Order 2 (double bounce, e.g. wall-ground): {stats['counts']['order2']}")
    if args.max_bounces >= 3:
        print(f"Order 3 (triple bounce):                   {stats['counts']['order3']}")
    print(f"{stats['t_per_pulse_ms']:.2f} ms/pulse")

    out_path = 'multibounce_gpu.json' if on_gpu else 'multibounce_cpu.json'
    with open(out_path, 'w') as f:
        json.dump(dict(args=vars(args), stats=stats), f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
