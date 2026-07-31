"""Closed-form wall-to-(ground OR another building) double bounce for
point-ASC. Generalizes _reflect_and_intersect_ground_plane's "reflect off
the wall's normal, intersect a KNOWN, FIXED target analytically" trick:
instead of the target always being the ground plane z=0, it's now
"whichever is nearest: the ground plane, or any OTHER building's known
axis-aligned box" -- ray/AABB (slab method) intersection is itself
closed-form (no search), and there are only as many candidate boxes as
there are buildings (tens, not a dense ray grid), so this stays in the
same complexity class as the ground-only version: O(facets x buildings)
per pulse, still zero stochastic ray casting.

Once the box-level winner is known, the exact face hit (and hence the
outward normal / correct wall-vs-roof reflectivity) is resolved by
comparing the hit point against that box's own 5 outward faces --
still closed-form, not a facet search, and correct here because every
wall of a given building shares one reflectivity value in this scene
generator (scene_multi_building_adaptive.py).

No occlusion check on the box-hit path yet -- mirrors the
occlusion_check=False baseline that already matched SBR best on the
ground-only leg2 (run_multi_building_validation.py); whether occlusion
matters for wall-to-wall paths specifically is a separate question for
after this closes (or doesn't close) the SSIM gap on its own.
"""
import time
import numpy as np
from materials import effective_specular_reflectivity
from multibounce_demo import _azimuth_sinc_taper

C = 299792458.0


def to_numpy(xp, a):
    try:
        return xp.asnumpy(a)
    except AttributeError:
        return np.asarray(a)


def ray_box_intersect(xp, o, d, box_min, box_max):
    """o: (N,3), d: (N,3), box_min/box_max: (M,3) -> t_enter: (N,M),
    +inf where there's no valid forward intersection. Standard slab
    method, vectorized over all (ray, box) pairs at once."""
    o_ = o[:, None, :]
    d_ = d[:, None, :]
    bmin = box_min[None, :, :]
    bmax = box_max[None, :, :]
    eps = 1e-12
    parallel = xp.abs(d_) < eps
    d_safe = xp.where(parallel, eps, d_)
    t1 = (bmin - o_) / d_safe
    t2 = (bmax - o_) / d_safe
    tmin = xp.minimum(t1, t2)
    tmax = xp.maximum(t1, t2)
    inside = (o_ >= bmin) & (o_ <= bmax)
    tmin = xp.where(parallel, xp.where(inside, -xp.inf, xp.inf), tmin)
    tmax = xp.where(parallel, xp.where(inside, xp.inf, -xp.inf), tmax)
    t_enter = xp.max(tmin, axis=-1)
    t_exit = xp.min(tmax, axis=-1)
    valid = (t_enter <= t_exit) & (t_exit >= 0) & (t_enter > 1e-6)
    return xp.where(valid, t_enter, xp.inf)


def box_face_normal(xp, G, box_min, box_max):
    """Which of a box's 5 outward faces (4 walls + roof; floor excluded --
    these are convex boxes hit from outside) does G lie on? Returns the
    (N,3) outward normal and (N,) face index (0-3 = walls, 4 = roof)."""
    d_xmin = xp.abs(G[:, 0] - box_min[:, 0])
    d_xmax = xp.abs(G[:, 0] - box_max[:, 0])
    d_ymin = xp.abs(G[:, 1] - box_min[:, 1])
    d_ymax = xp.abs(G[:, 1] - box_max[:, 1])
    d_zmax = xp.abs(G[:, 2] - box_max[:, 2])
    dists = xp.stack([d_xmin, d_xmax, d_ymin, d_ymax, d_zmax], axis=1)
    face = xp.argmin(dists, axis=1)
    normals_table = xp.asarray([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return normals_table[face], face


def _reflect_and_intersect_scene(xp, C_bounce, N_bounce, d_in, ground_half_extent,
                                  box_min, box_max, source_building_id, exclude_building_id2=None):
    """Generalized version of _reflect_and_intersect_ground_plane: the
    reflected ray's target is whichever is nearer, the ground plane or
    another building's box (excluding the source facet's OWN building).
    Returns G (hit point), valid mask, d_out, building_hit (-1=ground,
    else building index), and the outward normal at the hit surface.

    exclude_building_id2: optional second per-ray building id to exclude,
    on top of source_building_id -- needed for order-3, where the ray
    now ORIGINATES exactly on some building's own surface (the leg-2 hit
    point) and would otherwise find a spurious t~0 self-intersection
    against that same box. Pass -1 for rays with nothing extra to
    exclude (e.g. leg-2 hit the ground, which isn't in box_min/box_max
    at all, so there's nothing to self-exclude there)."""
    d_out = d_in - 2.0 * xp.sum(d_in * N_bounce, axis=1, keepdims=True) * N_bounce

    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = xp.where(heading_down, d_out[:, 2], -1.0)
    t_ground = xp.where(heading_down, -C_bounce[:, 2] / d_out_z_safe, xp.inf)
    G_ground = C_bounce + xp.where(xp.isfinite(t_ground), t_ground, 0.0)[:, None] * d_out
    in_bounds = (xp.abs(G_ground[:, 0]) <= ground_half_extent) & (xp.abs(G_ground[:, 1]) <= ground_half_extent)
    ground_valid = heading_down & (t_ground > 1e-9) & in_bounds
    t_ground = xp.where(ground_valid, t_ground, xp.inf)

    t_box = ray_box_intersect(xp, C_bounce, d_out, box_min, box_max)   # (N, M)
    N_rays = C_bounce.shape[0]
    rows = xp.arange(N_rays)
    t_box = t_box.copy()
    t_box[rows, source_building_id] = xp.inf   # never reflect off your own building
    if exclude_building_id2 is not None:
        valid_excl2 = exclude_building_id2 >= 0
        safe_id2 = xp.where(valid_excl2, exclude_building_id2, 0)
        cur = t_box[rows, safe_id2]
        t_box[rows, safe_id2] = xp.where(valid_excl2, xp.inf, cur)

    all_t = xp.concatenate([t_ground[:, None], t_box], axis=1)        # (N, M+1)
    winner_idx = xp.argmin(all_t, axis=1)
    t_win = xp.take_along_axis(all_t, winner_idx[:, None], axis=1)[:, 0]
    valid = xp.isfinite(t_win)
    G = C_bounce + xp.where(valid, t_win, 0.0)[:, None] * d_out

    is_ground = winner_idx == 0
    building_hit = xp.where(is_ground, -1, winner_idx - 1)

    ground_normal = xp.tile(xp.asarray([[0.0, 0.0, 1.0]]), (N_rays, 1))
    bh_clamped = xp.clip(building_hit, 0, box_min.shape[0] - 1)
    bld_normal, bld_face = box_face_normal(xp, G, box_min[bh_clamped], box_max[bh_clamped])
    surf_normal = xp.where(is_ground[:, None], ground_normal, bld_normal)
    is_roof_hit = (~is_ground) & (bld_face == 4)

    return dict(G=G, valid=valid, d_out=d_out, building_hit=building_hit,
                is_ground=is_ground, is_roof_hit=is_roof_hit, surf_normal=surf_normal)


def _building_boxes_from_facets(xp, facets_buildings):
    """box_min/box_max per building (n_buildings,3), built straight from
    the host-side building_cx/cy/w/d/h metadata -- NOT multibounce_demo's
    own _building_aabbs, which assumes every building has the same
    facet count (n_facets // n_buildings) to recover building_id_of_facet.
    That assumption breaks here: the adaptive scene gives each building a
    DIFFERENT number of height-partitioned facets, so this uses the
    facet_building_id array scene_multi_building_adaptive.py now carries
    explicitly instead of re-deriving it."""
    cx = facets_buildings['building_cx']; cy = facets_buildings['building_cy']
    w = facets_buildings['building_w']; d = facets_buildings['building_d']; h = facets_buildings['building_h']
    box_min = xp.asarray(np.column_stack([cx - w / 2.0, cy - d / 2.0, np.zeros_like(cx)]))
    box_max = xp.asarray(np.column_stack([cx + w / 2.0, cy + d / 2.0, h]))
    return box_min, box_max


def _azimuth_sinc_taper_local(xp, wavelength, L, n_xy, look_xy):
    return _azimuth_sinc_taper(xp, wavelength, L, n_xy, look_xy)


def run_asc_box_projected_multibounce(xp, on_gpu, facets_buildings, facets_ground, plat, freqs, ref_pos,
                                       ground_material=None, return_components=False, include_order3=False):
    """Same leg-1 as run_asc_cached_multibounce (sensor->facet, closed
    form, unchanged). Leg 2 now targets whichever is nearer of {ground
    plane, any OTHER building's box} instead of always the ground --
    the fix for the scope gap compare_ground_points.py found (SBR's real
    order-2 bucket includes wall-to-wall bounces this couldn't represent
    before).

    include_order3=True adds a genuine third bounce: from the leg-2 hit
    point G2, reflect the outgoing direction off G2's OWN surface normal
    (ground normal if leg2 hit ground, that building's face normal if it
    hit a box) and call the SAME closed-form primitive again to find a
    THIRD surface (ground or yet another building), excluding both the
    original source building and (if leg2 hit one) that building too.
    Same complexity class as leg2 -- still O(facets x buildings) per
    pulse, one more hop, no ray tracing. Only candidates that were
    already valid2 (a real leg-2 path) are extended to leg3."""
    n_pulses = plat.shape[0]
    K = freqs.shape[0]

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    L_wall = 2.0 * facets_buildings['half_u']
    normal_xy = Nb[:, :2]
    fbid = facets_buildings['facet_building_id']

    half_extent_g = float(facets_ground.get('footprint_half_extent', 50.0))
    if ground_material is None:
        ground_material = facets_ground.get('material', 'dry_soil')
    wavelength = C / float(to_numpy(xp, freqs).mean())

    box_min, box_max = _building_boxes_from_facets(xp, facets_buildings)

    # per-building reflectivity, in case a box wins leg2: reuse that
    # building's own facet amp values (walls of one building share one
    # refl value in this scene generator, so averaging its facets'
    # amp recovers it exactly -- no new material assumption needed).
    n_buildings = facets_buildings['n_buildings']
    amp_by_building = xp.zeros(n_buildings)
    for b in range(n_buildings):
        mask = fbid == b
        amp_by_building[b] = Ab[mask].mean()

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    s_by_leg = None
    if return_components:
        s_by_leg = dict(leg1=xp.zeros((n_pulses, K), dtype=xp.complex128),
                         leg2=xp.zeros((n_pulses, K), dtype=xp.complex128))
        if include_order3:
            s_by_leg['leg3'] = xp.zeros((n_pulses, K), dtype=xp.complex128)

    t_total = 0.0
    for p in range(n_pulses):
        t0 = time.perf_counter()
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)

        look = Cb - o[None, :]
        R_asc = xp.linalg.norm(look, axis=1)
        d_in = look / R_asc[:, None]
        cos_inc1 = xp.sum(-d_in * Nb, axis=1)
        visible1 = cos_inc1 > 0
        taper1 = _azimuth_sinc_taper(xp, wavelength, L_wall, normal_xy, -d_in[:, :2])
        amp_eff1 = xp.where(visible1, Ab * xp.abs(cos_inc1) * taper1, 0.0)

        dR1 = R_asc - R_ref
        phase1 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR1) / C)
        leg1_contrib = (amp_eff1[None, :] * phase1).sum(axis=1)
        s[p, :] += leg1_contrib
        if return_components:
            s_by_leg['leg1'][p, :] = leg1_contrib

        hit = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, half_extent_g, box_min, box_max, fbid)
        G, valid_geom2, d_out = hit['G'], hit['valid'], hit['d_out']
        surf_normal, is_ground, building_hit = hit['surf_normal'], hit['is_ground'], hit['building_hit']

        cos_ground = xp.abs(xp.sum(d_out * surf_normal, axis=1))
        valid2 = visible1 & valid_geom2

        theta2 = xp.arccos(xp.clip(cos_ground, 0.0, 1.0))
        R_eff_ground = xp.asarray(effective_specular_reflectivity(ground_material, to_numpy(xp, theta2), wavelength))

        bh_clamped = xp.clip(building_hit, 0, n_buildings - 1)
        R_eff_building = amp_by_building[bh_clamped]
        R_eff2 = xp.where(is_ground, R_eff_ground, R_eff_building)

        leg2_len = xp.linalg.norm(G - Cb, axis=1)
        L_total = R_asc + leg2_len + xp.linalg.norm(o[None, :] - G, axis=1)
        amp_eff2 = xp.where(valid2, Ab * xp.abs(cos_inc1) * taper1 * R_eff2 * cos_ground, 0.0)

        R_equiv2 = L_total / 2.0
        dR2 = R_equiv2 - R_ref
        phase2 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR2) / C)
        leg2_contrib = (amp_eff2[None, :] * phase2).sum(axis=1)
        s[p, :] += leg2_contrib
        if return_components:
            s_by_leg['leg2'][p, :] = leg2_contrib

        if include_order3:
            # recurse the SAME primitive from the leg-2 hit point: reflect
            # d_out off the leg-2 surface's own normal, find the nearest
            # THIRD surface, excluding both the original source building
            # and (if leg2 hit one) that building too.
            hit3 = _reflect_and_intersect_scene(xp, G, surf_normal, d_out, half_extent_g,
                                                 box_min, box_max, fbid, exclude_building_id2=building_hit)
            G3, valid_geom3, d_out3 = hit3['G'], hit3['valid'], hit3['d_out']
            surf_normal3, is_ground3, building_hit3 = hit3['surf_normal'], hit3['is_ground'], hit3['building_hit']

            cos_leg3 = xp.abs(xp.sum(d_out * surf_normal3, axis=1))
            valid3 = valid2 & valid_geom3

            theta3 = xp.arccos(xp.clip(cos_leg3, 0.0, 1.0))
            R_eff_ground3 = xp.asarray(effective_specular_reflectivity(ground_material, to_numpy(xp, theta3), wavelength))
            bh3_clamped = xp.clip(building_hit3, 0, n_buildings - 1)
            R_eff_building3 = amp_by_building[bh3_clamped]
            R_eff3 = xp.where(is_ground3, R_eff_ground3, R_eff_building3)

            leg3_len = xp.linalg.norm(G3 - G, axis=1)
            L_total3 = R_asc + leg2_len + leg3_len + xp.linalg.norm(o[None, :] - G3, axis=1)
            amp_eff3 = xp.where(valid3, amp_eff2 * R_eff3 * cos_leg3, 0.0)

            R_equiv3 = L_total3 / 2.0
            dR3 = R_equiv3 - R_ref
            phase3 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR3) / C)
            leg3_contrib = (amp_eff3[None, :] * phase3).sum(axis=1)
            s[p, :] += leg3_contrib
            if return_components:
                s_by_leg['leg3'][p, :] = leg3_contrib

        t_total += (time.perf_counter() - t0)

    stats = dict(n_facets=Cb.shape[0], t_total_s=t_total, t_per_pulse_ms=t_total / n_pulses * 1000.0)
    if return_components:
        stats['s_by_leg'] = s_by_leg
    return s, stats
