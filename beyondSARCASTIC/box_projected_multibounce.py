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
from trihedral_asc_closed_form import asc_visible_envelope

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):   # no-op fallback if tqdm isn't installed
        return iterable

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
                                  box_min, box_max, source_building_id, exclude_building_id2=None,
                                  culling=None):
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
    at all, so there's nothing to self-exclude there).

    culling: optional (facet_indices_by_building, building_candidates) pair
    from _precompute_building_culling. When given, the O(F x M) dense box
    search is replaced with a per-building grouped search: each building's
    OWN facets only get tested against the (precomputed, static, usually
    far smaller than M) list of OTHER buildings within a rigorously-derived
    range, instead of every building in the scene. Validated (internal CPU
    check, this session) bit-identical to the unculled path on real scene
    data across 12 pulses / 416,484 rays -- zero mismatched valid/is_ground/
    building_hit flags, max position difference 0.0m -- as long as the
    range passed to _precompute_building_culling came from
    _rigorous_bounce_range (a proven upper bound), not a guessed distance.
    source_building_id must be the SAME array used to build
    facet_indices_by_building (always fbid in this module's call sites,
    for both leg2 and leg3 -- see that function's docstring) -- this isn't
    re-validated here, an inconsistent culling argument would silently
    misgroup rays."""
    d_out = d_in - 2.0 * xp.sum(d_in * N_bounce, axis=1, keepdims=True) * N_bounce
    N_rays = C_bounce.shape[0]

    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = xp.where(heading_down, d_out[:, 2], -1.0)
    t_ground = xp.where(heading_down, -C_bounce[:, 2] / d_out_z_safe, xp.inf)
    G_ground = C_bounce + xp.where(xp.isfinite(t_ground), t_ground, 0.0)[:, None] * d_out
    in_bounds = (xp.abs(G_ground[:, 0]) <= ground_half_extent) & (xp.abs(G_ground[:, 1]) <= ground_half_extent)
    ground_valid = heading_down & (t_ground > 1e-9) & in_bounds
    t_ground = xp.where(ground_valid, t_ground, xp.inf)

    if culling is None:
        t_box = ray_box_intersect(xp, C_bounce, d_out, box_min, box_max)   # (N, M)
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
    else:
        facet_indices_by_building, building_candidates = culling
        n_buildings = box_min.shape[0]
        best_t = xp.full(N_rays, xp.inf)
        best_building = xp.full(N_rays, -1, dtype=xp.int64)
        for b in range(n_buildings):
            idx = facet_indices_by_building[b]
            if idx.shape[0] == 0:
                continue
            cand = building_candidates[b]
            if cand.shape[0] == 0:
                continue
            t_sub = ray_box_intersect(xp, C_bounce[idx], d_out[idx], box_min[cand], box_max[cand])  # (n_b, n_cand)
            self_mask = cand == b
            if bool(xp.any(self_mask)):
                t_sub = t_sub.copy()
                t_sub[:, self_mask] = xp.inf   # never reflect off your own building
            if exclude_building_id2 is not None:
                excl_sub = exclude_building_id2[idx]
                excl_mask = cand[None, :] == excl_sub[:, None]
                t_sub = xp.where(excl_mask, xp.inf, t_sub)
            sub_win = xp.argmin(t_sub, axis=1)
            sub_t = xp.take_along_axis(t_sub, sub_win[:, None], axis=1)[:, 0]
            best_t[idx] = sub_t
            best_building[idx] = cand[sub_win]

        all_t = xp.concatenate([t_ground[:, None], best_t[:, None]], axis=1)
        winner_idx = xp.argmin(all_t, axis=1)   # 0 = ground, 1 = best building candidate
        t_win = xp.take_along_axis(all_t, winner_idx[:, None], axis=1)[:, 0]
        valid = xp.isfinite(t_win)
        G = C_bounce + xp.where(valid, t_win, 0.0)[:, None] * d_out
        is_ground = winner_idx == 0
        building_hit = xp.where(is_ground, -1, best_building)

    ground_normal = xp.tile(xp.asarray([[0.0, 0.0, 1.0]]), (N_rays, 1))
    bh_clamped = xp.clip(building_hit, 0, box_min.shape[0] - 1)
    bld_normal, bld_face = box_face_normal(xp, G, box_min[bh_clamped], box_max[bh_clamped])
    surf_normal = xp.where(is_ground[:, None], ground_normal, bld_normal)
    is_roof_hit = (~is_ground) & (bld_face == 4)

    return dict(G=G, valid=valid, d_out=d_out, building_hit=building_hit,
                is_ground=is_ground, is_roof_hit=is_roof_hit, surf_normal=surf_normal)


def _segment_occluded_by_any_box(xp, seg_start, seg_dir, seg_len, box_min, box_max, exclude_ids_list):
    """AABB-only occlusion test for the FINITE segment [seg_start, seg_start
    + seg_len*seg_dir] (e.g. the leg-2 hop from facet C to hit point G):
    True where some OTHER building's box intersects the open segment before
    it reaches its own endpoint, i.e. would have blocked a real ray along
    this path. exclude_ids_list: buildings to NOT count as occluders for
    this segment (its own source building, and the building it's aiming
    at -- the segment legitimately touches that box's surface at t~seg_len,
    which isn't occlusion, it's the destination). Box-only, not facet-level
    -- cheaper than a full ray_facet_intersect pass, same complexity class
    (O(F x n_buildings)) as the rest of this module; good enough to test
    whether occlusion is the real cause of the leg2 coherence collapse
    before paying for anything more precise."""
    t_enter = ray_box_intersect(xp, seg_start, seg_dir, box_min, box_max)   # (N, M)
    N = seg_start.shape[0]
    rows = xp.arange(N)
    t_enter = t_enter.copy()
    for excl in exclude_ids_list:
        valid_excl = excl >= 0
        safe = xp.where(valid_excl, excl, 0)
        cur = t_enter[rows, safe]
        t_enter[rows, safe] = xp.where(valid_excl, xp.inf, cur)
    blocked = t_enter < (seg_len[:, None] - 1e-6)
    return xp.any(blocked, axis=1)


def _leg1_occlusion_chunked(xp, o, d_in, R_asc, box_min, box_max, fbid, chunk_facets=3000):
    """Chunked wrapper around _segment_occluded_by_any_box for leg1's own
    C->o visibility test (does any OTHER building block the direct return
    from a facet's own center back to the sensor?). Not just a convenience
    wrapper -- _segment_occluded_by_any_box's own ray_box_intersect call
    broadcasts to (N, n_buildings, 3) intermediates, which is fine for
    leg2/leg3 (always called on an already-small, pre-filtered candidate
    set) but genuinely unsafe called on leg1's FULL facet count at once:
    measured directly, one pulse at F=146,053/200 buildings did not
    complete in 170s unchunked (CPU), vs 73.5s total (including the
    ray-facet ground-truth comparison run alongside it) chunked at 3000
    facets/chunk. Bounded memory regardless of scene size, same principle
    as this module's other per-chunk loops (the (F,K) envelope combine,
    task #41).

    fbid (per facet's own building id) is the exclude list: a facet's
    return-to-sensor ray legitimately starts ON its own building's surface
    and would otherwise falsely register as blocked by its own box at
    t~0 -- same exclude-list convention leg2_occlusion_check already uses
    for its own G->o segment."""
    F = d_in.shape[0]
    out = xp.zeros(F, dtype=bool)
    o_b = xp.broadcast_to(o, (F, 3))
    for cs in range(0, F, chunk_facets):
        ce = min(cs + chunk_facets, F)
        out[cs:ce] = _segment_occluded_by_any_box(
            xp, o_b[cs:ce], d_in[cs:ce], R_asc[cs:ce], box_min, box_max,
            exclude_ids_list=[fbid[cs:ce]])
    return out


def _point_to_aabb_min_range(xp, o, box_min, box_max):
    """Minimum possible Euclidean distance from point o to each AABB
    (box_min[i], box_max[i]) -- the standard clamp-to-box formula: 0 on an
    axis where o already lies between box_min/box_max, else the gap to the
    nearest face on that axis. RIGOROUS, not empirical: no point inside the
    box can be closer to o than this value, by construction (same "prove it,
    don't eyeball it" standard as _rigorous_bounce_range's leg2 bound)."""
    dx = xp.maximum(box_min[:, 0] - o[0], 0.0) + xp.maximum(o[0] - box_max[:, 0], 0.0)
    dy = xp.maximum(box_min[:, 1] - o[1], 0.0) + xp.maximum(o[1] - box_max[:, 1], 0.0)
    dz = xp.maximum(box_min[:, 2] - o[2], 0.0) + xp.maximum(o[2] - box_max[:, 2], 0.0)
    return xp.sqrt(dx * dx + dy * dy + dz * dz)


def _leg1_occlusion_chunked_culled(xp, o, d_in, R_asc, box_min, box_max, fbid, chunk_facets=3000, margin=1e-3):
    """Performance-only variant of _leg1_occlusion_chunked -- same result,
    fewer candidate buildings tested per chunk. That function tests every
    facet against ALL n_buildings boxes unconditionally (_segment_occluded_
    by_any_box's own ray_box_intersect call has no range filter at all).

    But a building can only occlude the straight-line return from a facet at
    range R if some part of that building's box is closer to the sensor than
    R -- i.e. _point_to_aabb_min_range(o, box) < R. This is exact: if every
    point of a box has range >= R, the box cannot lie on the segment
    [o, facet], independent of scene layout. (Same rigorous-bound standard
    as leg2's _rigorous_bounce_range/leg2_culled_search.)

    Filtering happens once per FACET CHUNK (using that chunk's own max
    facet range as the bound), not per individual facet -- keeps this a
    plain boolean mask + gather instead of a ragged per-facet candidate
    list. This scene generator lays facets out building-contiguously, so a
    chunk is typically drawn from only a few spatially-clustered buildings
    already, which caps the achievable filtering here below what a fully
    range-sorted pass would give -- but it costs nothing extra to obtain
    (min_range_b is computed once per PULSE, not per chunk) and only ever
    shrinks the M dimension of the box test; a superset of true candidates
    can't change the answer, only the cost of computing it. margin=1e-3
    guards against excluding a building sitting exactly at the chunk's max
    range due to float roundoff -- not a physical tolerance."""
    F = d_in.shape[0]
    out = xp.zeros(F, dtype=bool)
    o_b = xp.broadcast_to(o, (F, 3))
    min_range_b = _point_to_aabb_min_range(xp, o, box_min, box_max)   # (n_buildings,) -- once per pulse
    for cs in range(0, F, chunk_facets):
        ce = min(cs + chunk_facets, F)
        chunk_max_R = float(to_numpy(xp, R_asc[cs:ce]).max())
        candidate = min_range_b <= (chunk_max_R + margin)
        cand_idx = xp.nonzero(candidate)[0]
        box_min_c = box_min[cand_idx]
        box_max_c = box_max[cand_idx]
        # remap this chunk's facet building ids into the filtered candidate
        # index space -- exclude_ids_list indices must refer to POSITIONS
        # in box_min_c/box_max_c, not the original 0..n_buildings-1 ids.
        # A facet's own building is always in the candidate set (the box
        # contains the facet itself, so the box's min range can't exceed
        # the facet's own range) so this never needs a -1 fallback for that
        # case, but _segment_occluded_by_any_box already handles excl<0
        # safely regardless.
        orig_to_filtered = xp.full(box_min.shape[0], -1, dtype=xp.int64)
        orig_to_filtered[cand_idx] = xp.arange(cand_idx.shape[0])
        fbid_chunk_remapped = orig_to_filtered[fbid[cs:ce]]
        out[cs:ce] = _segment_occluded_by_any_box(
            xp, o_b[cs:ce], d_in[cs:ce], R_asc[cs:ce], box_min_c, box_max_c,
            exclude_ids_list=[fbid_chunk_remapped])
    return out


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


def _azimuth_sinc_taper_local(xp, wavelength, L, normal, u_hat, look_dir):
    return _azimuth_sinc_taper(xp, wavelength, L, normal, u_hat, look_dir)


def _rigorous_bounce_range(xp, Cb, Nb, plat, box_max, margin=1.05):
    """RIGOROUS (not empirical) upper bound on how far a leg-2 reflected ray
    could possibly need to travel to hit a building box, across every pulse
    in this run. Two exact per-ray bounds, nearer of the two applies:

    1. Ground bound: if the ray heads downward, it hits the ground at a
       fixed, exactly-computable t_ground. Any building beyond that range
       LOSES to the ground hit regardless of whether it's geometrically
       intersected, so it can never change the outcome.
    2. Height-envelope bound: every building box has z in [0,
       max_building_height] (this scene's actual box_max, not an assumed
       constant). If the ray is climbing, it exits every building's height
       envelope at a fixed t = (max_height - start_z) / d_out_z; beyond
       that, no box can be intersected no matter how far the ray extends.

    Rays that are (nearly) level get no bound from this reasoning and are
    excluded from the max (not silently treated as zero) -- validated
    (internal CPU check, this session) that this reasoning gives a bound
    that safely covers the TRUE worst-case exact-search hit distance with a
    small, expected margin (116.05m rigorous vs 106.28m actual max, on a
    representative scene) -- not just "usually enough," a real upper bound.
    margin: additional multiplicative safety factor on top of the already-
    rigorous value (default 5%), cheap insurance against floating-point
    edge cases at the boundary.

    Computed across ALL pulses that will actually run (not sampled), since
    it's O(F) per pulse with no box search -- cheap relative to the O(F x M)
    savings this enables, and exact for this specific run's own geometry
    rather than an assumption that has to be re-validated if the scene or
    platform track changes."""
    max_building_height = float(to_numpy(xp, xp.max(box_max[:, 2])))
    worst = 0.0
    n_pulses = plat.shape[0]
    for p in range(n_pulses):
        o = plat[p]
        look = Cb - o[None, :]
        R_asc = xp.linalg.norm(look, axis=1)
        d_in = look / R_asc[:, None]
        cos_inc1 = xp.sum(-d_in * Nb, axis=1)
        visible1 = cos_inc1 > 0
        d_out = d_in - 2.0 * xp.sum(d_in * Nb, axis=1, keepdims=True) * Nb

        heading_down = d_out[:, 2] < -1e-9
        d_out_z_safe = xp.where(heading_down, d_out[:, 2], -1.0)
        t_ground = xp.where(heading_down, -Cb[:, 2] / d_out_z_safe, xp.inf)

        climbing = d_out[:, 2] > 1e-9
        d_out_z_safe_up = xp.where(climbing, d_out[:, 2], 1.0)
        t_height = xp.where(climbing, (max_building_height - Cb[:, 2]) / d_out_z_safe_up, xp.inf)

        per_ray_bound = xp.minimum(t_ground, t_height)
        finite = per_ray_bound[visible1 & xp.isfinite(per_ray_bound)]
        if finite.shape[0] > 0:
            worst = max(worst, float(to_numpy(xp, xp.max(finite))))
    return worst * margin


def _precompute_building_culling(xp, fbid, box_min, box_max, n_buildings, max_range):
    """Static, pulse-independent (building positions and facet ownership
    don't change pulse to pulse -- only platform position does), so this
    runs ONCE for the whole call, not per pulse. Returns:
      facet_indices_by_building: list of (xp) index arrays, facets owned by
        each building (grouped by fbid, same array used as
        source_building_id in every _reflect_and_intersect_scene call, leg2
        AND leg3, so this grouping is valid for both).
      building_candidates: list of (xp) index arrays, OTHER buildings within
        max_range of each building -- tested building-footprint-extent to
        building-footprint-extent (center distance minus both buildings'
        own footprint half-diagonals), not center-to-center, so a facet
        anywhere on its own building and a hit anywhere on the target
        building's footprint both stay safely covered, not just the two
        buildings' centroids."""
    fbid_np = to_numpy(xp, fbid)
    box_min_np, box_max_np = to_numpy(xp, box_min), to_numpy(xp, box_max)
    centers_xy = 0.5 * (box_min_np[:, :2] + box_max_np[:, :2])
    extents_xy = box_max_np[:, :2] - box_min_np[:, :2]
    radii = 0.5 * np.linalg.norm(extents_xy, axis=1)
    dxy = centers_xy[:, None, :] - centers_xy[None, :, :]
    center_dist = np.linalg.norm(dxy, axis=2)
    inflated_dist = center_dist - radii[:, None] - radii[None, :]
    building_candidates = [xp.asarray(np.nonzero(inflated_dist[b] < max_range)[0]) for b in range(n_buildings)]
    facet_indices_by_building = [xp.asarray(np.nonzero(fbid_np == b)[0]) for b in range(n_buildings)]
    return facet_indices_by_building, building_candidates


def run_asc_box_projected_multibounce(xp, on_gpu, facets_buildings, facets_ground, plat, freqs, ref_pos,
                                       ground_material=None, return_components=False, include_order3=False,
                                       progress=False, low_precision_envelope=False, free_pool_every=None,
                                       leg2_occlusion_check=False, split_leg2_by_target=False,
                                       leg2_retroreflection_check=False, retro_beamwidth_mult=3.0,
                                       leg2_building_enabled=True, leg2_retro_taper=False,
                                       leg2_culled_search=False, culled_range_margin=1.05,
                                       leg1_occlusion_check=False, leg1_occlusion_chunk_facets=3000,
                                       leg1_occlusion_culled=False, profile=False):
    """leg1_occlusion_check: default False. Opt-in fix for the gap this
    module's own docstring flagged from the start and decompose_sbr_asc_
    coherence.py's own module docstring names explicitly: leg1's visible1
    is a pure backface cull (cos_inc1 > 0) -- no check for whether some
    OTHER building sits between the platform and this facet. SBR's dense
    ray tracer gets this for free (ray_facet_intersect finds the nearest
    hit; a blocked facet just never registers).

    Measured directly (this session, real 1000m/200-building/146,053-facet
    cache, 10 pulses sampled evenly across the aperture): 11.3-11.6% of
    every ASC-leg1-visible facet-pulse is actually occluded by another
    building per SBR's own ray trace -- remarkably stable across the whole
    aperture (not a grazing-angle artifact, a structural property of how
    densely these buildings are packed). NOT evenly distributed: 3
    buildings (of 200) were 100% occluded -- ASC was drawing entire
    buildings that should be completely invisible -- and several more were
    87-97% occluded. Two of the worst-occluded buildings (by id) were
    independently also the worst-by-SSIM buildings in plot_tier2_from_
    cache.py's per-building comparison on the same cache, before this fix.

    Implementation: _leg1_occlusion_chunked, the SAME AABB-only box test
    (_segment_occluded_by_any_box) leg2_occlusion_check already uses for
    its own G->o return leg, applied to leg1's C->o leg instead, chunked
    over facets (see that function's own docstring -- calling it unchunked
    on leg1's full facet count measured as not completing a single pulse
    in 170s; chunked, ~70s including a ground-truth comparison run
    alongside it). Box-level, not facet-level -- but validated bit-
    identical to the exact per-facet ray trace on real scene data (this
    session: pulse 0 of the same 1000m/200-building cache, zero false
    positives, zero false negatives against ray_facet_intersect) -- these
    buildings are fully wall-tiled boxes with no real gaps, so "the ray
    crosses another building's AABB" and "the ray hits a real facet on
    that building" turned out to coincide exactly here, not just
    approximately.

    leg1_occlusion_chunk_facets: chunk size for the above, default 3000 --
    tune down if you OOM on a smaller GPU, up if you have headroom and
    want fewer chunk-loop iterations.

    leg1_occlusion_culled: default False. Performance-only sibling of
    leg2_culled_search, for leg1_occlusion_check instead of leg2's box
    search -- _leg1_occlusion_chunked (used when this is False) tests every
    facet against ALL n_buildings boxes unconditionally, same "no range
    filter at all" gap leg2_culled_search fixed for the leg2 search. Uses
    _leg1_occlusion_chunked_culled instead: a RIGOROUS (not empirical)
    point-to-AABB minimum-range bound (_point_to_aabb_min_range) computed
    once per pulse, then per facet-chunk, only buildings whose box could
    possibly be closer to the sensor than that chunk's own farthest facet
    are tested -- exact, not a heuristic, because a box entirely farther
    from the sensor than a facet's own range geometrically cannot lie on
    the straight-line segment between them. Validate bit-identical to the
    unculled path before trusting it at scale, same standard as every
    other culled-search flag in this module.

    Cascades to leg2/leg3 correctly, not just leg1's own count: valid2 is
    gated on visible1 (a facet blocked on its way IN can't be illuminated
    at all, so it can't source a real double-bounce either, any more than
    it can reflect straight back) -- this was already true of the
    pre-existing visible1 before this flag existed, so enabling this check
    tightens leg2/leg3 for free, correctly, through the same gate.

    leg2_culled_search: default False. Opt-in performance fix for
    _reflect_and_intersect_scene's leg2/leg3 box search, which tests every
    source facet's reflected ray against EVERY building's box unconditionally
    -- O(F x n_buildings) per pulse, dense, no culling at all (unlike SBR's
    ray_facet_intersect, which gets a building-level AABB prefilter). That
    was fine at tens of buildings but degrades as building count grows --
    measured directly: ASC's SBR-relative speedup dropped from ~6x (500m/50
    buildings) to ~3.85x (1000m/200 buildings) purely from this, unrelated
    to any physics change. When True, computes a RIGOROUS (not empirical)
    global range once per run (_rigorous_bounce_range -- exact per-ray
    ground-hit/height-envelope bounds, worst case across every pulse that
    will actually run) and groups the box search by building
    (_precompute_building_culling), so each building's own facets only test
    against the (usually far smaller) list of other buildings within that
    range instead of every building in the scene. Validated bit-identical
    to the unculled path on real scene data (internal CPU check, this
    session: 12 pulses / 416,484 rays, zero mismatched valid/is_ground/
    building_hit, max position difference 0.0m) -- a pure speedup, not an
    approximation, as long as the range came from _rigorous_bounce_range
    and not a guessed distance. culled_range_margin: extra multiplicative
    safety factor on top of the already-rigorous bound (default 5%).

    Same leg-1 as run_asc_cached_multibounce (sensor->facet, closed
    form, unchanged). Leg 2 now targets whichever is nearer of {ground
    plane, any OTHER building's box} instead of always the ground --
    the fix for the scope gap compare_ground_points.py found (SBR's real
    order-2 bucket includes wall-to-wall bounces this couldn't represent
    before).

    progress: tqdm bar over the per-pulse loop (elapsed/ETA/it-per-sec,
    plus running leg1/2/3 valid-facet counts in the postfix) -- same
    convention as run_multibounce_sbr's progress flag. Added because this
    loop is O(facets x buildings) per pulse (module docstring) with NO
    per-pulse feedback otherwise: on a big scene (tens of thousands of
    facets, tens of buildings) this can legitimately run for many minutes
    with nothing printed, indistinguishable from a hang without this.
    False (default) prints nothing, same as before.

    low_precision_envelope: passed through to asc_amplitude_envelope's
    own low_precision flag (see its docstring) -- float32/complex64 for
    the envelope computation instead of float64/complex128, halving the
    single largest per-pulse array at real-scene scale. Opt-in, default
    False: this codebase's canonical-shape validation coherence numbers
    were all measured at float64 and haven't been re-checked at float32,
    so this isn't a verified-safe default, just an available tradeoff.

    leg2_retro_taper: default False. This supersedes leg2_retroreflection_check
    as the real fix -- that binary gate measurably helped wall->building
    (leg2_building coherence 0.06->0.63 at 500m) but had ZERO effect on
    wall->ground (0.0991 unchanged before/after gating at 500m, 0.0562 ->
    0.0561 at 1000m), which is the tell that a hard yes/no cutoff is the
    wrong shape of fix, not that ground has no problem.

    Root cause, found by actually reading how leg1/leg2 share their
    envelope: box_projected_multibounce's chunked combine loop computes
    ONE envelope per facet-chunk (asc_visible_envelope, using L_az=L_wall,
    u_hat=Ub -- the WALL's own azimuth persistence/illumination taper) and
    reuses it UNCHANGED for both leg1_contrib and leg2_contrib. That's
    correct for leg1 (a single facet's own reflectivity really does vary
    with viewing angle relative to its own axis) but wrong for leg2: a
    dihedral corner's return strength is a property of the WHOLE two-
    bounce path's alignment with the narrow retroreflective geometry, not
    of the source facet's own illumination. Leg2 currently has NO
    persistence taper of its own at all -- every geometrically-findable
    wall-target pair contributes at essentially full strength regardless
    of how far the actual double-bounce path is from the narrow angular
    region where a real corner returns coherently. That is the over-
    counted "diffuse clutter floor instead of a few sharp corners"
    pattern visible in the backprojected leg2 images.

    When True: computes the same second-bounce misalignment angle
    leg2_retroreflection_check uses (true specular return direction vs.
    actual direction to the sensor), but instead of a hard cutoff, applies
    it as a continuous, frequency-dependent sinc taper --
    sinc(2*pi*f*L_wall*sin(misalignment)/c) -- the SAME mathematical form
    already validated for leg1's own taper (asc_amplitude_envelope), just
    pointed at the corner's own misalignment angle instead of the wall's
    illumination angle. Applied multiplicatively on top of (not instead
    of) the existing envelope, inside the chunked combine loop. Can be
    combined with leg2_retroreflection_check (belt-and-suspenders: hard
    floor plus graceful falloff) or used alone.

    leg2_building_enabled: default True. Opt-out comparison: when False,
    leg2 targets ONLY the ground (is_ground) -- wall->building candidates
    are dropped entirely, falling back to the originally-validated
    ground-only leg2 case. Added because even WITH
    leg2_retroreflection_check=True, leg2_building's own coherence vs SBR
    order2 stayed poor at the 1000m/200-building scale (0.045, barely
    above leg2_ground's 0.056) while the COMBINED coherence improvement
    there was better explained by leg2's total energy shrinking (fewer
    contributing terms diluting a still-low-correlation signal) than by
    building-target leg2 becoming genuinely more correct. Worth checking
    whether leg1 + leg2_ground alone matches or beats leg1 + leg2 (retro-
    gated) -- if so, the wall-to-building generalization isn't earning its
    complexity at this density and a simpler ground-only fallback may be
    the more defensible fix.

    leg2_retroreflection_check: default False. This is the fix, not just
    another ablation -- both occlusion checks (C->G, then G->o) left
    leg2's coherence vs SBR order2 completely unchanged, and the
    split_leg2_by_target diagnostic then showed WHY: on the 500m scene,
    wall->building bounces are only ~10% of leg2's event count but 99.8%
    of its ENERGY, and their coherence with order2 (0.06) is nearly as bad
    as leg2 combined -- a small number of "coincidentally nearest"
    building pairs are producing large, SBR-uncorrelated returns.

    The root cause: this function's whole approach (sum three segment
    lengths, treat the total as one coherent monostatic path) is only
    EXACT for a true 90-degree corner -- a vertical wall meeting flat
    ground is retroreflective by geometric identity, for any incidence
    angle, which is why the original ground-only leg2 never needed to
    verify the return path explicitly. Generalizing the TARGET SEARCH to
    "nearest of ground or any building" (this module's whole point) did
    NOT generalize the PHYSICS -- an arbitrary building pair generally
    does not meet at 90 degrees, so there's no guarantee the second
    surface's own law of reflection sends the ray back toward the sensor.
    Every nearest-box pair still gets scored as if it were a valid corner.

    When True: computes the TRUE second-bounce specular direction (mirror-
    reflecting d_out off the target surface's own normal, not assuming
    it retroreflects), compares it against the actual direction from G to
    the sensor, and zeros amp_eff2_geom wherever the misalignment angle
    exceeds retro_beamwidth_mult * (wavelength / L_wall) -- a diffraction-
    limited tolerance built from the SAME source-facet width (L_wall)
    already used for this codebase's azimuth sinc taper elsewhere (an
    approximation, not a rigorous two-bounce Fresnel-zone derivation, but
    consistent with how this codebase already treats angular sensitivity
    around a specular peak). For a true wall-ground corner this is exactly
    satisfied (retro angle ~0) regardless of L_wall, so ground-target leg2
    should pass through this gate essentially unchanged; wall-to-building
    pairs that don't happen to form a near-90-degree corner should mostly
    get suppressed. retro_beamwidth_mult: how many beamwidths of
    misalignment tolerance to allow (wider = more permissive gate).

    split_leg2_by_target: default False. Opt-in diagnostic (task #44/#45
    investigation) that, when return_components=True, additionally returns
    s_by_leg['leg2_ground'] and s_by_leg['leg2_building'] -- leg2 split by
    whether _reflect_and_intersect_scene's winning target was the ground
    plane (a true 90-degree wall-ground dihedral, retroreflective by
    geometric identity -- this function's "assume a coherent direct
    return" approximation is EXACT there) vs another building's box (an
    arbitrary building pair generally does NOT meet at 90 degrees, so
    there's no guarantee the target surface's own law of reflection sends
    the ray back toward the sensor -- nothing here checks that, every
    nearest-box pair gets scored as if it were a valid corner regardless).
    counts['leg2_ground'] / counts['leg2_building'] are populated too. Both
    occlusion ablations above (leg2_occlusion_check on C->G, then on the
    return leg) left leg2's ~0.02 coherence vs SBR order2 completely
    unchanged -- this split exists to test the leading remaining
    hypothesis (missing retroreflection validity, not occlusion) instead
    of continuing to guess.

    leg2_occlusion_check: default False. Opt-in ablation added to test why
    decompose_sbr_asc_coherence.py found leg2 raw coherence vs SBR order2
    collapsing to ~0.02 (essentially decorrelated) on the 1000m/200-
    building scene, despite leg1 staying at ~0.93.

    First attempt at this (now removed) tested the C->G outbound hop for
    occlusion by other buildings' boxes and found it was a complete no-op
    -- zero facets pruned, identical coherence. That's not a bug: G is
    chosen by _reflect_and_intersect_scene's own argmin over {ground, all
    OTHER buildings' boxes}, i.e. G is BY CONSTRUCTION the first thing the
    outgoing ray hits. Nothing can occlude C->G that argmin wouldn't
    already have picked as the (nearer) winner instead.

    The segment that was never checked at all is the RETURN leg, G back
    up to the sensor at o: after bouncing off a low wall or the ground,
    does the path back to a distant, elevated platform clear every OTHER
    building, or does something else in a dense field stand in the way?
    Nothing in this function's pre-existing geometry answers that -- the
    phase/range term (L_total) assumes a clear line regardless. When
    leg2_occlusion_check=True, this does a cheap AABB-only test
    (_segment_occluded_by_any_box) on G->o against every building's box
    (excluding only building_hit itself, since G sits ON that box's
    surface and would otherwise self-intersect at t~0) and zeros
    amp_eff2_geom wherever blocked. Box-level, not facet-level -- cheaper
    than real ray tracing, same O(F x n_buildings) complexity as the rest
    of this module.

    include_order3=True adds a genuine third bounce: from the leg-2 hit
    point G2, reflect the outgoing direction off G2's OWN surface normal
    (ground normal if leg2 hit ground, that building's face normal if it
    hit a box) and call the SAME closed-form primitive again to find a
    THIRD surface (ground or yet another building), excluding both the
    original source building and (if leg2 hit one) that building too.
    Same complexity class as leg2 -- still O(facets x buildings) per
    pulse, one more hop, no ray tracing. Only candidates that were
    already valid2 (a real leg-2 path) are extended to leg3.

    free_pool_every: on_gpu only -- call cupy's
    get_default_memory_pool().free_all_blocks() every this-many pulses.
    Default None (disabled). Originally added as a hypothesis-driven fix
    for cupy memory-pool fragmentation (varying per-pulse allocation
    sizes as the visible-facet count shifts pulse to pulse), mirroring a
    real fragmentation-driven OOM already fixed once between the SBR and
    closed-form stages elsewhere in this pipeline. BUT: when actually
    tested live (this session), the hypothesis was directly REFUTED --
    pool_mb (used/total) was measured IDENTICAL between two consecutive
    pulses while iteration time still grew, meaning fragmentation was
    not, in fact, what was happening in that run. The real cause turned
    out to be a completely different problem (asc_visible_envelope's
    sparse gather/scatter path, see task #38/#39). Left available and
    off by default rather than deleted, because periodic freeing isn't
    WRONG, just unproven and not free (each free forces the next
    allocation to re-request memory from the driver instead of hitting
    the pool, a real cost) -- don't turn this back on without your own
    pool_mb evidence that fragmentation is actually growing in your
    specific run; the postfix instrumentation below will show you."""
    n_pulses = plat.shape[0]
    K = freqs.shape[0]

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    Ub = facets_buildings['u_hat']
    L_wall = 2.0 * facets_buildings['half_u']
    normal_xy = Nb[:, :2]
    fbid = facets_buildings['facet_building_id']
    F = Cb.shape[0]

    half_extent_g = float(facets_ground.get('footprint_half_extent', 50.0))
    if ground_material is None:
        ground_material = facets_ground.get('material', 'dry_soil')
    wavelength = C / float(to_numpy(xp, freqs).mean())

    box_min, box_max = _building_boxes_from_facets(xp, facets_buildings)

    culling = None
    if leg2_culled_search:
        rigorous_range = _rigorous_bounce_range(xp, Cb, Nb, plat, box_max, margin=culled_range_margin)
        culling = _precompute_building_culling(xp, fbid, box_min, box_max,
                                                facets_buildings['n_buildings'], rigorous_range)
        if progress:
            print(f"  leg2_culled_search: rigorous range={rigorous_range:.1f}m")

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
        if split_leg2_by_target:
            s_by_leg['leg2_ground'] = xp.zeros((n_pulses, K), dtype=xp.complex128)
            s_by_leg['leg2_building'] = xp.zeros((n_pulses, K), dtype=xp.complex128)

    t_total = 0.0
    counts = dict(leg1=0, leg2=0, leg3=0)
    # profile=True: per-stage wall-clock breakdown of one pulse's cost --
    # occlusion search (leg1 + leg2's reflect/intersect box search) vs. the
    # envelope (sinc/taper) computation vs. the phase-exp/multiply/reduce
    # combine, since none of this was ever measured separately before (all
    # prior speedup work -- culling, chunk-size sweeps -- was guessing at
    # which of these dominates). Each checkpoint calls
    # xp.cuda.Stream.null.synchronize() before stopping its clock -- cupy
    # ops queue asynchronously, so an un-synchronized perf_counter() delta
    # measures how fast Python enqueues kernels, not how long they take to
    # run. Zero overhead when profile=False (no sync calls, no dict
    # bookkeeping beyond the one dict literal below).
    prof = dict(leg1_occlusion=0.0, reflect_intersect=0.0, leg2_occlusion=0.0,
                retro_check=0.0, envelope=0.0, phase1=0.0, phase2=0.0, other=0.0) if profile else None

    def _mark(t_prev):
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        return time.perf_counter()

    pbar = tqdm(range(n_pulses), desc="ASC pulses", disable=not progress)
    for p in pbar:
        t0 = time.perf_counter()
        if profile:
            tp = _mark(t0)
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)

        look = Cb - o[None, :]
        R_asc = xp.linalg.norm(look, axis=1)
        d_in = look / R_asc[:, None]
        cos_inc1 = xp.sum(-d_in * Nb, axis=1)
        visible1 = cos_inc1 > 0
        if leg1_occlusion_check:
            if leg1_occlusion_culled:
                occluded1 = _leg1_occlusion_chunked_culled(xp, o, d_in, R_asc, box_min, box_max, fbid,
                                                            chunk_facets=leg1_occlusion_chunk_facets)
            else:
                occluded1 = _leg1_occlusion_chunked(xp, o, d_in, R_asc, box_min, box_max, fbid,
                                                     chunk_facets=leg1_occlusion_chunk_facets)
            visible1 = visible1 & ~occluded1
        if profile:
            tp2 = _mark(tp)
            prof['leg1_occlusion'] += tp2 - tp
            tp = tp2

        # amp_eff1_geom / dR1: real, (F,)-scale, cheap regardless of scene
        # size -- alpha=1.0 (canonical GTD flat-plate/dihedral/trihedral
        # value, Potter & Moses 1996), L_el=0 (validated for a wall-ground
        # corner, task #32, 0.971 vs 0.44 coherence) drive the actual
        # envelope, computed per facet-chunk below, not here.
        amp_eff1_geom = xp.where(visible1, Ab * xp.abs(cos_inc1), 0.0)   # (F,) real, taper in env (below)
        dR1 = R_asc - R_ref
        counts['leg1'] += int(to_numpy(xp, visible1).sum())
        if profile:
            tp2 = _mark(tp)
            prof['other'] += tp2 - tp
            tp = tp2

        hit = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, half_extent_g, box_min, box_max, fbid, culling=culling)
        G, valid_geom2, d_out = hit['G'], hit['valid'], hit['d_out']
        surf_normal, is_ground, building_hit = hit['surf_normal'], hit['is_ground'], hit['building_hit']
        if profile:
            tp2 = _mark(tp)
            prof['reflect_intersect'] += tp2 - tp
            tp = tp2

        cos_ground = xp.abs(xp.sum(d_out * surf_normal, axis=1))
        valid2 = visible1 & valid_geom2
        if not leg2_building_enabled:
            valid2 = valid2 & is_ground

        theta2 = xp.arccos(xp.clip(cos_ground, 0.0, 1.0))
        R_eff_ground = xp.asarray(effective_specular_reflectivity(ground_material, to_numpy(xp, theta2), wavelength))

        bh_clamped = xp.clip(building_hit, 0, n_buildings - 1)
        R_eff_building = amp_by_building[bh_clamped]
        R_eff2 = xp.where(is_ground, R_eff_ground, R_eff_building)

        leg2_len = xp.linalg.norm(G - Cb, axis=1)
        return_vec = o[None, :] - G
        return_len = xp.linalg.norm(return_vec, axis=1)
        if leg2_occlusion_check:
            return_len_safe = xp.where(return_len > 1e-9, return_len, 1.0)
            return_dir = return_vec / return_len_safe[:, None]
            occluded_return = _segment_occluded_by_any_box(xp, G, return_dir, return_len, box_min, box_max,
                                                             exclude_ids_list=[building_hit])
            valid2 = valid2 & ~occluded_return
        if profile:
            tp2 = _mark(tp)
            prof['leg2_occlusion'] += tp2 - tp
            tp = tp2
        sin_retro = None
        if leg2_retroreflection_check or leg2_retro_taper:
            # Shared misalignment computation: the TRUE second-bounce
            # specular direction (mirror-reflecting d_out off the target
            # surface's own normal -- NOT assumed to retroreflect) vs. the
            # actual direction from G to the sensor. True automatically
            # for a 90-degree wall-ground corner AT BROADSIDE, degrades
            # away from it (see module docstring); NOT guaranteed at all
            # for an arbitrary building pair.
            retro_dir = d_out - 2.0 * xp.sum(d_out * surf_normal, axis=1, keepdims=True) * surf_normal
            return_len_safe2 = xp.where(return_len > 1e-9, return_len, 1.0)
            to_sensor_dir = return_vec / return_len_safe2[:, None]
            retro_cos = xp.clip(xp.sum(retro_dir * to_sensor_dir, axis=1), -1.0, 1.0)
            if leg2_retroreflection_check:
                retro_angle = xp.arccos(retro_cos)
                beamwidth = wavelength / xp.maximum(L_wall, wavelength)
                valid2 = valid2 & (retro_angle < (retro_beamwidth_mult * beamwidth))
            if leg2_retro_taper:
                sin_retro = xp.sqrt(xp.clip(1.0 - retro_cos * retro_cos, 0.0, 1.0))   # (F,)
        if profile:
            tp2 = _mark(tp)
            prof['retro_check'] += tp2 - tp
            tp = tp2
        L_total = R_asc + leg2_len + return_len
        amp_eff2_geom = xp.where(valid2, Ab * xp.abs(cos_inc1) * R_eff2 * cos_ground, 0.0)   # (F,) real
        R_equiv2 = L_total / 2.0
        dR2 = R_equiv2 - R_ref
        counts['leg2'] += int(to_numpy(xp, valid2).sum())

        if split_leg2_by_target:
            # Diagnostic only (task #44 investigation): does leg2 behave
            # differently for wall->GROUND targets (a true 90-degree
            # dihedral corner -- retroreflective by geometric identity, so
            # "assume a coherent direct return" is exact) vs wall->BUILDING
            # targets (an arbitrary building pair generally does NOT meet
            # at 90 degrees, so there's no guarantee the second surface's
            # own reflection law actually sends the ray back toward the
            # sensor -- this function never checks that, it just scores
            # every nearest-box pair as if it were a valid corner). If
            # ground-only coherence tracks leg1's ~0.93 while building-only
            # coherence is near zero, that confirms this is the mechanism
            # behind leg2's collapse, not occlusion (already ruled out).
            amp_eff2_ground = xp.where(is_ground, amp_eff2_geom, 0.0)
            amp_eff2_building = xp.where(is_ground, 0.0, amp_eff2_geom)
            counts['leg2_ground'] = counts.get('leg2_ground', 0) + int(to_numpy(xp, valid2 & is_ground).sum())
            counts['leg2_building'] = counts.get('leg2_building', 0) + int(to_numpy(xp, valid2 & ~is_ground).sum())

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
            # Reuses leg1's envelope same as leg2 did (preserving this
            # function's pre-existing convention of applying leg1's
            # taper/alpha to leg3 too) -- NOT necessarily correct physics,
            # see this same comment's original wording preserved in git
            # history / this session's notes: a chained facet-to-facet
            # bounce should arguably have no azimuth persistence at all.
            # Preserved as-is (include_order3 is off by default, already
            # documented as "slow, only weakly validated") -- a real fix
            # belongs with whoever next validates order3 specifically.
            amp_eff3_geom = xp.where(valid3, amp_eff2_geom * R_eff3 * cos_leg3, 0.0)   # (F,) real
            R_equiv3 = L_total3 / 2.0
            dR3 = R_equiv3 - R_ref
            counts['leg3'] += int(to_numpy(xp, valid3).sum())

        # ---- Chunked combine (task #41): env1 and phase1/2/3 are all
        # (F,K)-scale -- at F=146,053/K=4693 (a real scene this session
        # hit) that's ~11GB PER ARRAY, and this function used to build
        # FOUR of them (env1 plus one phase per leg) whole, unconditionally.
        # That OOM'd a 6GB GPU outright, not from fragmentation -- a single
        # one of those arrays alone exceeds total VRAM. Loop over facet
        # chunks instead: compute this chunk's alpha-scaled envelope ONCE
        # (asc_visible_envelope, mask_invisible=False -- amp_eff*_geom
        # above already zero every row that would have masked, task #39)
        # and reuse it for all 3 legs' phase x amplitude x envelope
        # products within the same chunk, accumulating directly into the
        # pulse's (K,) leg totals. No per-leg (F,K) array is ever
        # materialized whole, matching ray_facet_intersect's own chunking
        # convention (dense_sbr_demo.py) and _score_paths'/order-1's
        # (multibounce_demo.py, same task). ----
        leg1_contrib = xp.zeros(K, dtype=xp.complex128)
        leg2_contrib = xp.zeros(K, dtype=xp.complex128)
        leg3_contrib = xp.zeros(K, dtype=xp.complex128) if include_order3 else None
        leg2_ground_contrib = xp.zeros(K, dtype=xp.complex128) if split_leg2_by_target else None
        leg2_building_contrib = xp.zeros(K, dtype=xp.complex128) if split_leg2_by_target else None
        bytes_per_facet = max(1, K * 16)
        chunk_size = max(1, int(200_000_000 // bytes_per_facet))
        if profile:
            tpc = _mark(tp)   # close out setup-before-chunk-loop into 'other'
            prof['other'] += tpc - tp
        for cs in range(0, F, chunk_size):
            ce = min(cs + chunk_size, F)
            env_chunk = asc_visible_envelope(
                xp, o, Cb[cs:ce], freqs, visible1[cs:ce], alpha=1.0, L_az=L_wall[cs:ce], u_hat=Ub[cs:ce],
                L_el=0.0, low_precision=low_precision_envelope, mask_invisible=False)   # (chunk, K)
            if profile:
                tpc2 = _mark(tpc)
                prof['envelope'] += tpc2 - tpc
                tpc = tpc2

            phase1_c = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR1[cs:ce]) / C)
            leg1_contrib += (amp_eff1_geom[None, cs:ce] * phase1_c * env_chunk.T).sum(axis=1)
            if profile:
                tpc2 = _mark(tpc)
                prof['phase1'] += tpc2 - tpc
                tpc = tpc2

            phase2_c = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR2[cs:ce]) / C)
            env2_chunk = env_chunk
            if sin_retro is not None:
                # Dihedral-corner-specific persistence taper (see docstring):
                # sinc(2*pi*f*L_wall*sin(misalignment)/c), the SAME sinc
                # form asc_amplitude_envelope already uses for a facet's own
                # illumination taper, but built from the second-bounce
                # RETURN-PATH misalignment angle instead -- suppresses
                # exactly the geometrically-findable-but-not-actually-
                # retroreflective pairs a hard cutoff either kept in full or
                # discarded outright with no middle ground.
                arg_retro = 2.0 * xp.pi * xp.outer(freqs, L_wall[cs:ce] * sin_retro[cs:ce]) / C
                near_zero_r = xp.abs(arg_retro) < 1e-9
                arg_retro_safe = xp.where(near_zero_r, 1.0, arg_retro)
                retro_taper_chunk = xp.where(near_zero_r, 1.0, xp.sin(arg_retro_safe) / arg_retro_safe)
                env2_chunk = env_chunk * retro_taper_chunk.T
            leg2_contrib += (amp_eff2_geom[None, cs:ce] * phase2_c * env2_chunk.T).sum(axis=1)

            if split_leg2_by_target:
                leg2_ground_contrib += (amp_eff2_ground[None, cs:ce] * phase2_c * env2_chunk.T).sum(axis=1)
                leg2_building_contrib += (amp_eff2_building[None, cs:ce] * phase2_c * env2_chunk.T).sum(axis=1)

            if include_order3:
                phase3_c = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR3[cs:ce]) / C)
                leg3_contrib += (amp_eff3_geom[None, cs:ce] * phase3_c * env_chunk.T).sum(axis=1)
            if profile:
                tpc2 = _mark(tpc)
                prof['phase2'] += tpc2 - tpc
                tpc = tpc2
        if profile:
            tp = _mark(tpc)   # resync outer-loop clock with the chunk loop's

        s[p, :] += leg1_contrib
        if return_components:
            s_by_leg['leg1'][p, :] = leg1_contrib
        s[p, :] += leg2_contrib
        if return_components:
            s_by_leg['leg2'][p, :] = leg2_contrib
            if split_leg2_by_target:
                s_by_leg['leg2_ground'][p, :] = leg2_ground_contrib
                s_by_leg['leg2_building'][p, :] = leg2_building_contrib
        if include_order3:
            s[p, :] += leg3_contrib
            if return_components:
                s_by_leg['leg3'][p, :] = leg3_contrib

        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t_total += (time.perf_counter() - t0)
        if profile:
            prof['other'] += time.perf_counter() - tp   # s[p,:]+= assigns, this sync, free_pool_every, postfix

        if on_gpu and free_pool_every and (p + 1) % free_pool_every == 0:
            xp.get_default_memory_pool().free_all_blocks()

        if progress:
            postfix = dict(counts)
            if on_gpu:
                # Direct evidence for/against the pool-fragmentation
                # hypothesis above: if used_mb keeps climbing pulse over
                # pulse (even right after a free_pool_every free), that's
                # the allocator, not the math, getting slower. If it's
                # flat, look elsewhere.
                pool = xp.get_default_memory_pool()
                postfix['pool_mb'] = f"{pool.used_bytes() / 1e6:.0f}/{pool.total_bytes() / 1e6:.0f}"
            pbar.set_postfix(postfix, refresh=False)

    stats = dict(n_facets=Cb.shape[0], t_total_s=t_total, t_per_pulse_ms=t_total / n_pulses * 1000.0,
                 counts=counts)
    if return_components:
        stats['s_by_leg'] = s_by_leg
    if profile:
        stats['profile'] = prof
        prof_total = sum(prof.values())
        print(f"\n--- per-pulse profile (n_pulses={n_pulses}, totals across the whole run) ---")
        print(f"{'stage':<20s}{'total_s':>10s}{'ms/pulse':>12s}{'%':>8s}")
        for name, t in sorted(prof.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * t / prof_total if prof_total > 0 else 0.0
            print(f"{name:<20s}{t:>10.2f}{1000.0 * t / n_pulses:>12.2f}{pct:>7.1f}%")
        print(f"{'sum of stages':<20s}{prof_total:>10.2f}{1000.0 * prof_total / n_pulses:>12.2f}"
              f"{'':>8s}  (vs t_total_s={t_total:.2f} -- gap is sync/profiling overhead itself)")
        print("stage meanings: leg1_occlusion=leg1's box occlusion search; reflect_intersect=leg2's "
              "nearest-box reflect/intersect search (this is where leg2_culled_search's cull lives); "
              "leg2_occlusion/retro_check=leg2's optional return-path occlusion + retroreflection checks "
              "(near-zero if those flags are off); envelope=asc_visible_envelope's sinc/taper math; "
              "phase1/phase2=the complex exp+multiply+reduce combine per leg (phase2 also includes any "
              "retro-taper and split_leg2_by_target/order3 work inside that same chunk iteration); "
              "other=everything not individually timed (setup, bookkeeping, the final sync).")
    return s, stats
