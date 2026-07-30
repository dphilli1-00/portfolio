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


def _reflect_and_intersect_ground_plane(xp, C_bounce, N_bounce, d_in, ground_half_extent):
    """Closed-form: reflect d_in off (C_bounce, N_bounce)'s normal, then
    intersect the resulting ray with the ground PLANE z=0 analytically --
    no ray tracing, one division. Returns (G, valid, d_out).

    This is deliberately THE SAME formula run_asc_cached_multibounce uses
    per facet, factored out so run_multibounce_sbr's order-2/3 scoring
    can call it too, on the SAME (C_bounce, N_bounce) = a building
    facet's CENTER and NORMAL (not the literal ray-traced hit point).

    Why this matters, a real bug found while validating this: an earlier
    version of the dense-SBR scoring used the ACTUAL ray-traced ground
    hit point (accurate for that specific traced ray, which started from
    the literal hit_pt1 on the wall, not the wall facet's center) while
    ASC computes G by reflecting from the wall facet's CENTER (no ray
    tracing at all). Those are two different, only approximately-equal
    points -- a few meters of difference at the wall propagates into a
    much larger difference at the ground for near-grazing reflection
    angles, and a few meters is already many wavelengths at X-band, fully
    decorrelating phase between the two branches even for paths BOTH
    sides agree are valid. Using this identical closed-form formula on
    both sides (ray tracing only decides whether a path is geometrically
    valid/unobstructed, never where it analytically scores) is what
    single-bounce scoring already does correctly -- this makes the
    ground bounce follow the same rule instead of quietly breaking it."""
    d_out = d_in - 2.0 * xp.sum(d_in * N_bounce, axis=1, keepdims=True) * N_bounce
    heading_down = d_out[:, 2] < -1e-9
    d_out_z_safe = xp.where(heading_down, d_out[:, 2], -1.0)
    t_ground = xp.where(heading_down, -C_bounce[:, 2] / d_out_z_safe, 0.0)
    G = C_bounce + t_ground[:, None] * d_out
    in_bounds = (xp.abs(G[:, 0]) <= ground_half_extent) & (xp.abs(G[:, 1]) <= ground_half_extent)
    valid = heading_down & (t_ground > 1e-9) & in_bounds
    return G, valid, d_out


def _azimuth_sinc_taper(xp, wavelength, L, normal_xy, look_xy, min_horiz=1e-6):
    """
    Azimuthal beamwidth taper for a finite-length flat scatterer -- a
    wall's own horizontal extent (its u_hat length), which is what
    bounds a wall-ground dihedral's overall azimuthal persistence too
    (the ground patch is effectively unbounded by comparison, so the
    wall is the limiting aperture for the whole corner).

    This is literally the sinc(...) term from the canonical ASC/GTD
    model (Potter & Moses; Gerry/Potter/Moses/Chi; eq (1) in Wang et
    al., "3-D Attributed Scattering Center Model of Dynamic Group
    Targets by Forward Parametric Modeling Method," IEEE TGRS 2024,
    the paper this session read directly): a uniformly-illuminated
    aperture of length L produces a diffraction-limited beam in the
    direction along its own length, first null at sin(delta) =
    wavelength/L, NOT a hard geometric include/exclude cutoff.

    Neither branch in this file modeled this before now -- both the
    ray-traced side (pure ray-optics hit/miss against a facet's
    rectangular bounds) and the closed-form ASC side (backface cull +
    geometric reach test) used a binary pass/fail with no smooth
    angular rolloff. That binary-ness is the actual mechanism behind
    BOTH previously-measured failure directions, not two unrelated
    bugs: on the sparse 1000m scene, ridges far off their own broadside
    direction still counted at full strength as long as they
    geometrically reached the ground and back, inflating the ASC-side
    overcount; on the dense 300m scene, a single facet-center sample
    either passed or failed the reach test outright, with no graceful
    degradation near the boundary, producing the undercount. A smooth
    taper replaces both hard cutoffs with the same continuous physical
    quantity, applied identically on both sides.

    normal_xy, look_xy: (F,2) horizontal (x,y) components of the
    facet's outward normal and the illumination direction (facet ->
    source), not required to be unit length on input -- both are
    renormalized here. Facets whose normal has ~zero horizontal
    component (a flat roof or the flat ground, both normal=(0,0,1))
    have no well-defined azimuthal ridge direction at all -- taper=1
    (no rolloff applied) for those, since their persistence is an
    elevation/depression-angle effect this term doesn't model, not a
    horizontal-aperture one.
    """
    n_horiz_mag = xp.linalg.norm(normal_xy, axis=1)
    l_horiz_mag = xp.linalg.norm(look_xy, axis=1)
    safe = (n_horiz_mag > min_horiz) & (l_horiz_mag > min_horiz)

    n_mag_safe = xp.where(safe, n_horiz_mag, 1.0)
    l_mag_safe = xp.where(safe, l_horiz_mag, 1.0)
    n_n = normal_xy / n_mag_safe[:, None]
    l_n = look_xy / l_mag_safe[:, None]

    az_n = xp.arctan2(n_n[:, 1], n_n[:, 0])
    az_l = xp.arctan2(l_n[:, 1], l_n[:, 0])
    sin_delta = xp.sin(az_l - az_n)

    arg = 2.0 * xp.pi * L * sin_delta / wavelength
    sinc = xp.where(xp.abs(arg) < 1e-9, 1.0, xp.sin(arg) / arg)
    return xp.where(safe, sinc, 1.0)


def _range_distributed_taper(xp, wavelength, o, C1, N1, half_v1, d_in, ground_half_extent):
    """
    Closed-form amplitude taper representing a wall-ground dihedral's
    ground-bounce contribution as a CONTINUOUS distribution of
    sub-scatterers across the wall's own height (z in [0,H], H=2*half_v1)
    -- each height reflects to a different ground point with a
    different total round-trip path length -- rather than the single-
    point approximation (_reflect_and_intersect_ground_plane evaluated
    once, at the wall's vertical center) the rest of this codebase uses.

    Why this exists: measured directly (see _diagnose_candidate_gap3.py,
    this session), the ray-traced dense-SBR side's ~3x-inflated ground-
    bounce path count relative to ASC's one-point-per-wall count was NOT
    an occlusion or validity bug -- it's ray tracing naturally
    discovering that rays hitting different HEIGHTS on the same wall
    reflect to different ground points (different patches, in the tiled
    ground grid), each with its own phase. ASC's closed-form path
    collapses all of that to a single point and needs an explicit
    correction to represent the same physics; the ray-traced side
    already captures it for free through its multiple independently-
    scored ground hits. THIS TAPER IS THEREFORE APPLIED ONLY ON THE ASC
    SIDE (run_asc_cached_multibounce), never mirrored into _score_paths
    -- doing so there would double-count a spread the ray tracer already
    represents literally, not approximately.

    Derivation: integrating a uniform density of point scatterers along
    the wall's height z in [0,H], each contributing exp(-j*k*L_total(z))
    (k=2*pi/wavelength, L_total = the full round-trip path length
    through that height's own ground-reflection point), and treating
    L_total(z) as locally linear in z (valid since a building's height
    is a few tens of meters against a multi-km range) integrates in
    closed form to the center height's own phase (already what
    _reflect_and_intersect_ground_plane computes) times a sinc factor.
    This is the same derivation the ASC/GTD model itself uses for its
    length-dependent parameters -- eq (3) in Wang et al., "3-D
    Attributed Scattering Center Model of Dynamic Group Targets by
    Forward Parametric Modeling Method," IEEE TGRS 2024 (the paper this
    session read directly), the cylindrical-surface distributed-
    scattering-center example, where the "L_k" length parameter "in the
    RLOS direction" exists specifically to capture unequal phase across
    a spread set of scattering sources -- this is that same L_k idea,
    applied here to a wall-ground dihedral's ground footprint instead of
    a curved surface's specular strip.

    Ray tracing (not this function) decides path VALIDITY elsewhere in
    this codebase; this only ever adjusts AMPLITUDE for a path already
    agreed valid, same convention as _azimuth_sinc_taper.
    """
    C1_bottom = C1.copy(); C1_bottom[:, 2] = C1[:, 2] - half_v1
    C1_top = C1.copy(); C1_top[:, 2] = C1[:, 2] + half_v1

    G_bot, _, _ = _reflect_and_intersect_ground_plane(xp, C1_bottom, N1, d_in, ground_half_extent)
    G_top, _, _ = _reflect_and_intersect_ground_plane(xp, C1_top, N1, d_in, ground_half_extent)

    o_b = o[None, :]
    L_bot = xp.linalg.norm(o_b - C1_bottom, axis=1) + xp.linalg.norm(C1_bottom - G_bot, axis=1) + xp.linalg.norm(o_b - G_bot, axis=1)
    L_top = xp.linalg.norm(o_b - C1_top, axis=1) + xp.linalg.norm(C1_top - G_top, axis=1) + xp.linalg.norm(o_b - G_top, axis=1)
    dL = L_top - L_bot

    k = 2.0 * xp.pi / wavelength
    arg = k * dL / 2.0
    return xp.where(xp.abs(arg) < 1e-9, 1.0, xp.sin(arg) / arg)


def _range_distributed_taper_kf(xp, freqs, o, C1, N1, half_v1, d_in, ground_half_extent):
    """
    Corrected, frequency-RESOLVED version of _range_distributed_taper.
    That function collapsed the wall-height-spread ground-bounce effect
    to one scalar per facet, evaluated at a single mean wavelength, and
    measured WORSE than no taper at all (see its docstring for the
    diagnosis). This version keeps frequency explicit throughout the
    derivation instead of substituting a mean value partway through,
    and returns a (K,F) array -- one taper value PER FREQUENCY BIN PER
    FACET -- meant to multiply elementwise into the (K,F) phase array
    before summing over facets, NOT into a per-facet amplitude scalar
    the way every other amplitude term in this file works.

    Why frequency has to stay explicit: the physical effect being
    modeled (different heights along the wall reflecting to different
    ground points, hence different path lengths) is exactly what gives
    a wideband signal range resolution once backprojected -- that's
    what backprojection is FOR. Pre-collapsing to one mean-wavelength
    number throws away the frequency diversity before backprojection
    ever gets to use it, forcing a single wrong suppression value onto
    every frequency instead of letting each frequency's own phase
    naturally spread a tall wall's return across the range pixels that
    actually correspond to its different heights -- the same way ray
    tracing's multiple, independently-scored ground-patch hits do.

    Returns taper(f) = sinc(pi*f*deltaL/c) per frequency bin, where
    deltaL is the SAME top-vs-bottom round-trip path-length difference
    _range_distributed_taper computes -- only the mean-wavelength
    substitution was wrong, not the underlying geometry.
    """
    C1_bottom = C1.copy(); C1_bottom[:, 2] = C1[:, 2] - half_v1
    C1_top = C1.copy(); C1_top[:, 2] = C1[:, 2] + half_v1

    G_bot, _, _ = _reflect_and_intersect_ground_plane(xp, C1_bottom, N1, d_in, ground_half_extent)
    G_top, _, _ = _reflect_and_intersect_ground_plane(xp, C1_top, N1, d_in, ground_half_extent)

    o_b = o[None, :]
    L_bot = xp.linalg.norm(o_b - C1_bottom, axis=1) + xp.linalg.norm(C1_bottom - G_bot, axis=1) + xp.linalg.norm(o_b - G_bot, axis=1)
    L_top = xp.linalg.norm(o_b - C1_top, axis=1) + xp.linalg.norm(C1_top - G_top, axis=1) + xp.linalg.norm(o_b - G_top, axis=1)
    dL = L_top - L_bot   # (F,) -- unchanged from the scalar version

    arg = xp.pi * xp.outer(freqs, dL) / C   # (K,F): pi*f*deltaL/c, per frequency bin
    return xp.where(xp.abs(arg) < 1e-9, 1.0, xp.sin(arg) / arg)


def _building_aabbs(xp, facets_buildings):
    """Precompute one axis-aligned bounding box per building (not per
    facet) from the host-side building_cx/cy/w/d/h metadata
    make_building_scene already carries -- box_min/box_max: (n_buildings,3),
    plus building_id_of_facet: (n_facets,) mapping each building facet
    index to its parent building (make_building_scene always appends
    exactly n_facets/n_buildings facets per building, in order, so this
    is a simple integer division, not a search)."""
    cx = facets_buildings['building_cx']; cy = facets_buildings['building_cy']
    w = facets_buildings['building_w']; d = facets_buildings['building_d']; h = facets_buildings['building_h']
    n_buildings = facets_buildings['n_buildings']
    facets_per_building = facets_buildings['n_facets'] // n_buildings
    box_min = xp.asarray(np.column_stack([cx - w / 2.0, cy - d / 2.0, np.zeros_like(cx)]))
    box_max = xp.asarray(np.column_stack([cx + w / 2.0, cy + d / 2.0, h]))
    building_id_of_facet = xp.asarray(np.repeat(np.arange(n_buildings), facets_per_building))
    return box_min, box_max, building_id_of_facet


def _segments_blocked_by_buildings(xp, P0, P1, box_min, box_max, exclude_building_id=None):
    """Cheap occlusion approximation: does the straight segment P0->P1
    (one segment per candidate path, (F,3) each) pass through any OTHER
    building's axis-aligned bounding box? Standard slab ray/segment-AABB
    test, vectorized over all (segment, building) pairs at once --
    O(F x B), not O(rays x facets): no ray tracing, no per-pulse aim
    grid, just F*B box tests. For F~a few hundred candidate paths and
    B~a couple hundred buildings that's still comfortably under a
    million cheap float ops, versus dense-SBR's O(rays x facets)
    (millions of ray-facet intersections) per pulse.

    exclude_building_id: (F,) building index each segment's own facet
    belongs to -- excluded from its own occlusion test (a wall's own
    building trivially "contains" the point the segment starts from,
    which isn't a real occlusion).

    This is an AABB test, not exact building geometry -- a segment that
    grazes past a building's box corner without actually touching the
    real (smaller) building volume can be flagged as blocked when it
    isn't. That's a conservative bias (more likely to under-count than
    over-count valid paths), the opposite failure mode from having no
    occlusion check at all, and a deliberate trade for staying cheap."""
    d = P1 - P0                                   # (F,3)
    d_safe = xp.where(xp.abs(d) > 1e-12, d, 1e-12)
    t1 = (box_min[None, :, :] - P0[:, None, :]) / d_safe[:, None, :]   # (F,B,3)
    t2 = (box_max[None, :, :] - P0[:, None, :]) / d_safe[:, None, :]
    tmin = xp.max(xp.minimum(t1, t2), axis=2)      # (F,B)
    tmax = xp.min(xp.maximum(t1, t2), axis=2)
    tmin_seg = xp.maximum(tmin, 0.0)
    tmax_seg = xp.minimum(tmax, 1.0)
    hit = tmin_seg <= tmax_seg                     # (F,B)

    if exclude_building_id is not None:
        B = box_min.shape[0]
        F = P0.shape[0]
        self_mask = xp.arange(B)[None, :] == exclude_building_id[:, None]
        hit = hit & (~self_mask)

    return xp.any(hit, axis=1)                     # (F,) -- blocked by at least one other building


def _segment_occluded_by_facets(xp, P0, P1, facets, eps=1e-3):
    """EXACT occlusion test (not AABB): does the straight segment P0->P1
    cross any real facet in `facets`? Reuses ray_facet_intersect's own
    per-facet math -- rectangular in-bounds test, front-face-only, the
    identical intersection code the ray-traced dense-SBR tracer runs --
    just capped to the segment's own length instead of an infinite ray
    (same pattern as _return_visible, generalized to an arbitrary
    segment instead of always "back to the platform").

    Why this exists after _segments_blocked_by_buildings: AABB testing
    against a building's bounding BOX turned out not to be the limiting
    approximation here -- these are already box-shaped buildings, so an
    AABB is close to exact for them geometrically, and the measured
    fidelity gain was real but modest (SSIM 0.42->0.45, overcount ratio
    1.93x->1.39x, ~931 phantom paths still left on the real scene).
    Testing the exact facets instead of a box removes the remaining
    approximation error at the geometry level. Still O(candidates x
    facets), not O(rays x facets): the candidate set here is the (small,
    already analytically-pre-filtered) set of paths the cheap geometric
    check flagged as plausible, not a dense exploratory ray grid -- what
    makes this cheaper than full ray tracing is having 3 KNOWN analytic
    segments to test per candidate instead of needing many rays to
    statistically discover them."""
    d = P1 - P0
    dist = xp.linalg.norm(d, axis=1)
    dist_safe = xp.where(dist > 0, dist, 1.0)
    d_n = d / dist_safe[:, None]
    o_off = P0 + eps * d_n
    hit_pt, hit_amp, cos_inc, hit_mask, nearest_idx = ray_facet_intersect(xp, o_off, d_n, facets)
    t_hit = xp.where(hit_mask, xp.linalg.norm(hit_pt - o_off, axis=1), xp.inf)
    return hit_mask & (t_hit < (dist - 2.0 * eps))


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


def _decode_unique_paths_with_rep(xp, keys_all, valid_mask, n_levels, F_c):
    """Same decoding as _decode_unique_paths, but also returns, for each
    unique valid path, the index of ONE representative ray (into the
    original 0..R-1 ray arrays) that produced it.

    Needed because facet-CENTER collapsing (the convention every other
    bounce in this codebase uses, and a fine approximation for building
    facets a few meters across) is too coarse for the ground: even
    tiled into patch_m-sized patches (make_ground_facet), a patch center
    can sit many meters from where a specific ray actually crossed z=0,
    and the closed-form ASC path (run_asc_cached_multibounce) computes
    its ground intersection exactly, with no patch grid at all. At X-band
    wavelengths a few meters of position error is already enough range
    error to fully decorrelate phase -- this is what actually caused the
    near-zero SSIM/coherence the first version of this fix still had (see
    make_ground_facet's docstring for the full story). Using one
    representative ray's OWN traced hit point (continuous, not gridded)
    for ground-level positions in _score_paths, instead of any center
    lookup, is what actually fixes it."""
    valid_ray_idx = xp.nonzero(valid_mask)[0]
    keys_valid = keys_all[valid_mask]
    if keys_valid.shape[0] == 0:
        return _decode_unique_paths(xp, keys_valid, n_levels, F_c), valid_ray_idx[:0]
    uniq, first_pos = xp.unique(keys_valid, return_index=True)
    rep_ray_idx = valid_ray_idx[first_pos]
    levels = []
    rem = uniq
    for _ in range(n_levels):
        levels.append(rem % F_c)
        rem = rem // F_c
    levels.reverse()
    return levels, rep_ray_idx


def _score_paths(xp, o, ref_pos, freqs, level_idxs, level_centers, level_normals, level_amps,
                  ground_idx=None, ground_material=None, wavelength=None, ground_pos_override=None,
                  level_uhat=None, level_halfu=None):
    """Given deduped path index arrays (one (U,) array per bounce level),
    evaluate the analytic per-path contribution and sum coherently.
    Returns (K,) contribution and the path count U.

    Position per bounce is normally that level's FACET CENTER -- a fine
    approximation for building facets a few meters across (both the
    dense-SBR and ASC sides agree on using it, which is why single-bounce
    and the building leg of a double-bounce path stay consistent). NOT a
    fine approximation for the ground: even tiled into patches
    (make_ground_facet), a patch center can be many meters from where a
    specific path actually crossed z=0, and ASC's closed-form ground
    intersection is continuous, not gridded -- a few meters of position
    mismatch is already enough range error to fully decorrelate X-band
    phase. ground_pos_override lets the caller supply, per level, a
    (U,3) array of ACTUAL traced hit points to use instead of the center
    lookup wherever that level's bounce landed on the ground (see
    _decode_unique_paths_with_rep) -- this is what makes ground-bounce
    scoring match the closed-form ASC side's precision instead of a
    patch grid's.

    Reflectivity per bounce is normally the facet's own static amp value
    (a building wall's pre-drawn reflectivity, unchanged from single-
    bounce). For any bounce landing on the ground (level_idxs[i] >=
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

    pts = []
    for i in range(len(level_idxs)):
        p_i = level_centers[i][level_idxs[i]]
        if ground_pos_override is not None and ground_pos_override[i] is not None and ground_idx is not None:
            is_ground_i = level_idxs[i] >= ground_idx
            p_i = xp.where(is_ground_i[:, None], ground_pos_override[i], p_i)
        pts.append(p_i)
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
            # ground now occupies a RANGE of indices (one per ground patch,
            # see make_ground_facet) appended after all building indices,
            # not a single trailing index -- >= ground_idx (the first
            # ground index), not == , catches every patch
            is_ground = level_idxs[i] >= ground_idx
            if bool(to_numpy(is_ground).any()):
                theta_i = xp.arccos(xp.clip(cos_i, 0.0, 1.0))
                theta_i_np = to_numpy(theta_i)
                ground_refl_np = effective_specular_reflectivity(ground_material, theta_i_np, wavelength)
                ground_refl = xp.asarray(ground_refl_np)
                refl_i = xp.where(is_ground, ground_refl, refl_i)

        if level_halfu is not None and wavelength is not None:
            # Azimuthal sinc-beamwidth taper (see _azimuth_sinc_taper) --
            # applies to whichever facet this level's bounce actually
            # landed on (wall or ground); ground/roof facets have a
            # vertical normal and get taper=1 automatically (the helper's
            # near-zero-horizontal-normal guard), so this is safe to call
            # unconditionally rather than branching on is_ground.
            # (level_uhat is accepted for API symmetry/future use but not
            # needed here -- the taper's broadside direction comes from
            # the facet's own NORMAL azimuth, not u_hat, and only the
            # facet's length along u_hat, half_u, is actually used.)
            L_i = 2.0 * level_halfu[i][level_idxs[i]]   # ridge/facet length along its own u_hat axis
            illum_dir = -incoming_n   # direction from this bounce back toward where it came from
            taper_i = _azimuth_sinc_taper(xp, wavelength, L_i, norms[i][:, :2], illum_dir[:, :2])
            refl_i = refl_i * taper_i

        amp_eff = amp_eff * refl_i * cos_i

    R_ref = xp.linalg.norm(o - ref_pos)
    R_equiv = L_total / 2.0      # see module docstring: matches the existing 4*pi/round-trip=2R convention
    dR = R_equiv - R_ref
    phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)   # (K,U)
    contrib = (amp_eff[None, :] * phase).sum(axis=1)
    return contrib, U


def run_multibounce_sbr(xp, on_gpu, facets_buildings, facets_ground, plat, aim_pts, freqs, ref_pos,
                         max_bounces=3, eps=1e-3, return_components=False):
    """
    return_components: when True, stats['s_by_order'] additionally holds
    the PER-ORDER phase histories (order1/order2/order3, each (n_pulses,K))
    separately, not just their sum in `s`. Needed for isolating a specific
    bounce order's own fidelity -- comparing combined images would dilute
    a double-bounce disagreement under the much larger, already-validated
    single-bounce agreement, the same "whole-image SSIM is diluted"
    lesson the Tier 2 validation slide already learned for per-building
    scoring vs. whole-image scoring.
    """
    n_pulses = plat.shape[0]
    K = freqs.shape[0]

    facets_combined = concat_facets(xp, facets_buildings, facets_ground)
    F_b = facets_buildings['center'].shape[0]
    F_c = facets_combined['center'].shape[0]
    ground_idx = F_b   # first ground-patch index -- all indices >= this are ground (see make_ground_facet)
    ground_material = facets_ground.get('material', 'dry_soil')
    ground_half_extent = float(facets_ground.get('footprint_half_extent', to_numpy(facets_ground['half_u'])[0]))
    wavelength = C / float(to_numpy(freqs).mean())   # fractional bandwidth here is modest enough
    # that treating wavelength as ~constant across it (rather than per-frequency-sample) is fine
    # for the roughness/specular check -- it's a slowly-varying geometric factor, not the fine
    # range-resolution phase term itself.

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    Cc, Nc, Ac = facets_combined['center'], facets_combined['normal'], facets_combined['amp']
    Ub, HUb = facets_buildings['u_hat'], facets_buildings['half_u']
    Uc, HUc = facets_combined['u_hat'], facets_combined['half_u']

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    counts = dict(order1=0, order2=0, order3=0)
    t_total = 0.0
    s_by_order = None
    if return_components:
        s_by_order = dict(order1=xp.zeros((n_pulses, K), dtype=xp.complex128),
                           order2=xp.zeros((n_pulses, K), dtype=xp.complex128),
                           order3=xp.zeros((n_pulses, K), dtype=xp.complex128))

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
            vL = 2.0 * HUb[visible1]
            look = vpos - o[None, :]
            look = look / xp.linalg.norm(look, axis=1, keepdims=True)
            cos_v = xp.abs(xp.sum(-look * vnorm, axis=1))
            taper_v = _azimuth_sinc_taper(xp, wavelength, vL, vnorm[:, :2], -look[:, :2])
            R_v = xp.linalg.norm(vpos - o[None, :], axis=1)
            dR1 = R_v - R_ref
            amp_eff1 = vamp * cos_v * taper_v
            phase1 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR1) / C)
            order1_contrib = (amp_eff1[None, :] * phase1).sum(axis=1)
            s[p, :] += order1_contrib
            if return_components:
                s_by_order['order1'][p, :] = order1_contrib
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
        lvl2, rep2 = _decode_unique_paths_with_rep(xp, key2, valid2, 2, F_c)
        # Ground bounce position: recompute the SAME closed-form
        # reflect-off-facet-center + intersect-ground-plane formula
        # run_asc_cached_multibounce uses, from idx1's center/normal --
        # NOT the literal ray-traced hit point. Both were tried; using the
        # literal hit point (accurate for that one ray, but NOT the same
        # point ASC's idealized center-based reflection computes) still
        # left SSIM/coherence near zero, because a few meters of
        # first-bounce position difference (traced hit point vs. facet
        # center) blows up into a much larger ground-point difference at
        # near-grazing angles -- many wavelengths at X-band, fully
        # decorrelating phase even for paths both branches agree are
        # valid. Ray tracing (hit_mask2/clear2, above) still decides
        # VALIDITY; this decides the analytic SCORE, matching ASC exactly
        # for any facet both sides consider a real double-bounce path.
        idx1_u = lvl2[0]
        C1_u = Cb[idx1_u]; N1_u = Nb[idx1_u]
        d_in_u = C1_u - o[None, :]
        d_in_u = d_in_u / xp.linalg.norm(d_in_u, axis=1, keepdims=True)
        G_analytic, _valid_geom_u, _ = _reflect_and_intersect_ground_plane(xp, C1_u, N1_u, d_in_u, ground_half_extent)
        ground_pos2 = [None, G_analytic]
        contrib2, n2 = _score_paths(xp, o, ref_pos, freqs, lvl2, [Cb, Cc], [Nb, Nc], [Ab, Ac],
                                     ground_idx=ground_idx, ground_material=ground_material,
                                     wavelength=wavelength, ground_pos_override=ground_pos2,
                                     level_uhat=[Ub, Uc], level_halfu=[HUb, HUc])
        s[p, :] += contrib2
        if return_components:
            s_by_order['order2'][p, :] = contrib2
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
            lvl3, rep3 = _decode_unique_paths_with_rep(xp, key3, valid3, 3, F_c)
            # same representative ray's own hit_pt2/hit_pt3 used together
            # (not mixed with another ray's) so a wall-ground-wall or
            # wall-wall-ground path stays internally geometrically
            # consistent, not just individually-accurate per leg
            ground_pos3 = [None, hit_pt2[rep3], hit_pt3[rep3]]
            contrib3, n3 = _score_paths(xp, o, ref_pos, freqs, lvl3, [Cb, Cc, Cc], [Nb, Nc, Nc], [Ab, Ac, Ac],
                                         ground_idx=ground_idx, ground_material=ground_material,
                                         wavelength=wavelength, ground_pos_override=ground_pos3,
                                         level_uhat=[Ub, Uc, Uc], level_halfu=[HUb, HUc, HUc])
            s[p, :] += contrib3
            if return_components:
                s_by_order['order3'][p, :] = contrib3
            counts['order3'] += n3

        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t_total += (time.perf_counter() - t0)

    stats = dict(counts=counts, t_total_s=t_total,
                 t_per_pulse_ms=t_total / n_pulses * 1000.0,
                 n_facets_buildings=F_b, n_facets_combined=F_c)
    if return_components:
        stats['s_by_order'] = s_by_order
    return s, stats


def _wall_height_dL(xp, o, C1, N1, half_v1, d_in, ground_half_extent):
    """
    Round-trip path-length spread between a wall's top and bottom
    reflecting to the ground, in closed form -- the same DeltaL computed
    inside _range_distributed_taper, factored out here so it can be used
    for a REGIME DECISION (is this facet electrically large enough that
    a single point is unsafe?) rather than folded into a taper. See
    run_asc_cached_multibounce's regime_adaptive mode.
    """
    C1_bottom = C1.copy(); C1_bottom[:, 2] = C1[:, 2] - half_v1
    C1_top = C1.copy(); C1_top[:, 2] = C1[:, 2] + half_v1
    G_bot, _, _ = _reflect_and_intersect_ground_plane(xp, C1_bottom, N1, d_in, ground_half_extent)
    G_top, _, _ = _reflect_and_intersect_ground_plane(xp, C1_top, N1, d_in, ground_half_extent)
    o_b = o[None, :]
    L_bot = xp.linalg.norm(o_b - C1_bottom, axis=1) + xp.linalg.norm(C1_bottom - G_bot, axis=1) + xp.linalg.norm(o_b - G_bot, axis=1)
    L_top = xp.linalg.norm(o_b - C1_top, axis=1) + xp.linalg.norm(C1_top - G_top, axis=1) + xp.linalg.norm(o_b - G_top, axis=1)
    return L_top - L_bot


def _ground_bounce_subpoint_contrib(xp, o, R_ref, freqs, Cb, Nb, Ab, HVb, d_in, cos_inc1, taper1,
                                     valid2, ground_half_extent, ground_material, wavelength, n_sub,
                                     z_weights=None):
    """
    Replace the single facet-CENTER ground-bounce point with n_sub
    discrete points sampled along each wall's height, each independently
    reflected via the SAME closed-form _reflect_and_intersect_ground_plane
    and scored at ITS OWN exact position/phase -- then summed coherently.

    Why this exists, after two failed taper attempts: a single point's
    amplitude can be tapered (weakened), but it can never appear at
    several genuinely different image pixels -- and that's what a tall
    wall's ground bounce actually needs to do, since DeltaL (the path-
    length spread between a wall's top and bottom reflecting to the
    ground) reaches ~200m on real scenes here, thousands of wavelengths
    at X-band. Ray tracing already gets this for free: each ray's hit
    height lands at its own true ground position. This function is the
    ASC-side equivalent, done with n_sub discrete closed-form points
    instead of one -- still no ray tracing, still O(n_sub x F) per
    pulse, just no longer O(1 x F).

    valid2 (F,) is reused UNCHANGED from the existing single-point
    validity/occlusion pipeline -- this function only replaces how an
    ALREADY-validated facet's contribution is scored, not whether it
    counts as valid at all (occlusion is still decided once per facet
    at the center, not re-checked per sub-point -- a reasonable
    approximation since sub-points along one wall pass through
    essentially the same surrounding geometry, and re-running full
    per-sub-point occlusion would push cost toward ray tracing's own
    O(rays x facets), defeating the point of staying closed-form).

    amplitude per sub-point is Ab/n_sub (not Ab) so that if all
    sub-points happened to land in phase (DeltaL=0), the coherent sum
    recovers exactly the original single-point amplitude -- consistent
    energy calibration with every other single-point term in this file.

    MEASURED RESULT (this session, _taper_ab_test.py / _taper_ab_test2.py):
    numerically convergent (stable by n_sub~16-32) and phase RMS improves
    substantially (127deg -> 76deg against the ray-traced order-2 ground
    truth) -- but SSIM and coherence both get WORSE than the simpler
    n_sub=1 + azimuth-taper baseline (SSIM 0.484->0.44, coherence
    0.178->0.06), and re-weighting sub-points by the ray tracer's own
    MEASURED hit-density-vs-height (see _diagnose_hit_density.py,
    z_weights param) only partially recovers SSIM and does not recover
    coherence at all. Likely explanation: SSIM/coherence against this
    particular ray-traced reference partly reward matching the ray
    tracer's OWN specific discretization artifacts (its aim-grid
    geometry, its ground-patch tiling), not just "more correct physics"
    in the abstract -- a genuinely better physical model doesn't
    automatically score better against a reference with its own
    particular approximations baked in. Kept as an opt-in path
    (default n_sub=1, i.e. unused) rather than the default -- it is a
    real, working, better-motivated model, just not (yet) a measured
    win on the metrics this deck's validation actually uses.
    """
    F = Cb.shape[0]
    edges = xp.linspace(-1.0, 1.0, n_sub + 1)
    z_frac = (edges[:-1] + edges[1:]) / 2.0            # (n_sub,) bin centers, avoids the z=0 exact-endpoint degenerate case
    z_offset = z_frac[None, :] * HVb[:, None]           # (F, n_sub)

    C1_sub = xp.broadcast_to(Cb[:, None, :], (F, n_sub, 3)).copy()
    C1_sub = C1_sub.reshape(F * n_sub, 3)
    C1_sub[:, 2] = C1_sub[:, 2] + z_offset.reshape(F * n_sub)

    N1_flat = xp.broadcast_to(Nb[:, None, :], (F, n_sub, 3)).reshape(F * n_sub, 3)
    d_in_flat = xp.broadcast_to(d_in[:, None, :], (F, n_sub, 3)).reshape(F * n_sub, 3)

    G_flat, valid_geom_flat, d_out_flat = _reflect_and_intersect_ground_plane(
        xp, C1_sub, N1_flat, d_in_flat, ground_half_extent)

    o_b = o[None, :]
    L_flat = (xp.linalg.norm(xp.broadcast_to(o_b, (F * n_sub, 3)) - C1_sub, axis=1)
              + xp.linalg.norm(C1_sub - G_flat, axis=1)
              + xp.linalg.norm(xp.broadcast_to(o_b, (F * n_sub, 3)) - G_flat, axis=1))

    cos_ground_flat = xp.abs(d_out_flat[:, 2])
    theta_ground_flat = xp.arccos(xp.clip(cos_ground_flat, 0.0, 1.0))
    R_eff_flat = xp.asarray(effective_specular_reflectivity(
        ground_material, to_numpy(theta_ground_flat), wavelength))

    # z_weights (n_sub,): per-bin fraction of a facet's total amplitude
    # assigned to that height bin -- defaults to uniform (1/n_sub each).
    # Passing the RAY TRACER's own measured hit-density-vs-height
    # histogram here (see _diagnose_hit_density.py) tests whether
    # matching that empirical distribution, instead of assuming flat,
    # closes the SSIM/coherence gap uniform sampling left open.
    if z_weights is None:
        z_weights = xp.full((n_sub,), 1.0 / float(n_sub))
    valid_sub = valid_geom_flat.reshape(F, n_sub) & valid2[:, None]
    amp_base = (Ab * xp.abs(cos_inc1) * taper1)[:, None]                    # (F,1)
    amp_sub = amp_base * z_weights[None, :] * R_eff_flat.reshape(F, n_sub) * cos_ground_flat.reshape(F, n_sub)
    amp_sub = xp.where(valid_sub, amp_sub, 0.0)                              # (F, n_sub)

    R_equiv_sub = L_flat.reshape(F, n_sub) / 2.0
    dR_sub = R_equiv_sub - R_ref                                            # (F, n_sub)
    phase_sub = xp.exp(-1j * 4.0 * xp.pi * freqs[:, None, None] * dR_sub[None, :, :] / C)  # (K,F,n_sub)
    contrib = (amp_sub[None, :, :] * phase_sub).sum(axis=(1, 2))            # (K,)
    return contrib


def run_asc_cached_multibounce(xp, on_gpu, facets_buildings, facets_ground, plat, freqs, ref_pos,
                                ground_material=None, include_ground_bounce=True,
                                return_components=False, occlusion_check=True,
                                occlusion_method='exact', ground_bounce_n_sub=1,
                                ground_bounce_z_weights=None, regime_adaptive=False,
                                escalate_threshold_cells=1.0, escalate_n_sub=16):
    """
    ASC-cached forward model, extended with a closed-form wall-ground
    double bounce -- no ray tracing anywhere in this function, still
    O(facets) per pulse, same complexity class as the existing single-
    bounce run_asc_cached in sbr_vs_asc_compare.py (which this leaves
    untouched).

    The idea (this is the ASC-side answer to "can it see a ground bounce
    without re-tracing"): for every building facet, the sensor->facet
    ray direction is already known analytically (facet position is read
    straight from the mesh, same as single-bounce ASC). Reflect that
    direction off the facet's own normal -- also closed-form, no search
    -- and intersect the resulting ray with the ground PLANE (z=0) in
    closed form (a flat-plane ray intersection is one division, not a
    search over facets). That gives the ground bounce point directly,
    with no ray-facet loop at all.

    Occlusion, measured then fixed: the original version of this function
    did NO occlusion/shadow-ray check on either leg at all (a facet was
    scored purely on whether ITS OWN reflection geometry plausibly
    reaches the ground and returns toward the sensor). Measured against
    the ray-traced dense-SBR ground truth
    (validation_multibounce_compare.py): of the facets this closed-form
    check called geometrically valid, only ~29% actually had an
    unobstructed path in reality -- ~71% were phantoms, blocked by some
    OTHER building somewhere along one of the three legs.

    occlusion_check=True (default) turns on occlusion filtering;
    occlusion_method picks how:

    - 'exact' (default): test each of the 3 legs against the REAL
      building facets (_segment_occluded_by_facets), reusing
      ray_facet_intersect's own per-facet math -- the identical
      intersection test the ray-traced tracer uses, just applied to 3
      analytically-known segments per candidate instead of a dense
      exploratory ray grid. Still O(candidates x facets), not O(rays x
      facets) -- cheap because there are only a few hundred candidates
      to check, not thousands of rays needed to discover them
      statistically. This is what actually closes most of the fidelity
      gap: an 'aabb' pass alone (below) only took the path-count
      overcount ratio from ~1.9x to ~1.4x and barely moved SSIM/
      coherence, because these are already box-shaped buildings -- a
      bounding box is nearly exact for them already, so approximating
      the BOX wasn't the bottleneck; something finer than "box vs box"
      was needed, which for box buildings means testing the actual
      facets, not a coarser proxy for them.
    - 'aabb': cheaper, coarser -- test each leg against every OTHER
      building's axis-aligned bounding box (_segments_blocked_by_buildings)
      instead of its real facets. Kept as an option for the speed/
      fidelity tradeoff, not because it's recommended when fidelity
      matters.

    Set occlusion_check=False to recover the original no-check behavior
    for comparison.

    No order-3 (triple bounce) term -- reflecting the wall-ground
    double-bounce ray a second time off an arbitrary OTHER building's
    wall has no closed-form target the way a fixed ground plane does
    (which building, if any, is a search problem again), so a genuinely
    ray-tracing-free triple bounce isn't available with this approach.

    regime_adaptive=True: per-facet decision, not a global one. Every
    fix tried before this (azimuth taper, range/L_k tapers, uniform and
    density-weighted multi-point decomposition -- see this file's other
    docstrings for the full, honestly-measured history) either applies
    uniformly to every facet or not at all. This mode computes DeltaL
    (_wall_height_dL, the same round-trip path-length spread the taper
    attempts used) for every facet, compares it to the system's own
    range resolution cell (c / (2*bandwidth), inferred from freqs), and
    only escalates to the more expensive multi-point treatment
    (_ground_bounce_subpoint_contrib, escalate_n_sub points) for facets
    whose DeltaL exceeds escalate_threshold_cells resolution cells --
    i.e. facets that are ACTUALLY electrically too large for a single
    point to be a safe approximation at this system's own resolution.
    Everything else stays on the cheap single-point path, unchanged.
    This doesn't try to close the fidelity gap by being more clever
    about the physics (that was tried, repeatedly, and mostly didn't
    help) -- it tries to make the gap mostly irrelevant, by predicting
    ahead of time which facets need the expensive treatment instead of
    either always paying for it everywhere or never paying for it
    anywhere.
    """
    n_pulses = plat.shape[0]
    K = freqs.shape[0]

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    L_wall = 2.0 * facets_buildings['half_u']   # each wall's own length along its ridge -- fixed
    HVb = facets_buildings['half_v']            # each wall's own half-height -- fixed
    normal_xy = Nb[:, :2]                        # geometry, computed once outside the per-pulse loop
    # facets_ground is now a GRID of small patches (see make_ground_facet) --
    # half_u/half_v on any single patch is that patch's own tiny extent, not
    # the ground's overall footprint; use footprint_half_extent for the
    # overall in-bounds check instead (falls back to the old single-facet
    # reading only if an older-style facets_ground dict is ever passed in).
    half_extent_g = float(facets_ground.get('footprint_half_extent', to_numpy(facets_ground['half_u'])[0]))
    if ground_material is None:
        ground_material = facets_ground.get('material', 'dry_soil')
    wavelength = C / float(to_numpy(freqs).mean())

    resolution_cell = None
    escalate_threshold = None
    if regime_adaptive:
        freqs_np = to_numpy(freqs)
        bandwidth = float(freqs_np.max() - freqs_np.min())
        resolution_cell = C / (2.0 * bandwidth) if bandwidth > 0 else float('inf')
        escalate_threshold = escalate_threshold_cells * resolution_cell

    box_min = box_max = building_id_of_facet = None
    if occlusion_check and occlusion_method == 'aabb':
        box_min, box_max, building_id_of_facet = _building_aabbs(xp, facets_buildings)

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    t_total = 0.0
    n_ground_bounce_total = 0
    n_ground_bounce_occl_blocked = 0
    n_escalated_total = 0
    s_by_leg = None
    if return_components:
        s_by_leg = dict(leg1=xp.zeros((n_pulses, K), dtype=xp.complex128),
                         leg2=xp.zeros((n_pulses, K), dtype=xp.complex128))

    for p in range(n_pulses):
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)

        t0 = time.perf_counter()

        # ---- leg 1: sensor -> facet, identical to run_asc_cached except
        # for the added azimuth taper (see _azimuth_sinc_taper) ----
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

        if include_ground_bounce:
            # ---- leg 2: reflect off each facet's own normal, closed-form
            # intersect with the ground plane z=0 (no search) -- same
            # helper run_multibounce_sbr's order-2 scoring calls, so a
            # facet both branches agree is valid gets the EXACT same
            # ground point, not two independently-approximated ones ----
            G, valid_geom2, d_out = _reflect_and_intersect_ground_plane(xp, Cb, Nb, d_in, half_extent_g)
            cos_ground = xp.abs(d_out[:, 2])   # = |dot(-d_out, ground_normal)|, ground_normal=(0,0,1)
            valid2 = visible1 & valid_geom2

            if occlusion_check:
                # only test candidates that already passed the free
                # geometric check above -- no reason to occlusion-test
                # facets that were never going to contribute anyway
                cand_idx = xp.nonzero(valid2)[0]
                if cand_idx.shape[0] > 0:
                    o_rep = xp.broadcast_to(o[None, :], (cand_idx.shape[0], 3))
                    if occlusion_method == 'exact':
                        # test the real building facets, not a bounding
                        # box -- see docstring for why AABB alone wasn't
                        # enough for box-shaped buildings
                        blocked1 = _segment_occluded_by_facets(xp, o_rep, Cb[cand_idx], facets_buildings)
                        blocked2 = _segment_occluded_by_facets(xp, Cb[cand_idx], G[cand_idx], facets_buildings)
                        blocked3 = _segment_occluded_by_facets(xp, G[cand_idx], o_rep, facets_buildings)
                    else:
                        excl = building_id_of_facet[cand_idx]
                        blocked1 = _segments_blocked_by_buildings(xp, o_rep, Cb[cand_idx], box_min, box_max, excl)
                        blocked2 = _segments_blocked_by_buildings(xp, Cb[cand_idx], G[cand_idx], box_min, box_max, excl)
                        blocked3 = _segments_blocked_by_buildings(xp, G[cand_idx], o_rep, box_min, box_max, excl)
                    blocked = blocked1 | blocked2 | blocked3
                    n_ground_bounce_occl_blocked += int(to_numpy(blocked).sum())
                    occl_full = xp.zeros(valid2.shape[0], dtype=bool)
                    occl_full[cand_idx] = blocked
                    valid2 = valid2 & (~occl_full)

            theta_ground = xp.arccos(xp.clip(cos_ground, 0.0, 1.0))
            theta_ground_np = to_numpy(theta_ground)
            R_eff_ground_np = effective_specular_reflectivity(ground_material, theta_ground_np, wavelength)
            R_eff_ground = xp.asarray(R_eff_ground_np)

            L_total = R_asc + xp.linalg.norm(G - Cb, axis=1) + xp.linalg.norm(o[None, :] - G, axis=1)
            # taper1 reused deliberately, not recomputed: the wall is the
            # limiting aperture for the WHOLE dihedral's azimuthal
            # persistence (the ground patch is effectively unbounded by
            # comparison), so the same per-facet taper that gates leg1
            # gates leg2 too -- see _azimuth_sinc_taper's docstring.
            #
            # taper_range (_range_distributed_taper) was tried here and
            # MEASURED WORSE across every metric (SSIM 0.484->0.413,
            # coherence 0.178->0.017, see _taper_ab_test.py) -- NOT wired
            # in. Root cause, diagnosed after the fact: the function
            # evaluates its sinc argument at a single MEAN wavelength (the
            # same simplification this codebase already uses for the
            # roughness/azimuth terms, where bandwidth is modest enough
            # relative to those angle-dependencies to not matter). That
            # simplification is wrong for THIS term specifically: the
            # true effect is inherently frequency-dependent -- a wideband
            # signal integrated correctly range-RESOLVES different wall
            # heights into different range bins during backprojection
            # (which already computes phase2 per exact frequency, not a
            # mean wavelength) rather than needing them pre-collapsed
            # into one suppressed scalar. Collapsing to a mean-wavelength
            # scalar before backprojection ever runs destroys exactly the
            # frequency diversity that would otherwise do this correctly,
            # producing over-aggressive, physically wrong suppression
            # instead. A correct version would need the taper computed
            # per frequency bin (K,F), not once per facet (F,) -- a
            # bigger restructure than fits this pass; left unimplemented
            # rather than shipped as a measured regression.
            amp_eff2 = xp.where(valid2, Ab * xp.abs(cos_inc1) * taper1 * R_eff_ground * cos_ground, 0.0)

            R_equiv2 = L_total / 2.0
            dR2 = R_equiv2 - R_ref
            phase2 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR2) / C)
            # _range_distributed_taper_kf: implemented, tested, ALSO
            # measured worse (SSIM 0.484->0.423, coherence 0.178->0.017 --
            # nearly identical to the broken scalar version, see
            # _range_distributed_taper's docstring) -- NOT wired in.
            #
            # The frequency-resolved fix didn't help because the diagnosed
            # cause was wrong: this isn't a frequency-collapsing bug, it's
            # a wrong physical model. Measured directly (_diagnose_dl.py):
            # DeltaL (round-trip path length between a wall's top and
            # bottom reflecting to the ground) reaches 200m on this scene,
            # thousands of wavelengths at X-band. That's not a coherent
            # aperture that tapers gracefully off-broadside -- treating
            # the ENTIRE building height as one coherently-summed point
            # source predicts near-total cancellation, correctly, GIVEN
            # that (wrong) assumption. But that's not what ray tracing
            # actually does: it never coherently sums the whole wall into
            # one point at all -- each ray's hit height independently
            # lands at ITS OWN true ground position and gets scored
            # there, i.e. genuinely distinct scattering centers, not one
            # point with a directivity taper. No amplitude taper on a
            # SINGLE point can reproduce that -- a taper can only make
            # one point's return weaker, it can't make the same energy
            # reappear at several genuinely different image pixels. The
            # actual fix would be sampling a handful of discrete heights
            # along the wall (each its own scored point, closer to what
            # ray tracing already does), not a taper on one -- a bigger,
            # structurally different change than either attempt here.
            #
            # ground_bounce_n_sub > 1: that fix, implemented
            # (_ground_bounce_subpoint_contrib) -- replaces the single
            # facet-center point with n_sub points spread along the
            # wall's height, each scored at its own exact position, same
            # validity/occlusion decision (valid2) as the single-point
            # version. amp_eff2/phase2 above are unused in this branch,
            # left computed for the n_sub=1 (default) path and for
            # comparison/debugging.
            if regime_adaptive:
                # Per-facet decision, not a global one: escalate only
                # the facets whose own DeltaL is actually large relative
                # to what this system's bandwidth can resolve anyway --
                # see this function's docstring for why this replaces
                # "which model is more physically correct" with "which
                # facets can the cheap model safely handle."
                dL = _wall_height_dL(xp, o, Cb, Nb, HVb, d_in, half_extent_g)
                escalate_mask = xp.abs(dL) > escalate_threshold
                safe_mask = valid2 & (~escalate_mask)
                esc_mask = valid2 & escalate_mask
                n_escalated_total += int(to_numpy(esc_mask).sum())

                amp_eff2_safe = xp.where(safe_mask, Ab * xp.abs(cos_inc1) * taper1 * R_eff_ground * cos_ground, 0.0)
                leg2_safe = (amp_eff2_safe[None, :] * phase2).sum(axis=1)
                leg2_escalated = _ground_bounce_subpoint_contrib(
                    xp, o, R_ref, freqs, Cb, Nb, Ab, HVb, d_in, cos_inc1, taper1,
                    esc_mask, half_extent_g, ground_material, wavelength, escalate_n_sub)
                leg2_contrib = leg2_safe + leg2_escalated
            elif ground_bounce_n_sub > 1:
                zw = None
                if ground_bounce_z_weights is not None:
                    zw = xp.asarray(ground_bounce_z_weights)
                leg2_contrib = _ground_bounce_subpoint_contrib(
                    xp, o, R_ref, freqs, Cb, Nb, Ab, HVb, d_in, cos_inc1, taper1,
                    valid2, half_extent_g, ground_material, wavelength, ground_bounce_n_sub,
                    z_weights=zw)
            else:
                leg2_contrib = (amp_eff2[None, :] * phase2).sum(axis=1)
            s[p, :] += leg2_contrib
            if return_components:
                s_by_leg['leg2'][p, :] = leg2_contrib

            n_ground_bounce_total += int(to_numpy(valid2).sum())

        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t_total += (time.perf_counter() - t0)

    stats = dict(n_facets=Cb.shape[0], t_total_s=t_total,
                 t_per_pulse_ms=t_total / n_pulses * 1000.0,
                 n_ground_bounce_paths=n_ground_bounce_total,
                 n_ground_bounce_occl_blocked=n_ground_bounce_occl_blocked,
                 occlusion_check=occlusion_check, occlusion_method=occlusion_method if occlusion_check else None,
                 ground_material=ground_material)
    if regime_adaptive:
        stats['regime_adaptive'] = True
        stats['resolution_cell_m'] = resolution_cell
        stats['escalate_threshold_m'] = escalate_threshold
        stats['escalate_n_sub'] = escalate_n_sub
        stats['n_escalated_paths'] = n_escalated_total
        stats['n_escalated_fraction'] = n_escalated_total / max(n_ground_bounce_total, 1)
    if return_components:
        stats['s_by_leg'] = s_by_leg
    return s, stats


def _next_surface_hit(xp, C, d, facets_buildings, ground_half_extent, exclude_idx=None):
    """Given rays leaving points C in direction d, find the NEAREST thing
    each one hits next -- another building facet, or the ground plane --
    whichever is actually closer along the ray. No search, no ray
    bouncing: both candidates are found exactly and directly (one
    ray_facet_intersect call for the wall candidate, one closed-form
    plane intersection for the ground candidate), then compared once by
    distance. This is what makes a general (not fixed-order) closed-form
    multi-bounce chain possible -- the earlier version of this file
    picked ONE fixed next-surface type (ground) and was wrong whenever a
    ray actually hit another wall first, which the user correctly flagged:
    real triple-bounce paths can be wall-wall-wall, wall-wall-ground,
    wall-ground-wall, etc. -- whichever surface is truly nearest, not
    whichever this code assumed.

    Returns: hit_pt (F,3), hit_normal (F,3), is_ground (F,) bool,
    wall_idx (F,) int (meaningful only where ~is_ground), valid (F,) bool.
    """
    F = C.shape[0]

    # wall candidate: exact, deterministic, one ray per row -- same
    # intersection math the ray tracer uses, just with an analytically
    # known direction instead of a guessed aim-grid one
    hit_pt_w, hit_amp_w, cos_w, hit_mask_w, idx_w = ray_facet_intersect(xp, C, d, facets_buildings)
    if exclude_idx is not None:
        hit_mask_w = hit_mask_w & (idx_w != exclude_idx)
    t_w = xp.where(hit_mask_w, xp.linalg.norm(hit_pt_w - C, axis=1), xp.inf)

    # ground candidate: closed-form plane z=0 intersection, same formula
    # _reflect_and_intersect_ground_plane uses internally
    heading_down = d[:, 2] < -1e-9
    d_z_safe = xp.where(heading_down, d[:, 2], -1.0)
    t_g_raw = xp.where(heading_down, -C[:, 2] / d_z_safe, xp.inf)
    G = C + xp.where(xp.isinf(t_g_raw), 0.0, t_g_raw)[:, None] * d
    in_bounds_g = (xp.abs(G[:, 0]) <= ground_half_extent) & (xp.abs(G[:, 1]) <= ground_half_extent)
    valid_g = heading_down & (t_g_raw > 1e-9) & in_bounds_g
    t_g = xp.where(valid_g, t_g_raw, xp.inf)

    use_ground = t_g < t_w
    valid = xp.isfinite(xp.minimum(t_g, t_w))
    hit_pt = xp.where(use_ground[:, None], G, hit_pt_w)
    ground_normal = xp.zeros((F, 3), dtype=C.dtype)
    ground_normal[:, 2] = 1.0
    hit_normal = xp.where(use_ground[:, None], ground_normal, facets_buildings['normal'][idx_w])
    return hit_pt, hit_normal, use_ground, idx_w, valid


def run_asc_cached_order3_wall_ground_wall(xp, on_gpu, facets_buildings, facets_ground, plat, freqs, ref_pos,
                                            ground_material=None, occlusion_check=True,
                                            return_components=False):
    """
    Closed-form triple bounce, general surface order: sensor -> wall1 ->
    (nearest of {ground, another wall}) -> (nearest of {ground, another
    wall}) -> sensor. No ray tracing anywhere in this function.

    Function name kept from the first version of this (wall-ground-wall
    only) for continuity, but it now covers wall-wall-wall,
    wall-wall-ground, wall-ground-wall, and wall-ground-ground(degenerate,
    naturally excluded -- see _next_surface_hit: a ray leaving a flat
    ground bounce heading up can't have heading_down true again without
    an intervening wall reflection, so back-to-back ground hits can't
    happen by construction, not by a special case). Bounce 1 is always a
    wall (this codebase's convention throughout: order-1 rays only ever
    target building facets, ground only enters from bounce 2 on).

    This directly answers the session's live question: the azimuth taper
    (_azimuth_sinc_taper) gives each dihedral corner a beamwidth, but
    that beamwidth assumes an unobstructed view -- it says nothing about
    whether some OTHER piece of the scene sits in the way. An earlier
    docstring in this file called a ray-tracing-free triple bounce
    unavailable because "which surface does the reflected ray hit next"
    looked like a search problem. It isn't: once a bounce's exact
    outgoing direction is known analytically (plain law of reflection off
    a KNOWN point+normal), finding what it hits next is one deterministic
    comparison between two exactly-computed candidates (nearest wall,
    nearest ground point) -- see _next_surface_hit. That reuses the
    identical intersection math the dense-SBR ray tracer runs; the
    difference is dense-SBR needs many exploratory rays per pulse because
    it doesn't know where a ray will land ahead of time, while here the
    direction is exact at every step, so one test per candidate per step
    suffices. Zero aim-grid, zero statistical discovery.

    Per-bounce reflectivity is generalized uniformly across wall/ground:
    _azimuth_sinc_taper already returns 1.0 (no rolloff) for any surface
    whose normal has ~zero horizontal component -- which the ground
    plane's (0,0,1) normal always does -- so calling it at every bounce
    without branching on surface type is correct by construction, not a
    special case. The per-bounce material reflectivity does still branch
    (building Ab vs. ground's angle-dependent effective_specular_
    reflectivity), because those are genuinely different physical
    quantities, not a modeling shortcut.

    Occlusion reuses _segment_occluded_by_facets (the same exact,
    per-facet segment test run_asc_cached_multibounce already uses for
    the double-bounce case) across all four legs of the path.

    ground_material, occlusion_check, return_components: same meaning as
    run_asc_cached_multibounce. No regime-adaptive / sub-point options
    here -- this is a first pass at the geometry-and-occlusion question,
    not a rerun of the range-taper investigation.
    """
    n_pulses = plat.shape[0]
    K = freqs.shape[0]
    n_facets = facets_buildings['center'].shape[0]

    Cb, Nb, Ab = facets_buildings['center'], facets_buildings['normal'], facets_buildings['amp']
    L_wall = 2.0 * facets_buildings['half_u']
    normal_xy = Nb[:, :2]
    half_extent_g = float(facets_ground.get('footprint_half_extent', to_numpy(facets_ground['half_u'])[0]))
    if ground_material is None:
        ground_material = facets_ground.get('material', 'dry_soil')
    wavelength = C / float(to_numpy(freqs).mean())

    self_idx = xp.arange(n_facets)

    def _bounce_term(d_prev, hit_pt, hit_normal, is_ground, wall_idx):
        """Reflectivity*cos*taper at one bounce, generalized over
        wall/ground (see docstring)."""
        cos_k = xp.abs(xp.sum(-d_prev * hit_normal, axis=1))
        L_k = xp.where(is_ground, 1.0, L_wall[wall_idx])          # ignored for ground (taper -> 1)
        normal_k_xy = xp.where(is_ground[:, None], 0.0, normal_xy[wall_idx])
        taper_k = _azimuth_sinc_taper(xp, wavelength, L_k, normal_k_xy, -d_prev[:, :2])
        theta_k = xp.arccos(xp.clip(cos_k, 0.0, 1.0))
        R_ground_k = xp.asarray(effective_specular_reflectivity(ground_material, to_numpy(theta_k), wavelength))
        refl_k = xp.where(is_ground, R_ground_k, Ab[wall_idx])
        d_out_k = d_prev - 2.0 * xp.sum(d_prev * hit_normal, axis=1, keepdims=True) * hit_normal
        return refl_k * cos_k * taper_k, d_out_k

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    t_total = 0.0
    n_candidates_total = 0
    n_occl_blocked = 0
    s_by_leg = None
    if return_components:
        s_by_leg = dict(order3=xp.zeros((n_pulses, K), dtype=xp.complex128))

    for p in range(n_pulses):
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)
        t0 = time.perf_counter()

        # bounce 1: always a wall (this codebase's convention)
        look = Cb - o[None, :]
        R1 = xp.linalg.norm(look, axis=1)
        d_in = look / R1[:, None]
        cos_inc1 = xp.sum(-d_in * Nb, axis=1)
        visible1 = cos_inc1 > 0
        taper1 = _azimuth_sinc_taper(xp, wavelength, L_wall, normal_xy, -d_in[:, :2])
        d_out1 = d_in - 2.0 * xp.sum(d_in * Nb, axis=1, keepdims=True) * Nb
        term1 = xp.where(visible1, Ab * xp.abs(cos_inc1) * taper1, 0.0)

        # bounce 2: nearest of {another wall, the ground}
        hit_pt2, hit_normal2, is_ground2, idx2, valid2 = _next_surface_hit(
            xp, Cb, d_out1, facets_buildings, half_extent_g, exclude_idx=self_idx)
        term2, d_out2 = _bounce_term(d_out1, hit_pt2, hit_normal2, is_ground2, idx2)

        # bounce 3: nearest of {another wall, the ground}, excluding the
        # facet just left (only meaningful when bounce 2 was a wall --
        # harmless no-op otherwise since idx2 is unused when is_ground2)
        hit_pt3, hit_normal3, is_ground3, idx3, valid3_geom = _next_surface_hit(
            xp, hit_pt2, d_out2, facets_buildings, half_extent_g, exclude_idx=idx2)
        term3, d_out3 = _bounce_term(d_out2, hit_pt3, hit_normal3, is_ground3, idx3)

        # A 90-degree wall-ground dihedral is an EXACT retroreflector by
        # construction (walls are vertical, ground is horizontal, always
        # perpendicular) -- that's why leg2 in run_asc_cached_multibounce
        # can just assume the final leg goes straight back to the sensor
        # with no separate check. A wall-wall pair is NOT guaranteed to be
        # mutually perpendicular (two buildings can face any direction
        # relative to each other), so that assumption silently stopped
        # being physically justified the moment a second wall entered the
        # chain -- this was the actual bug behind the first test's 10x
        # overcount and near-zero coherence, not a missing occlusion
        # case. Fix: verify the bounce's TRUE outgoing direction (d_out3,
        # already computed by the law of reflection, not assumed) is
        # actually close to the straight line back to the sensor, the
        # same thing _return_visible verifies for real on the ray-traced
        # side via an explicit shadow ray -- this is the closed-form
        # equivalent, a direction comparison instead of a ray cast.
        to_sensor = o[None, :] - hit_pt3
        to_sensor_n = to_sensor / xp.linalg.norm(to_sensor, axis=1, keepdims=True)
        pointing_cos = xp.sum(d_out3 * to_sensor_n, axis=1)
        valid_pointing = pointing_cos > 0.9997   # ~1.4 deg tolerance

        valid_all = visible1 & valid2 & valid3_geom & valid_pointing

        if occlusion_check:
            cand_idx = xp.nonzero(valid_all)[0]
            if cand_idx.shape[0] > 0:
                o_rep = xp.broadcast_to(o[None, :], (cand_idx.shape[0], 3))
                blocked1 = _segment_occluded_by_facets(xp, o_rep, Cb[cand_idx], facets_buildings)
                blocked2 = _segment_occluded_by_facets(xp, Cb[cand_idx], hit_pt2[cand_idx], facets_buildings)
                blocked3 = _segment_occluded_by_facets(xp, hit_pt2[cand_idx], hit_pt3[cand_idx], facets_buildings)
                blocked4 = _segment_occluded_by_facets(xp, hit_pt3[cand_idx], o_rep, facets_buildings)
                blocked = blocked1 | blocked2 | blocked3 | blocked4
                n_occl_blocked += int(to_numpy(blocked).sum())
                occl_full = xp.zeros(valid_all.shape[0], dtype=bool)
                occl_full[cand_idx] = blocked
                valid_all = valid_all & (~occl_full)

        amp_eff3 = xp.where(valid_all, term1 * term2 * term3, 0.0)

        L_total3 = (R1 + xp.linalg.norm(hit_pt2 - Cb, axis=1)
                    + xp.linalg.norm(hit_pt3 - hit_pt2, axis=1) + xp.linalg.norm(o[None, :] - hit_pt3, axis=1))
        R_equiv3 = L_total3 / 2.0
        dR3 = R_equiv3 - R_ref
        phase3 = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR3) / C)
        contrib3 = (amp_eff3[None, :] * phase3).sum(axis=1)
        s[p, :] += contrib3
        if return_components:
            s_by_leg['order3'][p, :] = contrib3

        n_candidates_total += int(to_numpy(valid_all).sum())

        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t_total += (time.perf_counter() - t0)

    stats = dict(n_facets=n_facets, t_total_s=t_total, t_per_pulse_ms=t_total / n_pulses * 1000.0,
                 n_order3_paths=n_candidates_total, n_order3_occl_blocked=n_occl_blocked,
                 occlusion_check=occlusion_check, ground_material=ground_material)
    if return_components:
        stats['s_by_leg'] = s_by_leg
    return s, stats


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
