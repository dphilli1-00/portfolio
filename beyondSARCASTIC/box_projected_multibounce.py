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


def run_asc_box_projected_multibounce(xp, on_gpu, facets_buildings, facets_ground, plat, freqs, ref_pos,
                                       ground_material=None, return_components=False, include_order3=False,
                                       progress=False, low_precision_envelope=False, free_pool_every=None,
                                       leg2_occlusion_check=False, split_leg2_by_target=False,
                                       leg2_retroreflection_check=False, retro_beamwidth_mult=3.0,
                                       leg2_building_enabled=True, leg2_retro_taper=False):
    """Same leg-1 as run_asc_cached_multibounce (sensor->facet, closed
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
    pbar = tqdm(range(n_pulses), desc="ASC pulses", disable=not progress)
    for p in pbar:
        t0 = time.perf_counter()
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)

        look = Cb - o[None, :]
        R_asc = xp.linalg.norm(look, axis=1)
        d_in = look / R_asc[:, None]
        cos_inc1 = xp.sum(-d_in * Nb, axis=1)
        visible1 = cos_inc1 > 0

        # amp_eff1_geom / dR1: real, (F,)-scale, cheap regardless of scene
        # size -- alpha=1.0 (canonical GTD flat-plate/dihedral/trihedral
        # value, Potter & Moses 1996), L_el=0 (validated for a wall-ground
        # corner, task #32, 0.971 vs 0.44 coherence) drive the actual
        # envelope, computed per facet-chunk below, not here.
        amp_eff1_geom = xp.where(visible1, Ab * xp.abs(cos_inc1), 0.0)   # (F,) real, taper in env (below)
        dR1 = R_asc - R_ref
        counts['leg1'] += int(to_numpy(xp, visible1).sum())

        hit = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, half_extent_g, box_min, box_max, fbid)
        G, valid_geom2, d_out = hit['G'], hit['valid'], hit['d_out']
        surf_normal, is_ground, building_hit = hit['surf_normal'], hit['is_ground'], hit['building_hit']

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
        for cs in range(0, F, chunk_size):
            ce = min(cs + chunk_size, F)
            env_chunk = asc_visible_envelope(
                xp, o, Cb[cs:ce], freqs, visible1[cs:ce], alpha=1.0, L_az=L_wall[cs:ce], u_hat=Ub[cs:ce],
                L_el=0.0, low_precision=low_precision_envelope, mask_invisible=False)   # (chunk, K)

            phase1_c = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR1[cs:ce]) / C)
            leg1_contrib += (amp_eff1_geom[None, cs:ce] * phase1_c * env_chunk.T).sum(axis=1)

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
    return s, stats
