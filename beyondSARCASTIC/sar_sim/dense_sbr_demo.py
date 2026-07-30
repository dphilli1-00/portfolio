"""
Primitive-level dense SBR (Shooting and Bouncing Rays) forward model --
the "must re-trace per pulse" approach, implemented for real, not just
cited, so we get an actual measured R_ANSYS-style number on real
hardware instead of a scaled/borrowed one.

Contrast with point_scatterer_demo.py: that script sums a FIXED list of
point scatterers with a closed-form phase formula every pulse (cheap --
this is the ASC-cached style: the expensive geometric work is assumed
already done, once). This script instead does the expensive work itself,
every single pulse: cast a grid of primary rays from the platform toward
the scene, test EVERY ray against EVERY facet for intersection, keep the
nearest hit, and turn that into a scattering contribution. That's dense
SBR at a primitive level -- single-bounce only, flat rectangular facets
instead of a full triangle mesh, no PTD/UTD diffraction, no multi-bounce.
It's the "Gotcha" slide's claim made runnable: cost scales with
rays x facets x pulses, recomputed from scratch every pulse.

Backend-agnostic: `import numpy as xp` normally, or run with --gpu to
`import cupy as xp` instead (same code, drop-in GPU arrays). This
sandbox has no GPU (checked: no nvidia-smi, no CUDA) -- only the CPU
path is timed here. The --gpu path is meant to be run on a real CUDA
machine (e.g. a laptop with an NVIDIA GPU); cupy must be installed
there matching the local CUDA version (pip install cupy-cudaXXx).

Usage:
    python3 dense_sbr_demo.py                  # CPU (numpy), default sizes
    python3 dense_sbr_demo.py --gpu             # GPU (cupy), same algorithm
    python3 dense_sbr_demo.py --footprint 1000 --density 200 --rays 100 --pulses 200
"""

import argparse
import time
import json
import numpy as np

from materials import diffuse_coefficient

C = 299_792_458.0


def get_backend(use_gpu):
    if use_gpu:
        import cupy as xp
        return xp, True
    return np, False


# ----------------------------------------------------------------------
# Scene: buildings as axis-aligned boxes -> 4 vertical walls + 1 flat roof
# ----------------------------------------------------------------------
def make_building_scene(xp, footprint_m, density_per_km2, seed=0,
                         min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0),
                         roof_refl_scale=1.3):
    """
    footprint_m: side length of the square scene (m), e.g. 1000 for 1km x 1km
    density_per_km2: structures per km^2 (matches the deck's TerraSAR-X
        anchored assumption: 20,000 structures / 100km^2 = 200/km^2)
    roof_refl_scale: roof amplitude = wall reflectivity * this factor.
        Flat roofs are close to normal-incidence for a platform near
        nadir (this scene's ~21 deg depression angle isn't far off), so
        they tend to return more strongly than a grazing-incidence wall
        -- 1.3x is a rough bump, not a measured material constant. Set
        to 1.0 for "roof = same reflectivity as walls" if you'd rather
        not carry that assumption.

    Buildings are closed boxes now: 4 vertical walls + 1 flat horizontal
    roof at z=height, normal (0,0,1). No floor facet -- a monostatic
    airborne/spaceborne radar looking down never has line of sight to a
    building's underside, so it would never intersect one; adding it
    would either sit dead (extra unused facets, wasted compute) or, if
    ray_facet_intersect's front-face test were ever weakened, become the
    same kind of invisible-backface leak the roof fix here was meant to
    close. Rays that already miss the roof and every wall now correctly
    report a miss (bare ground) instead of leaking through an open top.

    Returns facet arrays (all shape (F,3) or (F,)):
        center, u_hat, v_hat, normal, half_u, half_v, reflect_amp
    """
    rng = np.random.default_rng(seed)
    area_km2 = (footprint_m / 1000.0) ** 2
    n_buildings = max(1, int(round(density_per_km2 * area_km2)))

    cx = rng.uniform(-footprint_m / 2, footprint_m / 2, n_buildings)
    cy = rng.uniform(-footprint_m / 2, footprint_m / 2, n_buildings)
    w = rng.uniform(min_wh[0], max_wh[0], n_buildings)   # x-extent
    d = rng.uniform(min_wh[1], max_wh[1], n_buildings)   # y-extent
    h = rng.uniform(min_wh[2], max_wh[2], n_buildings)   # height
    refl = rng.uniform(0.6, 1.0, n_buildings)            # per-building wall reflectivity

    centers, u_hats, v_hats, normals, half_u, half_v, amps = [], [], [], [], [], [], []

    for i in range(n_buildings):
        hx, hy, hz = w[i] / 2, d[i] / 2, h[i] / 2
        bcx, bcy = cx[i], cy[i]
        faces = [
            # (face center offset, u_hat (in-plane), v_hat (in-plane), normal, half_u, half_v, amp)
            # 4 vertical walls: +x, -x, +y, -y
            ((hx, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), hy, hz, refl[i]),
            ((-hx, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), hy, hz, refl[i]),
            ((0, hy, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), hx, hz, refl[i]),
            ((0, -hy, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0), hx, hz, refl[i]),
            # flat roof, z = building height, facing straight up
            ((0, 0, hz), (1, 0, 0), (0, 1, 0), (0, 0, 1), hx, hy, refl[i] * roof_refl_scale),
        ]
        for (off, u_h, v_h, n_h, hu, hv, fa) in faces:
            centers.append([bcx + off[0], bcy + off[1], hz + off[2]])
            u_hats.append(u_h)
            v_hats.append(v_h)
            normals.append(n_h)
            half_u.append(hu)
            half_v.append(hv)
            amps.append(fa)

    to_xp = lambda a: xp.asarray(np.array(a, dtype=np.float64))
    return dict(
        center=to_xp(centers), u_hat=to_xp(u_hats), v_hat=to_xp(v_hats),
        normal=to_xp(normals), half_u=to_xp(half_u), half_v=to_xp(half_v),
        amp=to_xp(amps), n_buildings=n_buildings, n_facets=len(centers),
        # per-building metadata (numpy, not xp -- consumed for crop/report
        # bookkeeping on the host, not in any GPU math), so callers doing
        # per-building analysis (e.g. per-sector SSIM) don't have to
        # reverse-engineer footprint/height from the flattened facet list.
        building_cx=cx, building_cy=cy, building_w=w, building_d=d, building_h=h,
    )


def make_ground_facet(xp, footprint_m, margin_m=0.0, material='dry_soil', patch_m=40.0):
    """
    A GRID of flat facets tiling z=0 (normal +z), patch_m on a side each,
    spanning the scene footprint (+ margin) -- a ray-traceable ground
    plane, NOT one giant facet.

    That last part matters and was a real bug the first version of this
    function had: every OTHER facet-scoring path in this codebase
    (multibounce_demo._score_paths) evaluates a bounce's range/phase
    using its facet's CENTER as a stand-in for "where the ray actually
    hit" -- a fine approximation for building facets a few meters across,
    but nonsensical for a single facet spanning an entire 1km-scale
    footprint, where the true hit point could be hundreds of meters from
    that facet's nominal center. Running validation_multibounce_compare.py
    against the one-giant-facet version produced SSIM=0.05 / coherence
    =0.01 between the ray-traced and closed-form double-bounce images --
    not because ASC's no-occlusion approximation is that bad, but because
    every ray-traced ground bounce was being range-scored at (0,0,0)
    regardless of where it actually landed, while the closed-form ASC
    side (which computes its ground intersection analytically, no
    facet-center lookup involved at all) used the correct location.
    Tiling into patch_m-sized facets restores "center is a valid proxy"
    without any special-casing in the scoring code -- same assumption
    building facets already rely on, just applied to the ground too.

    This is NOT the same thing as make_ground_clutter()'s diffuse
    point-scatterer background. That layer models incoherent, isotropic
    bare-earth texture and deliberately has no facet/normal/occlusion
    behavior at all -- added identically to both branches so it does NOT
    touch the SBR-vs-ASC comparison. This ground grid is the opposite: a
    coherent, specular reflector ray tracing can actually bounce off of,
    needed so a wall-ground dihedral path has something to close the
    loop against.

    The 'amp' field here is a PLACEHOLDER (1.0) -- the ground's actual
    contribution to a multi-bounce path is angle- and wavelength-
    dependent (materials.effective_specular_reflectivity), computed per-
    path at scoring time, not baked in per-patch here. See
    multibounce_demo.py's _score_paths / run_asc_cached_multibounce.

    Used only by the multi-bounce tracer (bounce order >= 2); the
    existing single-bounce dense-SBR path never sees this facet, so
    Tier 2's already-validated single-bounce numbers are unaffected.
    """
    half = footprint_m / 2.0 + margin_m
    n_side = max(1, int(np.ceil((2.0 * half) / patch_m)))
    edges = np.linspace(-half, half, n_side + 1)
    centers_1d = (edges[:-1] + edges[1:]) / 2.0
    half_patch = float((edges[1] - edges[0]) / 2.0)

    gx, gy = np.meshgrid(centers_1d, centers_1d)
    cx, cy = gx.ravel(), gy.ravel()
    n = cx.shape[0]

    to_xp = lambda a: xp.asarray(np.array(a, dtype=np.float64))
    centers = np.column_stack([cx, cy, np.zeros(n)])
    return dict(
        center=to_xp(centers),
        u_hat=to_xp(np.tile([[1.0, 0.0, 0.0]], (n, 1))),
        v_hat=to_xp(np.tile([[0.0, 1.0, 0.0]], (n, 1))),
        normal=to_xp(np.tile([[0.0, 0.0, 1.0]], (n, 1))),
        half_u=to_xp(np.full(n, half_patch)), half_v=to_xp(np.full(n, half_patch)),
        amp=to_xp(np.full(n, 1.0)),
        material=material, n_ground_facets=n, patch_m=float(edges[1] - edges[0]),
        footprint_half_extent=half,
    )


def concat_facets(xp, *facet_dicts):
    """Concatenate two or more facets dicts (e.g. buildings + ground) into
    one combined array set, for tracing bounces against the union of both.
    Only combines the (F,3)/(F,) geometric arrays -- not the per-building
    metadata (building_cx/cy/w/d/h), which callers doing per-building
    analysis should keep reading from the original buildings-only dict."""
    keys = ['center', 'u_hat', 'v_hat', 'normal', 'half_u', 'half_v', 'amp']
    return {k: xp.concatenate([fd[k] for fd in facet_dicts], axis=0) for k in keys}


# ----------------------------------------------------------------------
# Dense SBR: cast rays, test every ray against every facet, keep nearest hit
# ----------------------------------------------------------------------
def _ray_facet_intersect_chunk(xp, ray_o, ray_d, facets):
    """Un-chunked core: same math as before, called per chunk by
    ray_facet_intersect. Do not call directly with large R*F -- see that
    function's docstring for why.

    ray_o is (R,3) here (per-ray origin) -- generalized from the original
    single-origin (3,) version so bounce>=2 rays (which each start at a
    different hit point from the previous bounce, not a shared platform
    position) can reuse this exact intersection core instead of a second
    copy of the math. Single-origin callers broadcast to (R,3) once in
    ray_facet_intersect before chunking; this function no longer cares
    which case it's in."""
    C_ = facets['center']       # (F,3)
    N_ = facets['normal']       # (F,3)
    U_ = facets['u_hat']        # (F,3)
    V_ = facets['v_hat']        # (F,3)
    HU = facets['half_u']       # (F,)
    HV = facets['half_v']       # (F,)
    AMP = facets['amp']         # (F,)

    denom = ray_d @ N_.T                       # (R,F)  d . n
    valid_denom = xp.abs(denom) > 1e-8

    # (R,F) numerator: (C_f - O_r) . N_f -- per-ray now, not shared across
    # rays, since each ray may start from a different origin
    num = xp.sum((C_[None, :, :] - ray_o[:, None, :]) * N_[None, :, :], axis=2)   # (R,F)
    t = num / xp.where(valid_denom, denom, xp.inf)   # (R,F)

    hit_pt = ray_o[:, None, :] + t[:, :, None] * ray_d[:, None, :]   # (R,F,3)
    rel = hit_pt - C_[None, :, :]                                       # (R,F,3)
    a = xp.sum(rel * U_[None, :, :], axis=2)   # (R,F) local u-coord
    b = xp.sum(rel * V_[None, :, :], axis=2)   # (R,F) local v-coord

    in_bounds = (xp.abs(a) <= HU[None, :]) & (xp.abs(b) <= HV[None, :])
    # front-face only: ray_d . N < 0 means the ray approaches from the
    # outward-normal side (a legitimate front hit). Without this, a ray
    # that clears a near wall's top edge can pass into the open-topped
    # building and register a "hit" on a far wall's INTERIOR (back) face
    # -- same plane, same in-bounds quad, just struck from behind. That's
    # a spurious scatterer no real radar (or the ASC model, whose backface
    # cull excludes it by construction) would ever see. Caught this via a
    # systematic extra return per building in the dense-SBR image that
    # ASC-cached didn't have, at exactly the same range bin for every
    # building -- too consistent to be noise.
    front_face = denom < 0
    # t > 1e-6, not t > 0: bounce>=2 rays start exactly ON the facet they
    # just left (offset a small epsilon along the normal at the call site,
    # but floating-point residue can still land t right at ~0) -- a bare
    # t > 0 threshold let those rays immediately "re-hit" their own facet
    # of origin. 1e-6 is negligible at this scene's meter-to-kilometer
    # scale but clears that self-intersection.
    valid = valid_denom & in_bounds & (t > 1e-6) & front_face
    t_masked = xp.where(valid, t, xp.inf)

    nearest_idx = xp.argmin(t_masked, axis=1)          # (R,)
    nearest_t = xp.take_along_axis(t_masked, nearest_idx[:, None], axis=1)[:, 0]
    hit_mask = xp.isfinite(nearest_t)

    # gather winning facet's amplitude / normal / hit point
    hit_amp = AMP[nearest_idx]
    hit_normal = N_[nearest_idx]
    cos_inc = xp.abs(xp.sum(-ray_d * hit_normal, axis=1))   # (R,)
    hit_point = ray_o + xp.where(hit_mask, nearest_t, 0.0)[:, None] * ray_d

    return hit_point, hit_amp, cos_inc, hit_mask, nearest_idx


def ray_facet_intersect(xp, ray_o, ray_d, facets, max_chunk_bytes=200_000_000):
    """
    ray_o: (3,) a single shared origin (e.g. platform position for this
        pulse -- the bounce-1 case), OR (R,3) a per-ray origin (bounce>=2,
        where every ray starts at its own previous-bounce hit point).
        A (3,) input is broadcast to (R,3) once here.
    ray_d: (R,3) ray directions (normalized)
    facets: dict of (F,3)/(F,) arrays from make_building_scene (or a
        concat_facets(...) combination, e.g. buildings + ground)

    Returns: hit_point (R,3), hit_amp (R,), hit_cos (R,), hit_mask (R,) bool,
    nearest_idx (R,) int -- facet index each ray landed on (only meaningful
    where hit_mask is True; used by callers that need to deduplicate
    multiple rays landing on the same facet, e.g. to avoid coherently
    summing N near-identical ray contributions as if they were N separate
    scatterers -- see sbr_vs_asc_compare.run_dense_sbr_timed).
    Vectorized over (R, F) -- this is the expensive O(rays x facets) step,
    repeated fresh every pulse. That repetition is the entire point.

    Chunked over rays with a bounded per-chunk memory budget
    (max_chunk_bytes, applied to the largest (chunk, F, 3) float64 array).
    The un-chunked version materializes several full (R, F) and (R, F, 3)
    float64 arrays at once -- at R=67,600 rays x F=1,000 facets (a scene
    this project actually recommended running), that's ~6GB of named
    arrays alone, before counting the extra copies broadcast operations
    hold briefly. On a 6GB-class laptop GPU that's a real out-of-memory
    condition, and because a laptop GPU also drives the display, a CUDA
    OOM there can trigger a WDDM driver timeout that hangs or crashes the
    whole machine, not just the Python process -- which is what actually
    happened running this. Chunking bounds peak memory to
    max_chunk_bytes regardless of how large the scene gets, instead of
    relying on the caller to guess a safe rays x facets product.
    """
    R = ray_d.shape[0]
    F = facets['center'].shape[0]

    if ray_o.ndim == 1:
        ray_o = xp.broadcast_to(ray_o[None, :], ray_d.shape)

    bytes_per_ray_row = F * 3 * 8    # the (chunk, F, 3) float64 arrays dominate
    chunk_size = max(1, int(max_chunk_bytes / max(bytes_per_ray_row, 1)))

    if chunk_size >= R:
        return _ray_facet_intersect_chunk(xp, ray_o, ray_d, facets)

    hit_points, hit_amps, cos_incs, hit_masks, nearest_idxs = [], [], [], [], []
    for start in range(0, R, chunk_size):
        d_chunk = ray_d[start:start + chunk_size]
        o_chunk = ray_o[start:start + chunk_size]
        hp, ha, ci, hm, ni = _ray_facet_intersect_chunk(xp, o_chunk, d_chunk, facets)
        hit_points.append(hp)
        hit_amps.append(ha)
        cos_incs.append(ci)
        hit_masks.append(hm)
        nearest_idxs.append(ni)

    return (xp.concatenate(hit_points, axis=0), xp.concatenate(hit_amps, axis=0),
            xp.concatenate(cos_incs, axis=0), xp.concatenate(hit_masks, axis=0),
            xp.concatenate(nearest_idxs, axis=0))


def compute_layover_margin(standoff_m, altitude_m, max_height_m):
    """
    How far outside a building's own footprint you must aim a
    ground-projected ray to still catch its roof (or any elevated facet).

    A ray from the platform through a point at height h_f, extended down
    to the ground (z=0), lands h_f/(h_p - h_f) times the platform's
    ground-range offset further out than the facet itself -- that's
    layover, the same real SAR phenomenon that displaces tall buildings
    toward the sensor in actual imagery. It's not a quirk of this
    sampling scheme; it's *why* a ground-plane aim grid needs a margin at
    all. Found this by adding roof facets and watching dense SBR miss
    every single one -- ASC (no ray tracing, evaluates each facet's
    closed-form visibility directly) picked them all up fine, which is
    what made the blind spot obvious rather than just quietly wrong.

    Uses standoff as the representative ground-range offset (exact for a
    facet at scene center, an underestimate for facets far off-center
    along the platform track -- fine as a padding margin, not meant to
    be a per-facet exact bound).
    """
    if max_height_m >= altitude_m:
        raise ValueError("max_height_m must be below platform altitude")
    return max_height_m / (altitude_m - max_height_m) * standoff_m


def make_aim_grid(xp, footprint_m, n_rays, standoff_m, altitude_m, max_height_m):
    """Ground-plane primary-ray aim grid, padded by the layover margin so
    elevated facets (roofs, upper walls) aren't systematically missed by
    a grid sized to the raw building footprint alone."""
    margin = compute_layover_margin(standoff_m, altitude_m, max_height_m)
    half = footprint_m / 2.0 + margin
    g = np.linspace(-half, half, n_rays)
    gx, gy = np.meshgrid(g, g)
    return xp.asarray(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])), margin


# ----------------------------------------------------------------------
# Ground clutter: bare concrete/terrain background between buildings
# ----------------------------------------------------------------------
def make_ground_clutter(footprint_m, n_points=None, material='concrete', seed=1):
    """
    Sparse random ground scatterers standing in for bare terrain around and
    between buildings. Without this, everything outside a building footprint
    is a literal, exact zero -- real ground never is.

    Amplitude uses materials.diffuse_coefficient(material) as the Rayleigh
    scale, not a made-up number -- concrete's diffuse coefficient (0.03) is
    ~20-30x weaker than a building wall's reflectivity (0.6-1.0), which is
    the right regime: a faint speckle floor under strong discrete building
    returns, not something that competes with them.

    n_points defaults to one point per 10m x 10m cell (footprint_m/10)^2 --
    plenty for background texture, cheap because it's O(points) per pulse,
    not O(rays x points) like the ray-traced facets.

    Deliberately point-scatterer, not ray-traced: isotropic, no facet
    normal/visibility test, evaluated identically for both the dense-SBR
    and ASC-cached branches (see ground_clutter_phase). That means it adds
    zero occlusion/shadowing behavior and contributes an IDENTICAL term to
    both images, so it doesn't touch the SBR-vs-ASC comparison itself --
    it only adds shared background texture. A ray-traced, shadow-casting
    ground layer (buildings actually blocking clutter behind them) would
    be a legitimate, more expensive follow-on if that's what's wanted
    instead -- ask, don't assume.

    Random (not grid) placement: a periodic tile grid aliases into
    grating-lobe ridges under coherent SAR processing (same reasoning as
    point_scatterer_demo.make_target_scatterers).
    """
    if n_points is None:
        n_points = max(50, int((footprint_m / 10.0) ** 2))
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-footprint_m / 2, footprint_m / 2, size=(n_points, 2))
    z = np.zeros(n_points)
    base = diffuse_coefficient(material)
    amp = rng.rayleigh(base, size=n_points)
    return np.column_stack([xy, z, amp]).astype(np.float64)   # (N,4): x,y,z,amp


def ground_clutter_phase(xp, clutter_pts, plat, freqs, ref_pos):
    """
    Closed-form phase-history contribution from isotropic ground clutter
    points -- same math as point_scatterer_demo.phase_history, no ray
    casting, no visibility/occlusion test.
    """
    pos = xp.asarray(clutter_pts[:, :3])
    amp = xp.asarray(clutter_pts[:, 3])
    n_pulses = plat.shape[0]
    K = freqs.shape[0]
    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    for p in range(n_pulses):
        o = plat[p]
        R_ref = xp.linalg.norm(o - ref_pos)
        R = xp.linalg.norm(pos - o[None, :], axis=1)
        dR = R - R_ref
        phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)
        s[p, :] = (amp[None, :] * phase).sum(axis=1)
    return s


def run_dense_sbr(xp, on_gpu, footprint_m=1000.0, density_per_km2=200.0,
                   n_rays=100, n_pulses=200, n_freq=64, standoff=8000.0,
                   altitude=3000.0, fc=10e9, bandwidth=600e6, seed=0):
    facets = make_building_scene(xp, footprint_m, density_per_km2, seed=seed)
    F = facets['n_facets']

    # synthetic aperture (straight-line platform track, same convention as
    # point_scatterer_demo.py's synth_aperture)
    squint_len = 400.0
    u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
    plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])
    plat = xp.asarray(plat)

    # primary-ray aim grid: n_rays x n_rays points at z=0, padded for
    # layover so elevated facets (roofs) actually get hit -- see
    # compute_layover_margin. max_height_m matches make_building_scene's
    # default max_wh[2]=40.0; pass a matching value if you change that.
    aim_pts, _margin = make_aim_grid(xp, footprint_m, n_rays, standoff, altitude, max_height_m=40.0)
    R = aim_pts.shape[0]

    freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
    K = n_freq

    ref_pos = xp.zeros(3)

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    t_intersect_total = 0.0
    t_phase_total = 0.0

    for p in range(n_pulses):
        o = plat[p]
        d = aim_pts - o[None, :]
        d = d / xp.linalg.norm(d, axis=1, keepdims=True)

        t0 = time.perf_counter()
        hit_pt, hit_amp, cos_inc, hit_mask, _nearest_idx = ray_facet_intersect(xp, o, d, facets)
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        t_intersect_total += (t1 - t0)

        R_ref = xp.linalg.norm(o - ref_pos)
        R_hit = xp.linalg.norm(hit_pt - o[None, :], axis=1)
        dR = xp.where(hit_mask, R_hit - R_ref, 0.0)
        # NOTE: this sums one term per RAY, not per facet -- with N rays
        # landing on the same facet it over-counts that facet ~N-fold and
        # the total is not ray-density-invariant. Fine here since this
        # function is timing-only (ms/pulse for the intersection+phase
        # loop), but NOT physically valid for amplitude/image comparisons
        # -- see sbr_vs_asc_compare.run_dense_sbr_timed for the
        # per-facet-deduplicated version used in the Tier 2 validation.
        amp_eff = xp.where(hit_mask, hit_amp * cos_inc, 0.0)   # crude Lambertian-ish falloff

        # (K,) <- sum over R hit contributions, phase per frequency
        phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)   # (K,R)
        s[p, :] = (amp_eff[None, :] * phase).sum(axis=1)
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()
        t_phase_total += (t2 - t1)

    return dict(
        n_facets=F, n_buildings=facets['n_buildings'], n_rays=R, n_pulses=n_pulses,
        n_freq=K, footprint_m=footprint_m,
        t_intersect_s=t_intersect_total, t_phase_s=t_phase_total,
        t_total_s=t_intersect_total + t_phase_total,
        t_per_pulse_ms=(t_intersect_total + t_phase_total) / n_pulses * 1000.0,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true', help='use cupy instead of numpy')
    ap.add_argument('--footprint', type=float, default=1000.0, help='scene side length, m')
    ap.add_argument('--density', type=float, default=200.0, help='structures per km^2')
    ap.add_argument('--rays', type=int, default=60, help='ray grid side (rays = rays^2 per pulse)')
    ap.add_argument('--pulses', type=int, default=50, help='number of pulses')
    ap.add_argument('--freq', type=int, default=64, help='frequency samples per pulse')
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")
    print(f"Footprint: {args.footprint}m x {args.footprint}m, density {args.density}/km^2")

    result = run_dense_sbr(xp, on_gpu, footprint_m=args.footprint, density_per_km2=args.density,
                            n_rays=args.rays, n_pulses=args.pulses, n_freq=args.freq)
    print(json.dumps(result, indent=2))

    out_path = 'dense_sbr_timing_gpu.json' if on_gpu else 'dense_sbr_timing_cpu.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")
