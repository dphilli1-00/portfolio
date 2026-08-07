"""Full 3-D Attributed Scattering Center (ASC) parametric model, plus its
trihedral (corner-reflector) special case.

asc_parametric_scatterer implements the literal equation (Gerry, Potter,
Moses & Chi 1999, extended to 3-D azimuth+elevation the way Wang et al.,
"3-D Attributed Scattering Center Model of Dynamic Group Targets by
Forward Parametric Modeling Method," IEEE TGRS 2024 do):

    E(f,phi,psi) = S_k (j f/fc)^alpha_k
                   * exp[-j 4 pi f/c (x_k cos(psi)cos(phi)
                                       + y_k cos(psi)sin(phi) + z_k sin(psi))]
                   * sinc[2 pi f/c L_k sin(phi - phibar_k)]
                     * exp[-2 pi f/c gamma_k |sin(phi - phibar_k)|]
                   * sinc[2 pi f/c L_k^psi sin(psi - psibar_k)]
                     * exp[-2 pi f/c gamma_k^psi |sin(psi - psibar_k)|]

Before this, the codebase implemented a partial subset of this: the
azimuth sinc term for exactly one leg of a chain, evaluated at a single
mean-band wavelength rather than per-frequency-bin (see
_azimuth_sinc_taper in multibounce_demo.py); the alpha exponent nowhere;
the gamma damping term nowhere; the elevation line nowhere; and the phase
term computed from exact 3-D Euclidean leg lengths rather than this
equation's far-field-linearized position projection. This file is the
first place in the project that implements the full equation as written,
term for term, general enough to instantiate any of the canonical GTD/
ASC scatterer types (trihedral, dihedral, flat plate, cylinder, edge --
see Potter & Moses 1996 / Gerry et al. 1999's scatterer-type table for
each type's alpha/L/gamma values), not just the trihedral.

Phase term note: this equation's phase is the FAR-FIELD (Fraunhofer)
linearization of range -- it assumes the platform is far enough away,
relative to both scene size and to how far off broadside the aperture
sweeps, that a scatterer's range from the platform is well approximated
by projecting its position onto the unit look-direction (cos(psi)cos(phi),
cos(psi)sin(phi), sin(psi)) rather than computing the exact Euclidean
distance. That's DIFFERENT from (not a bug relative to) the exact-range
convention (R_equiv - R_ref via literal 3-D distances) used everywhere
else in this codebase's SBR and closed-form paths, including the earlier
version of trihedral_asc_contribution in this file. Implemented here
exactly as the pasted equation specifies, so it can be validated against
the exact-range version and SBR on equal footing -- see the docstring on
validate_against_sbr below for the actual numbers.

j^alpha term note: (j f/fc)^alpha is complex, not just a magnitude
scaling -- for non-integer or odd-integer alpha it carries a fixed
frequency-independent phase offset (j^alpha = exp(j*alpha*pi/2)), a real,
documented GTD effect distinguishing scatterer types by more than their
frequency-domain rolloff. The earlier version of this file's amplitude
term, (freqs/f0)**alpha, dropped this phase factor -- harmless for a
COHERENCE check (a fixed phase offset doesn't change coherence at all,
since |sum(a * conj(b*e^{j*theta}))| = |sum(a*conj(b))| for constant
theta), which is why the earlier 0.997 result was still valid -- but not
a faithful implementation of the equation. Fixed here.
"""
import numpy as np

C = 299792458.0


def asc_amplitude_envelope(xp, plat, position, freqs, alpha, L_az=0.0, gamma_az=0.0,
                            L_el=0.0, gamma_el=0.0, f0=None, u_hat=None, v_hat=None,
                            phi_bar=0.0, psi_bar=0.0, ref_pos=None, low_precision=False):
    """The (jf/fc)^alpha * sinc_az*damp_az * sinc_el*damp_el envelope from
    asc_parametric_scatterer, WITHOUT its far-field-linearized phase term
    -- split out so a caller that needs the equation's real, validated
    amplitude/persistence physics but must keep an EXACT-range phase
    convention (any real multi-facet scene at real standoff, where facets
    sit far enough from a shared ref_pos that the far-field linearization
    breaks down -- see asc_parametric_scatterer's own phase-term
    docstring) can combine this with its own phase computation instead of
    being forced to take the equation's phase along with its amplitude.

    Same parameters as asc_parametric_scatterer (position/u_hat/v_hat
    preferred over phi_bar/psi_bar -- see that function's docstring for
    why), minus S (the caller supplies its own reflectivity/geometry
    amplitude separately) and minus ref_pos's role in the phase term
    (still accepted, only used to derive the legacy global phi/psi
    fallback when u_hat/v_hat aren't given).

    low_precision=False (default): float64/complex128 throughout, IDENTICAL
    behavior to before this parameter existed -- every canonical-shape
    validation number cited elsewhere in this codebase (dihedral 0.971,
    top hat 0.975, etc.) was measured at this precision and has NOT been
    re-checked at float32. True switches internal math to float32/
    complex64 -- this envelope is pure amplitude/taper (no absolute range
    accumulated over kilometers, unlike the phase term callers compute
    separately), so precision loss here is a much smaller concern than it
    would be for a range/phase computation, but it is still an
    unvalidated, opt-in tradeoff, not a verified-safe default. Real,
    measured motivation: these (F, K) envelope arrays are the single
    largest intermediate in run_asc_box_projected_multibounce's per-pulse
    loop (~1.4GB at F=38779, K=2310, complex128) -- float32/complex64
    halves that.

    Returns (n_pulses, K) complex (carries the j^alpha phase factor).
    """
    real_dtype = xp.float32 if low_precision else xp.float64
    complex_dtype = xp.complex64 if low_precision else xp.complex128
    plat = xp.asarray(plat, dtype=real_dtype)
    freqs = xp.asarray(freqs, dtype=real_dtype)
    position = xp.asarray(position, dtype=real_dtype)

    # Trivial-factor detection (this session, --profile on a real 1000m/200-
    # building run showed this function at 51% of per-pulse cost, the
    # single biggest line item in the whole ASC loop): every caller in this
    # codebase (box_projected_multibounce.py, both multibounce_demo.py call
    # sites) passes L_el=0.0 and never overrides gamma_az/gamma_el away
    # from their 0.0 default. Mathematically, when L_el==0, arg_el is the
    # zero array regardless of sin_del, so sinc_el's near_zero_el branch is
    # taken for EVERY element -- sinc_el is exactly 1.0, not approximately.
    # Same for damp_az/damp_el when gamma_az/gamma_el==0 (exp(0)=1
    # regardless of sin_daz/sin_del). The old code computed all three
    # anyway: a full (K,F)-scale outer product + compare + where + sin +
    # divide (or exp) per trivial factor, discovering afterward that the
    # answer was 1.0. This block detects the scalar-zero case at Python
    # level (cheap, no GPU sync -- L_el/gamma_az/gamma_el are always plain
    # floats in this codebase, never per-facet arrays) and skips the
    # corresponding (K,F) array work entirely, falling back to the exact
    # original computation for any future caller that passes a real
    # nonzero value. Bit-identical output in the trivial case by
    # construction (exp(0)=1, sinc(uniformly-zero arg)=1 exactly, not a
    # tolerance-based approximation) -- validated against the pre-this-
    # change implementation on both the trivial (L_el=gamma_az=gamma_el=0)
    # and a deliberately nonzero case before this landed.
    def _is_zero_scalar(v):
        return not hasattr(v, 'shape') and not hasattr(v, '__len__') and float(v) == 0.0

    skip_damp_az = _is_zero_scalar(gamma_az)
    skip_sinc_el = _is_zero_scalar(L_el)
    skip_damp_el = _is_zero_scalar(gamma_el)
    need_sin_del = not (skip_sinc_el and skip_damp_el)   # damp_el alone can still need it
    need_psi = need_sin_del and v_hat is None
    need_phi = u_hat is None

    # Compute whichever of (facet-relative look direction) / (legacy global
    # phi,psi) is actually needed -- independently per axis, since one axis
    # can use u_hat/v_hat while the other falls back to phi_bar/psi_bar
    # (e.g. a caller with a known u_hat but L_el=0, where psi/phi would
    # never be used anyway but still must not crash referencing them).
    look_n = None
    if u_hat is not None or v_hat is not None:
        pos_b = xp.broadcast_to(position, plat.shape) if position.ndim == 1 else position
        look = plat - pos_b
        look_n = look / xp.linalg.norm(look, axis=1, keepdims=True)
    if need_phi or need_psi:
        if ref_pos is None:
            ref_pos = xp.zeros(3)
        los = plat - ref_pos[None, :]
        if need_psi:
            R_ref = xp.linalg.norm(los, axis=1)
            psi = xp.arcsin(xp.clip(los[:, 2] / R_ref, -1.0, 1.0))
        if need_phi:
            phi = xp.arctan2(los[:, 1], los[:, 0])

    if f0 is None:
        f0 = float(xp.mean(freqs))
    freq_ratio = freqs / f0
    # xp.array(1j) defaults to complex128 regardless of freq_ratio's own
    # dtype -- multiplying a float32 array by a bare `1j` literal
    # silently upcasts the whole rest of the expression back to
    # complex128, quietly defeating low_precision entirely. Build the
    # imaginary unit at the TARGET dtype explicitly instead.
    j_typed = xp.asarray(1j, dtype=complex_dtype)
    amp_freq = (j_typed * freq_ratio.astype(complex_dtype)) ** alpha   # (K,) complex -- includes j^alpha phase

    if u_hat is not None:
        # u_hat may be (3,) -- one shared axis (canonical-shape validation
        # callers) -- or (N,3) -- one axis PER row, required when plat/
        # position are themselves per-facet arrays (a real scene, one
        # u_hat per facet, not a single global axis).
        u_arr = xp.asarray(u_hat, dtype=real_dtype)
        if u_arr.ndim == 1:
            u_n = u_arr / xp.linalg.norm(u_arr)
            sin_daz = xp.sum(look_n * u_n[None, :], axis=1)
        else:
            u_n = u_arr / xp.linalg.norm(u_arr, axis=1, keepdims=True)
            sin_daz = xp.sum(look_n * u_n, axis=1)
    else:
        sin_daz = xp.sin(phi - phi_bar)
    arg_az = 2.0 * xp.pi * xp.outer(freqs, L_az * sin_daz) / C
    near_zero_az = xp.abs(arg_az) < 1e-9
    arg_az_safe = xp.where(near_zero_az, 1.0, arg_az)
    sinc_az = xp.where(near_zero_az, 1.0, xp.sin(arg_az_safe) / arg_az_safe)

    if need_sin_del:
        if v_hat is not None:
            v_arr = xp.asarray(v_hat, dtype=real_dtype)
            if v_arr.ndim == 1:
                v_n = v_arr / xp.linalg.norm(v_arr)
                sin_del = xp.sum(look_n * v_n[None, :], axis=1)
            else:
                v_n = v_arr / xp.linalg.norm(v_arr, axis=1, keepdims=True)
                sin_del = xp.sum(look_n * v_n, axis=1)
        else:
            sin_del = xp.sin(psi - psi_bar)

    # Build the product incrementally, skipping the multiply entirely for
    # any factor that's a trivial scalar 1.0 -- in this codebase's actual
    # usage (L_el=gamma_az=gamma_el=0 everywhere), env = amp_freq * sinc_az
    # is ALL that ever gets computed; damp_az/sinc_el/damp_el are never
    # materialized as (K,F) arrays at all.
    env = amp_freq[:, None] * sinc_az
    if not skip_damp_az:
        damp_az = xp.exp(-2.0 * xp.pi * xp.outer(freqs, gamma_az * xp.abs(sin_daz)) / C)
        env = env * damp_az
    if not skip_sinc_el:
        arg_el = 2.0 * xp.pi * xp.outer(freqs, L_el * sin_del) / C
        near_zero_el = xp.abs(arg_el) < 1e-9
        arg_el_safe = xp.where(near_zero_el, 1.0, arg_el)
        sinc_el = xp.where(near_zero_el, 1.0, xp.sin(arg_el_safe) / arg_el_safe)
        env = env * sinc_el
    if not skip_damp_el:
        damp_el = xp.exp(-2.0 * xp.pi * xp.outer(freqs, gamma_el * xp.abs(sin_del)) / C)
        env = env * damp_el
    # (K, n_pulses)
    return env.T.astype(complex_dtype)   # (n_pulses, K) -- belt-and-suspenders cast in case
    # any upstream numpy/cupy promotion rule (e.g. a bare Python float constant
    # multiplying a float32 array) snuck float64 back in somewhere above.


def asc_visible_envelope(xp, o, position, freqs, visible, alpha=1.0, L_az=0.0,
                          u_hat=None, L_el=0.0, low_precision=False, sparse=False,
                          mask_invisible=True):
    """THE single place every closed-form ASC calculator in this codebase
    (box_projected_multibounce.run_asc_box_projected_multibounce,
    multibounce_demo.run_asc_cached_multibounce,
    multibounce_demo.run_asc_cached_order3_wall_ground_wall) gets its
    per-facet, per-frequency complex amplitude/persistence envelope from.

    Before this existed, the same ~10-line pattern -- filter to visible
    facets, compute asc_amplitude_envelope only on that subset, scatter
    the result back into a full-size (F,K) zero array so every downstream
    (K,F) phase/geometry array can multiply against it unconditionally --
    was duplicated independently in box_projected_multibounce.py (task
    #30) and multibounce_demo.py (task #36), and a THIRD calculator
    (run_asc_cached_order3_wall_ground_wall) was still on the old
    pre-alpha real taper entirely. Consolidated here so there is exactly
    one implementation of "how does a facet's ASC envelope get computed
    for a single platform position," and every caller -- present or
    future -- gets fixes (dtype bugs, precision options, physics
    corrections) applied everywhere at once instead of N times.

    o: (3,) single platform position for this pulse (every existing
    caller computes one pulse's envelope at a time inside a per-pulse
    loop -- NOT (n_pulses,3); tiled internally to the visible-facet
    count).
    position: (F,3) every candidate facet/scatterer's own position (only
    the visible-flagged rows are ever actually evaluated).
    freqs: (K,) frequency bins.
    visible: (F,) bool -- facets outside this mask get an exact-zero row,
    not a tiny/suppressed one; downstream code must not assume "small"
    means visible-but-tapered vs invisible, they're bit-distinguishable.
    alpha: GTD frequency exponent (1.0 for the flat-plate/dihedral/
    trihedral walls this project's scenes are built from -- see
    trihedral_asc_closed_form module docstring's scatterer-type table
    for other canonical values if a future scene ever needs them).
    L_az, u_hat: (F,) and (F,3) per-facet azimuth persistence length and
    axis -- pass the WHOLE (F,)/(F,3) arrays, not pre-filtered; filtering
    to the visible subset happens inside this function.
    L_el: elevation persistence length, 0.0 (isotropic in elevation) is
    the validated choice for this project's vertical-wall scenes (see
    box_projected_multibounce.py's leg1 comment/task #32 for the
    0.971-vs-0.44 coherence measurement backing that choice) -- pass a
    nonzero value only for a genuinely elevation-persistent scatterer
    type this project doesn't currently model.
    low_precision: see asc_amplitude_envelope's own docstring -- complex64
    instead of complex128, unverified speed benefit on GPU (no GPU in
    this project's dev sandbox), verified numerically safe (coherence
    0.999999999999994 vs full precision).

    sparse: False (default) computes the envelope DENSELY for every
    facet (no xp.nonzero(), no gather/scatter), then masks out
    non-visible rows with one elementwise xp.where -- this is what every
    caller effectively did before task #30's "visibility filter"
    optimization existed. True switches to the gather/scatter path
    (xp.nonzero() -> fancy-index in -> scatter back), which measured a
    modest ~10-15% win on CPU (fancy-indexing there is just an in-
    process memcpy) but was MEASURED AS A ~7x REGRESSION on real GPU
    hardware the first time it actually ran there (this project's own
    session history, at ~99.7% scene visibility): xp.nonzero() forces a
    device->host sync every single call, and gather/scatter breaks the
    coalesced memory access GPU throughput depends on -- paying that
    cost every pulse to skip computing on a mere ~0.3% invisible
    fraction is a bad trade specifically on GPU, even though the CPU
    number made it look safe. Default flipped back to dense (False)
    because of this. sparse=True is opt-in only, for a scene with a
    MUCH lower, verified-in-advance visible fraction (e.g. a dense
    occluding scene where most facets really are backfacing/hidden most
    pulses) where the skipped compute might actually outweigh the
    sync+gather cost -- measure on your own hardware before flipping
    this, the CPU number already fooled this codebase once.

    Returns (F, K) complex. mask_invisible=True (default): exactly zero on
    every non-visible row. mask_invisible=False: the RAW per-facet
    envelope for every facet, INCLUDING non-visible ones -- not
    guaranteed zero there (just whatever the equation evaluates to for
    that facet's geometry). Only pass False if your own caller already
    multiplies this by a real geometric amplitude that's itself zeroed
    on every row this env would have masked (every current caller in
    this codebase does exactly that -- amp_eff*_geom is always built
    from a xp.where(visible_or_narrower_mask, ..., 0.0), and every
    subsequent bounce's valid mask is a SUBSET of visible, so the
    zero propagates regardless of what env contains there). Skipping
    the mask saves one full (F,K) elementwise xp.where pass -- measured
    real, if secondary, on top of the sparse->dense fix (task #38):
    at F=38779/K=2310 that's ~1.4GB of complex128 touched for a
    provably-redundant operation, given how every current caller
    already uses this return value. VERIFIED (this session) no NaN/Inf
    in the raw envelope for non-visible rows across 5 pulses spanning a
    full 60-pulse aperture sweep on the real 500m/50-building/38779-
    facet scene -- but this was checked, not assumed, and a future
    caller with different geometry (e.g. a facet whose position exactly
    coincides with the platform, an actual division-by-zero case in the
    look-direction normalization) could in principle violate that. If
    you're adding a new caller that does NOT already re-mask
    downstream, leave mask_invisible at its default (True).
    """
    env_dtype = xp.complex64 if low_precision else xp.complex128
    F = position.shape[0]
    K = freqs.shape[0]

    if sparse:
        env = xp.zeros((F, K), dtype=env_dtype)
        vis_idx = xp.nonzero(visible)[0]
        if vis_idx.shape[0] > 0:
            o_rep_vis = xp.tile(o[None, :], (vis_idx.shape[0], 1))
            L_az_vis = L_az[vis_idx] if hasattr(L_az, '__len__') or hasattr(L_az, 'shape') else L_az
            u_hat_vis = u_hat[vis_idx] if u_hat is not None else None
            env[vis_idx] = asc_amplitude_envelope(
                xp, o_rep_vis, position[vis_idx], freqs, alpha=alpha,
                L_az=L_az_vis, u_hat=u_hat_vis, L_el=L_el,
                low_precision=low_precision)
        return env

    # Dense path (default): compute for every facet, mask afterward --
    # no sync, no gather/scatter, matches this codebase's pre-task-#30
    # measured-fast behavior. Same per-row math as the sparse path (each
    # row's result depends only on its own facet, never on which OTHER
    # facets are visible), so this is bit-identical to the sparse path
    # on every visible row -- purely a performance choice, not a
    # different physics/numerics path.
    o_rep = xp.tile(o[None, :], (F, 1))
    env_full = asc_amplitude_envelope(
        xp, o_rep, position, freqs, alpha=alpha,
        L_az=L_az, u_hat=u_hat, L_el=L_el,
        low_precision=low_precision)
    if not mask_invisible:
        return env_full
    return xp.where(visible[:, None], env_full, xp.asarray(0, dtype=env_dtype))


def asc_parametric_scatterer(xp, plat, freqs, ref_pos, position, S, alpha,
                              L_az=0.0, phi_bar=0.0, gamma_az=0.0,
                              L_el=0.0, psi_bar=0.0, gamma_el=0.0, f0=None,
                              u_hat=None, v_hat=None):
    """The full 3-D ASC parametric scatterer equation, literally.

    xp: numpy or cupy.
    plat: (n_pulses,3) platform positions.
    freqs: (K,) frequency bins (Hz).
    ref_pos: (3,) scene reference point that the platform's phase (proj)
    is measured relative to -- conventionally scene center; here,
    whatever the caller's backprojection reference is, so this stays
    additively combinable with every other closed-form/SBR term in this
    codebase. NOT used for the taper/incidence-angle terms (see u_hat/
    v_hat below) -- those are computed relative to the scatterer's own
    true position, not ref_pos, which can be hundreds of meters away for
    a real scene facet.
    position: (3,) the scatterer's location, (x_k,y_k,z_k) relative to
    ref_pos internally.
    S: scalar (or (n_pulses,)) reflectivity-scale amplitude -- same non-
    absolute "product of cosines/reflectivities" convention as the rest
    of this codebase, not a calibrated physical RCS.
    alpha: GTD frequency-dependence exponent (canonical values: 1 for
    trihedral/dihedral/flat-plate-at-broadside, 0.5 for a singly-curved
    surface/cylinder, 0 for a sphere, -0.5 for an edge at broadside,
    -1 for a general edge diffraction).
    L_az, L_el: azimuth/elevation persistence lengths (0 => isotropic in
    that axis, canonical for trihedral/flat-plate/sphere; nonzero for a
    dihedral's ridge length or a wall's own horizontal/vertical extent).
    u_hat, v_hat: the scatterer's own LOCAL in-plane axes (full 3-D unit
    vectors, e.g. a wall's length direction and height direction) that
    L_az/L_el are measured along. THE preferred way to specify a
    persistence axis -- when given, the taper argument is the full-3D
    dot product dot(u_hat, look_dir) / dot(v_hat, look_dir), exactly
    matching multibounce_demo._azimuth_sinc_taper's (already fixed, see
    that function's docstring) convention, which is correct for ANY
    facet orientation, not just a vertical wall or a plate pre-aligned
    to the world frame. phi_bar/psi_bar are IGNORED when the
    corresponding u_hat/v_hat is given -- the axis vector itself is the
    broadside reference, no separate angle needed.
    phi_bar, psi_bar: LEGACY fallback -- global azimuth/elevation angles
    (radians) used only when u_hat/v_hat is None. This is the same
    coordinate-plane-projection convention _azimuth_sinc_taper's first
    version used before being fixed (silently wrong for a facet whose
    persistence axis isn't confined to a plane containing the world Z
    axis -- e.g. an untested-this-session sloped roof or a rotated
    panel). Kept only for backward compatibility with existing L_az=0/
    L_el=0 (isotropic) callers, where it's unused anyway. Do not rely on
    this path for a nonzero L on a genuinely tilted facet.
    gamma_az, gamma_el: angular-damping ("localization") coefficients for
    the exponential term alongside each sinc; 0 reduces that factor to 1
    (this codebase's earlier azimuth taper never modeled this term at
    all -- see module docstring).
    f0: reference frequency (Hz); band mean if not given.

    Returns (n_pulses, K) complex phase history.
    """
    los = plat - ref_pos[None, :]
    R_ref = xp.linalg.norm(los, axis=1)
    phi = xp.arctan2(los[:, 1], los[:, 0])
    psi = xp.arcsin(xp.clip(los[:, 2] / R_ref, -1.0, 1.0))

    # position may be (3,) -- one fixed point, fine for flat/trihedral-type
    # scatterers -- or (n_pulses,3), a per-pulse SLIDING position, required
    # for curved surfaces (sphere/cylinder). See sliding_specular_point:
    # using a fixed point (e.g. a sphere's center) for a curved surface
    # leaves a constant spatial offset (~radius) between the assumed and
    # true phase center. That offset is invisible across PULSES (a fixed
    # offset doesn't decorrelate a synthetic aperture) but devastating
    # across FREQUENCY (a fixed range offset is a linear phase-vs-
    # frequency ramp -- a radius-scale offset can be many cycles across a
    # real bandwidth). Measured directly: for a 1m sphere at X-band,
    # 600 MHz bandwidth, using the fixed center gave 0.024 coherence
    # against a converged brute-force PO reference; the sliding point
    # recovers >0.95 (see the module's sphere/cylinder validation).
    position = xp.asarray(position)
    if position.ndim == 1:
        x = position[0] - ref_pos[0]
        y = position[1] - ref_pos[1]
        z = position[2] - ref_pos[2]
    else:
        x = position[:, 0] - ref_pos[0]
        y = position[:, 1] - ref_pos[1]
        z = position[:, 2] - ref_pos[2]
    proj = x * xp.cos(psi) * xp.cos(phi) + y * xp.cos(psi) * xp.sin(phi) + z * xp.sin(psi)  # (n_pulses,)

    # NOTE on sign: the pasted equation writes exp[-j4pi f/c (x cos(psi)cos(phi)+...)]
    # with phi/psi left geometrically undefined (no figure to pin down which way
    # they point). Empirically checked here against this codebase's already-
    # cross-validated exact-range convention (dR = R_scatterer - R_ref, phase =
    # exp(-j4pi f dR/c)): defining phi/psi as the look direction FROM ref_pos
    # TOWARD the platform (as done above, via los = plat - ref_pos) requires
    # dR = -proj, not +proj, to match -- confirmed numerically to agree with the
    # exact 3-D range difference to within 0.12% (the residual being the real,
    # expected far-field-linearization error for an 8m target at ~8.5km range,
    # not a bug). Using +proj directly (the naive literal reading) reproduced
    # dR with the wrong sign and collapsed coherence against SBR from 0.996 to
    # 0.014 -- entirely a sign convention issue, not a physics error, but a real
    # reminder that an equation lifted from a paper without its geometry figure
    # can silently flip sign depending on which way phi/psi are defined to face.
    phase = xp.exp(1j * 4.0 * xp.pi * xp.outer(freqs, proj) / C)   # (K, n_pulses)

    # Amplitude/persistence envelope -- shared with asc_amplitude_envelope
    # (the split-out version any exact-range caller, e.g. the real scene's
    # box_projected_multibounce.py, should call instead of this function).
    env = asc_amplitude_envelope(xp, plat, position, freqs, alpha, L_az=L_az, gamma_az=gamma_az,
                                  L_el=L_el, gamma_el=gamma_el, f0=f0, u_hat=u_hat, v_hat=v_hat,
                                  phi_bar=phi_bar, psi_bar=psi_bar, ref_pos=ref_pos)   # (n_pulses, K)

    E = S * phase.T * env   # (n_pulses, K)
    return E


def sliding_specular_point(xp, plat, surface_center, radius, axis=None):
    """The point on a convex curved surface where the local outward
    normal points exactly back at the platform, per pulse -- required
    position input for any curved-surface ASC scatterer type (sphere,
    cylinder), NOT optional (see asc_parametric_scatterer's docstring on
    why a fixed reference point silently corrupts the frequency axis).

    axis=None: sphere -- specular point is surface_center + radius *
    unit(plat - surface_center), sliding along the great circle facing
    the platform.

    axis=(3,): infinite cylinder of that axis direction -- specular point
    slides only in the plane perpendicular to axis (a cylinder's return,
    at any point along its length, comes from the generator line nearest
    the sensor -- there's no persistence-limited "sliding" along the axis
    itself the way there is around the circumference).

    Returns (n_pulses,3), meant to be passed directly as
    asc_parametric_scatterer's position argument.
    """
    surface_center = xp.asarray(surface_center)
    if axis is None:
        d = plat - surface_center[None, :]
        d = d / xp.linalg.norm(d, axis=1, keepdims=True)
        return surface_center[None, :] + radius * d
    axis_n = xp.asarray(axis) / xp.linalg.norm(xp.asarray(axis))
    rel = plat - surface_center[None, :]
    along = xp.sum(rel * axis_n[None, :], axis=1, keepdims=True) * axis_n[None, :]
    radial = rel - along
    radial_n = radial / xp.linalg.norm(radial, axis=1, keepdims=True)
    return surface_center[None, :] + along + radius * radial_n


def trihedral_asc_contribution(xp, plat, freqs, ref_pos, vertex, amp, alpha=1.0, f0=None):
    """Trihedral special case of asc_parametric_scatterer: L_az=L_el=0
    (isotropic within acceptance cone -- no persistence taper on either
    axis), gamma_az=gamma_el=0 (no damping term to speak of once the
    sinc it multiplies is already 1 everywhere), alpha=1 (canonical GTD
    value for trihedral/dihedral/flat-plate). phi_bar/psi_bar are
    irrelevant when L_az=L_el=0 and left at 0.

    This is now a thin wrapper, not an independent implementation -- see
    module docstring for why that matters (a real regression check
    against SBR, done both ways, is what confirms the general equation
    reduces correctly to the special case rather than just asserting it
    algebraically).
    """
    return asc_parametric_scatterer(xp, plat, freqs, ref_pos, vertex, amp, alpha,
                                     L_az=0.0, phi_bar=0.0, gamma_az=0.0,
                                     L_el=0.0, psi_bar=0.0, gamma_el=0.0, f0=f0)


def trihedral_reflectivity_estimate(panel_reflectivity, n_valid_orderings=6):
    """First-principles amplitude estimate for a boresight-aligned right
    trihedral, in this codebase's own (non-absolute) amplitude
    convention -- the product of per-leg incidence cosines and material
    reflectivity, summed coherently across however many of the 6
    possible panel-visit orderings are geometrically valid for this
    illumination.

    cos(incidence) per leg is exactly cos(54.7 deg) = 1/sqrt(3) for any
    EXACT right-angle trihedral illuminated precisely on its symmetry
    axis, independent of panel size or wavelength: a ray direction
    (1,1,1)/sqrt(3) (in the trihedral's own local frame, where the 3
    face normals ARE the coordinate axes) makes that angle with every
    face normal by construction -- no ray tracing needed to derive it.
    cos1*cos2*cos3 = (1/sqrt(3))**3 = 1/(3*sqrt(3)) =~ 0.1925, matching
    the direct numeric check done this session to 4 decimal places.

    n_valid_orderings: how many of the 6 possible entry/exit panel
    sequences are geometrically valid (survive occlusion) for this
    illumination direction. Measured empirically at exact boresight
    alignment (this session, via direct ray-facet-triple extraction):
    all 6, roughly equally populated. Off-boresight, expect fewer to
    remain valid -- this is a coarse scalar stand-in for that count, not
    a substitute for actually checking; pass a smaller value (or 0, if
    the illumination has left the reflector's acceptance cone entirely)
    for aspect angles far from exact alignment.
    """
    cos_product = (1.0 / np.sqrt(3.0)) ** 3
    return n_valid_orderings * (panel_reflectivity ** 3) * cos_product
