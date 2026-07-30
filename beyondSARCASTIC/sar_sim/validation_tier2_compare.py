"""
Validation slide, Tier 2, actually run: form a chip from dense-SBR and from
ASC-cached phase history (SAME scene, SAME aperture, SAME noise realization
-- here, none, both are noiseless so we're isolating compression fidelity,
not fighting speckle), then score with SSIM (skimage, Wang et al. 2004) and
amplitude/phase RMS error, per the deck's Validation slide criteria:
SSIM >= 0.95 typical pass, both should degrade together outside the fitted
sector.

Reuses the same facets and forward models as sbr_vs_asc_compare.py --
importing rather than duplicating so there's exactly one definition of
"the scene" and "the two forward models" anywhere in this codebase.

Usage:
    python3 validation_tier2_compare.py --gpu --footprint 1000 --density 200 \
        --rays 100 --pulses 150 --freq 96 --img-size 256
"""

import argparse
import time
import json
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim

from dense_sbr_demo import (make_building_scene, get_backend, C, make_aim_grid,
                             make_ground_clutter, ground_clutter_phase,
                             compute_layover_margin)
from sbr_vs_asc_compare import run_dense_sbr_timed, run_asc_cached


def backproject(xp, s, plat_pos, freqs, grid_x, grid_y, fc, R_ref):
    """Time-domain backprojection, backend-agnostic (numpy or cupy).
    Same algorithm as point_scatterer_demo.backproject."""
    P, K = s.shape
    B = freqs[-1] - freqs[0]
    range_res = C / (2 * B)

    range_win = xp.asarray(np.hamming(K))
    az_win = xp.asarray(np.hamming(P))
    range_profile = xp.fft.fftshift(xp.fft.ifft(s * range_win[None, :], axis=1), axes=1)
    range_axis = (xp.arange(K) - K // 2) * range_res

    Xg, Yg = xp.meshgrid(grid_x, grid_y)
    img = xp.zeros(Xg.shape, dtype=xp.complex128)

    range_axis_np = to_numpy(range_axis)
    for p in range(P):
        dx = Xg - plat_pos[p, 0]
        dy = Yg - plat_pos[p, 1]
        R_pixel_abs = xp.sqrt(dx**2 + dy**2 + plat_pos[p, 2]**2)
        dR_pixel = R_pixel_abs - R_ref[p]
        rp = range_profile[p, :]
        dR_np = to_numpy(dR_pixel)
        rp_np = to_numpy(rp)
        samp_np = np.interp(dR_np, range_axis_np, rp_np.real) \
            + 1j * np.interp(dR_np, range_axis_np, rp_np.imag)
        samp = xp.asarray(samp_np)
        img = img + az_win[p] * samp * xp.exp(1j * 4.0 * xp.pi * fc * dR_pixel / C)

    return img


def to_numpy(a):
    """Pull a cupy or numpy array back to numpy (interp/imshow need numpy)."""
    if hasattr(a, 'get'):
        return a.get()
    return np.asarray(a)


def compute_ref_ranges(xp, plat, ref_pos):
    return xp.linalg.norm(plat - ref_pos[None, :], axis=1)


def complex_coherence(img_a, img_b):
    """
    Standard InSAR-style complex coherence: |sum(a * conj(b))| / sqrt(sum(|a|^2) * sum(|b|^2)).
    1.0 = perfect agreement (magnitude AND phase both aligned), 0 = totally
    uncorrelated. Unlike phase RMS, it's sensitive to amplitude-weighted
    agreement jointly, not phase alone -- a region where SBR and ASC agree
    on phase but disagree on which pixels are bright still shows up as
    reduced coherence. Same dilution problem as whole-image SSIM applies
    here too: coherence over the whole noiseless scene looks deceptively
    high because real per-building disagreement is diluted by the much
    larger area of near-identical shared clutter background -- compute it
    per-building for the number that matches the deck's "same sector"
    criterion.
    """
    num = np.abs(np.sum(img_a * np.conj(img_b)))
    den = np.sqrt(np.sum(np.abs(img_a) ** 2) * np.sum(np.abs(img_b) ** 2)) + 1e-12
    return float(num / den)


def coherence_map(img_a, img_b, win=7):
    """
    Spatially-resolved local coherence (the actual InSAR-style coherence
    MAP, not just a scalar) -- sliding-window version of complex_coherence,
    win x win pixels per window. Reveals WHERE disagreement lives instead
    of averaging it away into one number.

    Filters the complex cross-product itself (uniform_filter on real/imag
    parts separately, then recombine), NOT the magnitude of the product --
    filtering magnitude first would throw away the phase information that
    makes this coherence rather than just smoothed amplitude agreement.

    Expected structure in this scene: ground clutter is added IDENTICALLY
    to both branches (s_sbr += s_clutter, s_asc += s_clutter in main()),
    so background pixels should read close to coherence=1 almost by
    construction -- real disagreement should show up localized at
    buildings, not spread diffusely across the image. A map that's low
    everywhere (not just at structures) would point at something wrong
    with the shared clutter/backprojection path, not compression fidelity.
    """
    cross = img_a * np.conj(img_b)
    cross_f = uniform_filter(to_numpy(cross.real), win) + 1j * uniform_filter(to_numpy(cross.imag), win)
    num = np.abs(cross_f)
    den = np.sqrt(uniform_filter(np.abs(to_numpy(img_a)) ** 2, win) * uniform_filter(np.abs(to_numpy(img_b)) ** 2, win)) + 1e-12
    return num / den


def required_freq_samples(plat_np, ref_np, grid_np, bandwidth, margin_factor=1.15):
    """
    Minimum number of stepped-frequency samples K such that K*range_res
    covers every image pixel's reference-relative range across the whole
    aperture, with a safety margin.

    Why this exists: K*range_res undersized doesn't raise an error, it
    silently clamps np.interp's out-of-window samples to the range
    profile's edge value, which produces exactly the unfocused, striped,
    no-discrete-targets image this function was added after seeing (this
    script's own first run had the same failure at 200m before the freq/
    bandwidth args were hand-tuned to fix it -- that fix should have been
    made automatic then, not re-discovered by hand at 1000m).

    Required window scales with scene footprint (bigger scene -> bigger
    max reference-relative range spread), so a freq/bandwidth pair tuned
    at one footprint silently stops being enough at a larger one -- this
    recomputes it from the actual platform/grid geometry every run
    instead of carrying a fixed assumption forward.
    """
    Xg, Yg = np.meshgrid(grid_np, grid_np)
    R_ref = np.linalg.norm(plat_np - ref_np[None, :], axis=1)
    maxdr = 0.0
    for p in range(plat_np.shape[0]):
        dx = Xg - plat_np[p, 0]
        dy = Yg - plat_np[p, 1]
        R_abs = np.sqrt(dx ** 2 + dy ** 2 + plat_np[p, 2] ** 2)
        dR = R_abs - R_ref[p]
        maxdr = max(maxdr, float(np.abs(dR).max()))
    needed_window = 2.0 * maxdr * margin_factor
    range_res = C / (2.0 * bandwidth)
    return int(np.ceil(needed_window / range_res)), maxdr, needed_window


PHASE_HISTORY_CACHE_PARAMS = [
    'footprint', 'density', 'rays', 'pulses', 'standoff', 'altitude', 'fc', 'bandwidth', 'freq',
]


def phase_history_cache_key(args, scene_seed=0):
    """
    Deterministic key over exactly the parameters that change s_sbr/s_asc
    (the expensive dense-SBR ray-traced phase history and its ASC-cached
    counterpart) -- NOT img-size, --no-clutter, or plotting/scoring
    options, since none of those touch the forward sim at all. Computed
    from `args` AFTER all auto-bump logic has resolved (--pulses via the
    Nyquist guard, --freq via the range-window guard), so a cache built
    from a run that needed auto-bumping is correctly keyed on the
    resolved values, not whatever the user originally typed.
    """
    parts = [f"{p}={getattr(args, p)}" for p in PHASE_HISTORY_CACHE_PARAMS]
    parts.append(f"scene_seed={scene_seed}")
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return digest, raw


def load_phase_history_cache(cache_dir, key):
    path = os.path.join(cache_dir, f"{key}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return dict(
        s_sbr=data['s_sbr'], s_asc=data['s_asc'], plat=data['plat'], freqs=data['freqs'],
        sbr_ms_per_pulse=float(data['sbr_ms_per_pulse']), sbr_n_rays=int(data['sbr_n_rays']),
        asc_ms_per_pulse=float(data['asc_ms_per_pulse']),
    )


def save_phase_history_cache(cache_dir, key, raw_params, s_sbr, s_asc, plat, freqs, sbr_stats, asc_stats):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.npz")
    np.savez(path, s_sbr=s_sbr, s_asc=s_asc, plat=plat, freqs=freqs,
              sbr_ms_per_pulse=sbr_stats['t_per_pulse_ms'], sbr_n_rays=sbr_stats['n_rays'],
              asc_ms_per_pulse=asc_stats['t_per_pulse_ms'])
    with open(os.path.join(cache_dir, f"{key}.params.txt"), 'w') as f:
        f.write(raw_params + "\n")
    return path


def per_building_ssim(facets, db_sbr, db_asc, mag_sbr, mag_asc, phase_sbr, phase_asc,
                       grid_np, standoff_m, altitude_m, sidelobe_margin_m=10.0,
                       min_crop_px=9, data_range=40.0):
    """
    Score SSIM/RMS per building instead of one whole-scene average.

    Matches the deck's own Validation slide language ("per structure type,
    same sector") rather than the whole-image number, which is both
    inflated by any shared background (identical concrete clutter
    contributes "free" agreement to every crop that includes it) and
    diluted by real per-building differences averaging out across a big
    scene -- a single global SSIM can't tell you whether every building
    matches reasonably or a few are badly wrong and the rest are fine.
    This can.

    Crop half-extent per building = footprint half-width + THIS
    building's own layover displacement (height-dependent -- not the
    scene-wide worst-case margin used for the ray-aim grid, which would
    over-pad short buildings) + a fixed margin so SSIM's sliding window
    has spatial context to work with.
    """
    n = facets['n_buildings']
    cx, cy = facets['building_cx'], facets['building_cy']
    w, d, h = facets['building_w'], facets['building_d'], facets['building_h']
    px = len(grid_np)
    peak = mag_sbr.max() + 1e-12

    results = []
    for i in range(n):
        half_fp = max(w[i], d[i]) / 2.0
        layover = compute_layover_margin(standoff_m, altitude_m, h[i])
        half_extent = half_fp + layover + sidelobe_margin_m

        j_lo = max(0, int(np.searchsorted(grid_np, cx[i] - half_extent)))
        j_hi = min(px, int(np.searchsorted(grid_np, cx[i] + half_extent)))
        i_lo = max(0, int(np.searchsorted(grid_np, cy[i] - half_extent)))
        i_hi = min(px, int(np.searchsorted(grid_np, cy[i] + half_extent)))

        if (i_hi - i_lo) < min_crop_px or (j_hi - j_lo) < min_crop_px:
            results.append(dict(building=i, cx=float(cx[i]), cy=float(cy[i]), skipped=True,
                                 reason='crop smaller than min window (near/outside image edge)'))
            continue

        crop_sbr = db_sbr[i_lo:i_hi, j_lo:j_hi]
        crop_asc = db_asc[i_lo:i_hi, j_lo:j_hi]
        s = ssim(crop_sbr, crop_asc, data_range=data_range)

        m_sbr = mag_sbr[i_lo:i_hi, j_lo:j_hi]
        m_asc = mag_asc[i_lo:i_hi, j_lo:j_hi]
        # normalize by the WHOLE-IMAGE peak, not each crop's own peak --
        # per-crop normalization would make every building's amplitude
        # error look artificially small by construction
        amp_rms_b = float(np.sqrt(np.mean((m_sbr / peak - m_asc / peak) ** 2)))

        p_sbr = phase_sbr[i_lo:i_hi, j_lo:j_hi]
        p_asc = phase_asc[i_lo:i_hi, j_lo:j_hi]
        dphase = np.angle(np.exp(1j * (p_sbr - p_asc)))
        wgt = m_sbr / (m_sbr.sum() + 1e-12)
        phase_rms_b = float(np.degrees(np.sqrt(np.sum(wgt * dphase ** 2))))
        # signed bias (not squared) -- distinguishes a consistent offset
        # (structural bug, e.g. mocomp/reference-range convention mismatch
        # between the two branches) from symmetric random scatter, which
        # RMS alone can't tell apart
        phase_bias_b = float(np.degrees(np.sum(wgt * dphase)))
        # max error among SIGNIFICANT pixels only (>5% of this crop's own
        # peak) -- unmasked, this is meaningless: noise-floor pixels have
        # essentially uniform random phase and would dominate a max-error
        # stat with garbage that has nothing to do with compression fidelity
        sig_mask = m_sbr > 0.05 * (m_sbr.max() + 1e-12)
        phase_max_b = float(np.degrees(np.abs(dphase[sig_mask]).max())) if sig_mask.any() else 0.0

        c_sbr = m_sbr * np.exp(1j * p_sbr)
        c_asc = m_asc * np.exp(1j * p_asc)
        coherence_b = complex_coherence(c_sbr, c_asc)

        results.append(dict(building=i, cx=float(cx[i]), cy=float(cy[i]), height_m=float(h[i]),
                             crop_px=[int(i_hi - i_lo), int(j_hi - j_lo)], ssim=float(s),
                             amp_rms=amp_rms_b, phase_rms_deg=phase_rms_b,
                             phase_bias_deg=phase_bias_b, phase_max_deg=phase_max_b,
                             coherence=coherence_b, skipped=False))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=1000.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--rays', type=int, default=80)
    ap.add_argument('--pulses', type=int, default=100)
    ap.add_argument('--freq', type=int, default=64)
    ap.add_argument('--img-size', type=int, default=200)
    ap.add_argument('--standoff', type=float, default=8000.0)
    ap.add_argument('--altitude', type=float, default=3000.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    ap.add_argument('--az-res', type=float, default=None,
                     help='desired cross-range resolution (m). When set, aperture length is '
                          'derived from standoff+wavelength (d_theta=lambda/(2*az_res), '
                          'aperture=d_theta*standoff) instead of a fixed 400m, and --pulses is '
                          'auto-bumped to the Nyquist minimum (P=standoff*lambda/az_res^2, same '
                          'formula as radar_params.py) if too low. Without this, aperture stays '
                          'a fixed 400m regardless of standoff -- fine at short (~8km) standoff, '
                          'but at longer standoff that 400m subtends a much smaller angle, so '
                          'cross-range resolution silently gets much coarser than whatever the '
                          '--pulses count might suggest.')
    ap.add_argument('--no-clutter', action='store_true',
                     help='skip concrete ground-clutter background (buildings only, on black)')
    ap.add_argument('--cache-dir', type=str, default='tier2_cache',
                     help='where cached phase histories (s_sbr/s_asc) are stored')
    ap.add_argument('--no-cache', action='store_true',
                     help='ignore and do not write the phase-history cache -- always recompute '
                          'the expensive dense-SBR/ASC forward sim from scratch')
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

    facets = make_building_scene(xp, args.footprint, args.density, seed=0)
    print(f"{facets['n_buildings']} buildings, {facets['n_facets']} facets")

    wavelength = C / args.fc
    if args.az_res is not None:
        d_theta = wavelength / (2.0 * args.az_res)
        squint_len = d_theta * args.standoff
        min_pulses = int(np.ceil(args.standoff * wavelength / (args.az_res ** 2)))
        edge_deg = (args.footprint / args.standoff) * 180.0 / np.pi
        print(f"--az-res {args.az_res}m -> aperture={squint_len:.0f}m, "
              f"edge-to-edge incidence-angle swing over the scene ~{edge_deg:.2f} deg")
        if args.pulses < min_pulses:
            print(f"--pulses {args.pulses} is below Nyquist for this aperture/resolution "
                  f"(need >={min_pulses} -- azimuth under-sampling doesn't corrupt the SBR-vs-ASC "
                  f"comparison, since both branches share the same aperture/backprojection and "
                  f"any aliasing shows up identically in both and cancels in the difference, but "
                  f"it does add non-physical streaking to each image individually). "
              f"Auto-bumping --pulses {args.pulses} -> {min_pulses}.")
            args.pulses = min_pulses
    else:
        squint_len = 400.0

    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([
        u, np.full(args.pulses, -args.standoff), np.full(args.pulses, args.altitude)
    ]))

    aim_pts, margin = make_aim_grid(xp, args.footprint, args.rays, args.standoff, args.altitude,
                                     max_height_m=40.0)
    if margin > 0.01 * args.footprint:
        print(f"aim grid padded +/-{margin:.1f}m beyond footprint for roof layover "
              f"(grid side {args.footprint + 2*margin:.0f}m)")

    ref_pos = xp.zeros(3)
    grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))

    min_freq, maxdr, needed_window = required_freq_samples(
        to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
    if args.freq < min_freq:
        print(f"--freq {args.freq} gives a range window too small for this scene "
              f"(need >={needed_window:.0f}m to cover max |dR|={maxdr:.0f}m across the "
              f"aperture at {args.bandwidth/1e6:.0f}MHz bandwidth; window undersized silently "
              f"clamps/aliases instead of erroring -- that's what produced striped, "
              f"unfocused images at larger footprints before this check existed). "
              f"Auto-bumping --freq {args.freq} -> {min_freq}.")
        real_bw = 300e6
        if min_freq > 1024 and args.bandwidth < 0.95 * real_bw:
            # NOTE: do not suggest shrinking bandwidth down to whatever hits
            # a small K -- the naive delta_r = c/(2B) formula understates
            # what real hardware actually uses. TerraSAR-X runs up to
            # ~300MHz even for ~2m-class products: ground-range resolution
            # is slant resolution / sin(incidence), and range-compression
            # windowing (Hamming/Taylor, for sidelobe suppression) broadens
            # the mainlobe past the unwindowed ideal -- both push real
            # bandwidth well above the textbook minimum. radar_params.py's
            # own "REALITY CHECK" section computes B_real via that same
            # naive formula (75MHz for 2m) and is short of the real
            # TerraSAR-X spec for the same reason. Only fires below ~300MHz
            # -- if you're already at or above real hardware bandwidth,
            # there's no "cheaper, still-realistic" fix to suggest.
            real_K = int(np.ceil(needed_window / (C / (2.0 * real_bw))))
            print(f"  ({min_freq} freq samples at {args.bandwidth/1e6:.0f}MHz is a lot of "
                  f"FFT/interp work per pulse. Resist the urge to fix that by lowering "
                  f"bandwidth to shrink K -- the naive delta_r=c/(2B) formula understates real "
                  f"hardware; TerraSAR-X runs up to ~300MHz even for ~2m-class products "
                  f"(ground/slant projection + window broadening). At a realistic 300MHz: "
                  f"K~={real_K} needed for this scene/geometry -- that's the real cost, not "
                  f"something to shortcut around. Pass --bandwidth 300e6 to match real hardware "
                  f"instead of the {args.bandwidth/1e6:.0f}MHz default.)")
        elif min_freq > 1024:
            print(f"  ({min_freq} freq samples at {args.bandwidth/1e6:.0f}MHz is real cost, not "
                  f"a shortcut opportunity -- you're already at or above the ~300MHz TerraSAR-X-"
                  f"consistent bandwidth used elsewhere in this project, so there's no 'cheaper "
                  f"but still realistic' bandwidth to fall back to. This is genuinely what it "
                  f"costs to cover this scene's range window at this resolution.)")
        args.freq = min_freq

    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, args.freq))
    R_ref = compute_ref_ranges(xp, plat, ref_pos)

    # Cache key is computed here, AFTER --pulses and --freq have both been
    # auto-bumped to their resolved values above -- keying on the raw CLI
    # input would let two runs that resolve to the identical forward sim
    # (e.g. --freq 64 and --freq 2278, if 2278 is what 64 auto-bumps to
    # for this scene) miss each other's cache.
    cache_key, cache_raw = phase_history_cache_key(args)
    cached = None if args.no_cache else load_phase_history_cache(args.cache_dir, cache_key)

    if cached is not None:
        print(f"\n[1-2/4] Loaded cached phase history ({cache_key}) -- "
              f"skipping dense-SBR/ASC forward sim entirely.")
        print(f"  (cached at {cached['sbr_ms_per_pulse']:.2f} ms/pulse dense SBR, "
              f"{cached['asc_ms_per_pulse']:.2f} ms/pulse ASC, {cached['sbr_n_rays']} rays)")
        s_sbr = xp.asarray(cached['s_sbr'])
        s_asc = xp.asarray(cached['s_asc'])
        sbr_stats = dict(t_per_pulse_ms=cached['sbr_ms_per_pulse'], n_rays=cached['sbr_n_rays'])
        asc_stats = dict(t_per_pulse_ms=cached['asc_ms_per_pulse'])
        # plat/freqs are already built identically above from the same
        # (now-resolved) args that produced this cache key, so no need to
        # overwrite them from the cache file -- they're already correct.
    else:
        print("\n[1/4] Forward sim: dense SBR...")
        s_sbr, sbr_stats = run_dense_sbr_timed(xp, on_gpu, facets, plat, aim_pts, freqs, ref_pos)
        print(f"  {sbr_stats['t_per_pulse_ms']:.2f} ms/pulse")

        print("[2/4] Forward sim: ASC-cached...")
        s_asc, asc_stats = run_asc_cached(xp, on_gpu, facets, plat, freqs, ref_pos)
        print(f"  {asc_stats['t_per_pulse_ms']:.2f} ms/pulse")

        if not args.no_cache:
            path = save_phase_history_cache(
                args.cache_dir, cache_key, cache_raw, to_numpy(s_sbr), to_numpy(s_asc),
                to_numpy(plat), to_numpy(freqs), sbr_stats, asc_stats)
            print(f"  Cached phase history -> {path} (key {cache_key}). Re-running with the "
                  f"same footprint/density/rays/pulses/standoff/altitude/fc/bandwidth/freq will "
                  f"reload this instead of recomputing -- img-size, --no-clutter, and scoring "
                  f"changes are all free to vary without invalidating it.")

    if not args.no_clutter:
        clutter_pts = make_ground_clutter(args.footprint, material='concrete', seed=1)
        print(f"  + {len(clutter_pts)} concrete ground-clutter points "
              f"(materials.diffuse_coefficient('concrete'), shared identically by both branches)")
        s_clutter = ground_clutter_phase(xp, clutter_pts, plat, freqs, ref_pos)
        s_sbr = s_sbr + s_clutter
        s_asc = s_asc + s_clutter

    print("[3/4] Image formation (backprojection), both branches...")
    t0 = time.perf_counter()
    img_sbr = backproject(xp, s_sbr, plat, freqs, grid, grid, args.fc, R_ref)
    img_asc = backproject(xp, s_asc, plat, freqs, grid, grid, args.fc, R_ref)
    print(f"  backprojection: {time.perf_counter()-t0:.2f}s for both images")

    mag_sbr = np.abs(to_numpy(img_sbr))
    mag_asc = np.abs(to_numpy(img_asc))
    phase_sbr = np.angle(to_numpy(img_sbr))
    phase_asc = np.angle(to_numpy(img_asc))

    db_sbr = 20 * np.log10(mag_sbr / (mag_sbr.max() + 1e-12) + 1e-6)
    db_asc = 20 * np.log10(mag_asc / (mag_asc.max() + 1e-12) + 1e-6)

    print("[4/4] Scoring: SSIM + amplitude/phase RMS error...")
    # SSIM on normalized dB imagery (matches how these chips would actually
    # be viewed/compared), data_range spans the clipping window used below
    ssim_score = ssim(db_sbr, db_asc, data_range=40.0)

    # amplitude RMS error, normalized by each image's own peak (compression
    # fidelity, not absolute-calibration fidelity)
    amp_rms = np.sqrt(np.mean((mag_sbr / (mag_sbr.max() + 1e-12) -
                                mag_asc / (mag_asc.max() + 1e-12)) ** 2))
    # phase RMS error, wrapped to [-pi, pi], weighted by SBR magnitude so
    # phase noise in the noise floor doesn't dominate the score
    dphase = np.angle(np.exp(1j * (phase_sbr - phase_asc)))
    w = mag_sbr / (mag_sbr.sum() + 1e-12)
    phase_rms = np.sqrt(np.sum(w * dphase ** 2))
    # signed bias -- a consistent offset (not centered on zero) points at a
    # structural mismatch (e.g. mocomp/reference-range convention) between
    # the two branches, which RMS alone can't distinguish from symmetric
    # random scatter
    phase_bias = np.sum(w * dphase)
    # max error among significant pixels only -- unmasked, noise-floor
    # phase is uniform-random garbage and would dominate a max stat
    sig_mask_whole = mag_sbr > 0.05 * (mag_sbr.max() + 1e-12)
    phase_max = np.abs(dphase[sig_mask_whole]).max() if sig_mask_whole.any() else 0.0
    # complex coherence -- the InSAR-standard joint amplitude+phase
    # agreement metric, 0-1 like SSIM; sensitive to cases where phase
    # matches but the two disagree on which pixels are bright
    coherence = complex_coherence(to_numpy(img_sbr), to_numpy(img_asc))

    pass_ssim = ssim_score >= 0.95
    print(f"\n=== Tier 2 validation result (whole image -- diluted/inflated by shared "
          f"clutter, see per-building below for the real number) ===")
    print(f"SSIM:            {ssim_score:.4f}   ({'PASS' if pass_ssim else 'FAIL'} vs. >= 0.95 criterion)")
    print(f"Amplitude RMS:   {amp_rms:.4f}  (normalized, 0=identical)")
    print(f"Phase RMS:       {np.degrees(phase_rms):.2f} deg (magnitude-weighted)")
    print(f"Phase bias:      {np.degrees(phase_bias):+.2f} deg (signed, magnitude-weighted -- "
          f"far from 0 flags a systematic offset, not just scatter)")
    print(f"Phase max error: {np.degrees(phase_max):.2f} deg (worst pixel among those >5% of peak)")
    print(f"Coherence:       {coherence:.4f}  (InSAR-style |<a,b*>|/sqrt(<|a|^2><|b|^2>), 1=identical)")

    print(f"\n=== Tier 2 validation result (per building -- matches the Validation "
          f"slide's actual 'per structure type, same sector' criterion) ===")
    grid_np = to_numpy(grid)
    per_bldg = per_building_ssim(facets, db_sbr, db_asc, mag_sbr, mag_asc,
                                  phase_sbr, phase_asc, grid_np, args.standoff, args.altitude)
    scored = [r for r in per_bldg if not r['skipped']]
    skipped = [r for r in per_bldg if r['skipped']]
    b_ssim = np.array([r['ssim'] for r in scored])
    b_coherence = np.array([r['coherence'] for r in scored])
    b_phase_rms = np.array([r['phase_rms_deg'] for r in scored])
    b_phase_bias = np.array([r['phase_bias_deg'] for r in scored])
    n_pass_b = int((b_ssim >= 0.95).sum())
    print(f"Scored {len(scored)}/{len(per_bldg)} buildings "
          f"({len(skipped)} skipped -- crop ran off the image edge)")
    if len(scored) > 0:
        print(f"SSIM:       mean={b_ssim.mean():.4f}  median={np.median(b_ssim):.4f}  "
              f"min={b_ssim.min():.4f}  max={b_ssim.max():.4f}")
        print(f"Coherence:  mean={b_coherence.mean():.4f}  median={np.median(b_coherence):.4f}  "
              f"min={b_coherence.min():.4f}  max={b_coherence.max():.4f}")
        print(f"Phase RMS:  mean={b_phase_rms.mean():.2f}deg  median={np.median(b_phase_rms):.2f}deg  "
              f"max={b_phase_rms.max():.2f}deg")
        print(f"Phase bias: mean={b_phase_bias.mean():+.2f}deg  "
              f"|mean| across buildings -- a per-building bias that's consistently signed "
              f"one way (not scattered around 0) is the structural-mismatch tell")
        print(f"Buildings passing (SSIM>=0.95): {n_pass_b}/{len(scored)} "
              f"({100*n_pass_b/len(scored):.0f}%)")
        worst = sorted(scored, key=lambda r: r['ssim'])[:5]
        print("Worst 5 buildings (by SSIM):")
        for r in worst:
            print(f"  building {r['building']:4d}  (x={r['cx']:7.1f}, y={r['cy']:7.1f}, "
                  f"h={r['height_m']:4.1f}m)  SSIM={r['ssim']:.4f}  coherence={r['coherence']:.4f}  "
                  f"amp_rms={r['amp_rms']:.4f}  phase_rms={r['phase_rms_deg']:.1f}deg  "
                  f"phase_bias={r['phase_bias_deg']:+.1f}deg  phase_max={r['phase_max_deg']:.1f}deg  "
                  f"crop={r['crop_px'][0]}x{r['crop_px'][1]}px")
        worst_coh = sorted(scored, key=lambda r: r['coherence'])[:5]
        print("Worst 5 buildings (by coherence -- may not be the same set as worst-by-SSIM):")
        for r in worst_coh:
            print(f"  building {r['building']:4d}  (x={r['cx']:7.1f}, y={r['cy']:7.1f}, "
                  f"h={r['height_m']:4.1f}m)  coherence={r['coherence']:.4f}  SSIM={r['ssim']:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, db, title in zip(axes[:2], [db_sbr, db_asc], ['Dense SBR', 'ASC-cached']):
        im = ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower',
                        extent=[-args.footprint/2, args.footprint/2]*2)
        ax.set_title(title)
        ax.set_xlabel('cross-range (m)')
        ax.set_ylabel('range (m)')
    diff = db_sbr - db_asc
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower',
                          extent=[-args.footprint/2, args.footprint/2]*2)
    axes[2].set_title(f'Difference (dB)\nwhole-image SSIM={ssim_score:.3f}')
    axes[2].set_xlabel('cross-range (m)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if len(scored) > 0:
        axes[3].hist(b_ssim, bins=min(20, max(5, len(scored)//3)), color='#0B7285', edgecolor='white')
        axes[3].axvline(0.95, color='#B85C00', linestyle='--', linewidth=1.5, label='0.95 criterion')
        axes[3].set_title(f'Per-building SSIM\n{n_pass_b}/{len(scored)} pass '
                           f'({100*n_pass_b/len(scored):.0f}%), mean={b_ssim.mean():.3f}')
        axes[3].set_xlabel('SSIM')
        axes[3].set_ylabel('buildings')
        axes[3].legend(fontsize=8)
    else:
        axes[3].text(0.5, 0.5, 'no buildings scored', ha='center', va='center')

    fig.suptitle(f'Tier 2 validation: {facets["n_buildings"]} buildings, '
                 f'{args.footprint:.0f}m x {args.footprint:.0f}m, {args.pulses} pulses')
    fig.tight_layout()
    fig.savefig('tier2_sbr_vs_asc_comparison.png', dpi=150)
    print("\nSaved tier2_sbr_vs_asc_comparison.png")

    # ---- Separate figure: phase/coherence, on its own since it answers a
    # different question (does phase agree spatially, not just does the
    # image look the same) and was crowding the main 4-panel comparison. ----
    fig2, ax2 = plt.subplots(2, 2, figsize=(13, 11))

    # Spatially-resolved coherence map -- reveals WHERE disagreement lives,
    # not just one averaged-away number. Clutter is added identically to
    # both branches, so background should read near-1; low coherence should
    # be localized at buildings if compression fidelity is the real story.
    coh_map = coherence_map(img_sbr, img_asc, win=7)
    im3 = ax2[0, 0].imshow(coh_map, cmap='viridis', vmin=0, vmax=1, origin='lower',
                            extent=[-args.footprint/2, args.footprint/2]*2)
    ax2[0, 0].set_title(f'Coherence map (7x7 window)\nwhole-image coherence={coherence:.3f}')
    ax2[0, 0].set_xlabel('cross-range (m)')
    ax2[0, 0].set_ylabel('range (m)')
    plt.colorbar(im3, ax=ax2[0, 0], fraction=0.046)

    # Spatial phase-difference map, magnitude-masked -- dphase is already a
    # full (img_size, img_size) array (computed above for the whole-image
    # scalar stats), just never plotted as a map before. Masked to the same
    # >5% peak threshold as phase_max so noise-floor pixels (uniform random
    # phase by construction) don't wash out the real disagreement visually.
    dphase_deg = np.degrees(dphase)
    dphase_masked = np.where(sig_mask_whole, dphase_deg, np.nan)
    im4 = ax2[0, 1].imshow(dphase_masked, cmap='twilight_shifted', vmin=-20, vmax=20, origin='lower',
                            extent=[-args.footprint/2, args.footprint/2]*2)
    ax2[0, 1].set_title(f'Phase difference (deg), masked to >5% peak\n'
                         f'bias={np.degrees(phase_bias):+.2f}deg, RMS={np.degrees(phase_rms):.2f}deg')
    ax2[0, 1].set_xlabel('cross-range (m)')
    ax2[0, 1].set_ylabel('range (m)')
    plt.colorbar(im4, ax=ax2[0, 1], fraction=0.046, label='deg')

    if len(scored) > 0:
        heights = np.array([r['height_m'] for r in scored])
        sc = ax2[1, 0].scatter(b_ssim, b_coherence, c=heights, cmap='viridis', s=26, edgecolor='k', linewidth=0.3)
        lo = min(b_ssim.min(), b_coherence.min()) - 0.01
        ax2[1, 0].plot([lo, 1], [lo, 1], 'k--', linewidth=0.8, label='y=x')
        ax2[1, 0].set_xlabel('SSIM')
        ax2[1, 0].set_ylabel('coherence')
        ax2[1, 0].set_title('Per-building: SSIM vs. coherence\n(color = building height)')
        ax2[1, 0].legend(fontsize=8)
        plt.colorbar(sc, ax=ax2[1, 0], fraction=0.046, label='height (m)')

        ax2[1, 1].hist(b_phase_rms, bins=min(20, max(5, len(scored)//3)), color='#B85C00', edgecolor='white')
        ax2[1, 1].set_title(f'Per-building phase RMS\nmean={b_phase_rms.mean():.2f}deg, '
                             f'max={b_phase_rms.max():.2f}deg')
        ax2[1, 1].set_xlabel('phase RMS (deg)')
        ax2[1, 1].set_ylabel('buildings')
    else:
        ax2[1, 0].text(0.5, 0.5, 'no buildings scored', ha='center', va='center')
        ax2[1, 1].text(0.5, 0.5, 'no buildings scored', ha='center', va='center')

    fig2.suptitle(f'Tier 2 phase/coherence detail: {facets["n_buildings"]} buildings, '
                  f'{args.footprint:.0f}m x {args.footprint:.0f}m, {args.pulses} pulses')
    fig2.tight_layout()
    fig2.savefig('tier2_phase_coherence.png', dpi=150)
    print("Saved tier2_phase_coherence.png")

    result = dict(
        footprint_m=args.footprint, density_per_km2=args.density,
        n_buildings=facets['n_buildings'], n_facets=facets['n_facets'],
        n_pulses=args.pulses, n_rays=sbr_stats['n_rays'],
        whole_image=dict(ssim=float(ssim_score), amp_rms=float(amp_rms),
                          phase_rms_deg=float(np.degrees(phase_rms)),
                          phase_bias_deg=float(np.degrees(phase_bias)),
                          phase_max_deg=float(np.degrees(phase_max)),
                          coherence=float(coherence), pass_ssim=bool(pass_ssim)),
        per_building_summary=dict(
            n_scored=len(scored), n_skipped=len(skipped),
            n_pass=n_pass_b,
            pass_fraction=(n_pass_b / len(scored)) if scored else None,
            ssim_mean=float(b_ssim.mean()) if len(scored) else None,
            ssim_median=float(np.median(b_ssim)) if len(scored) else None,
            ssim_min=float(b_ssim.min()) if len(scored) else None,
            ssim_max=float(b_ssim.max()) if len(scored) else None,
            coherence_mean=float(b_coherence.mean()) if len(scored) else None,
            coherence_median=float(np.median(b_coherence)) if len(scored) else None,
            coherence_min=float(b_coherence.min()) if len(scored) else None,
            coherence_max=float(b_coherence.max()) if len(scored) else None,
            phase_rms_mean_deg=float(b_phase_rms.mean()) if len(scored) else None,
            phase_rms_max_deg=float(b_phase_rms.max()) if len(scored) else None,
            phase_bias_mean_deg=float(b_phase_bias.mean()) if len(scored) else None,
        ),
        per_building=per_bldg,
        dense_sbr_ms_per_pulse=sbr_stats['t_per_pulse_ms'],
        asc_cached_ms_per_pulse=asc_stats['t_per_pulse_ms'],
    )
    with open('tier2_validation_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved tier2_validation_result.json")


if __name__ == '__main__':
    main()
