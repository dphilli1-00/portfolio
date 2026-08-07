"""
Diagnostic: plot_tier2_from_cache.py's combined (order1+order2 vs leg1+leg2)
whole-image SSIM/coherence on the real 1000m/200-building cache came back
much worse than expected (SSIM=0.83, phase_bias=-79deg, only 1/200 buildings
passing) -- and a near-90-degree phase bias is a red flag for a real bug,
not just "the known leg1/leg2_ground degradation is severe." This isolates
WHERE the problem is by scoring leg1-only, leg2-only, and combined
SEPARATELY against the same caches, instead of only ever looking at the sum:

  - If leg1-only alone already shows a large phase bias / bad SSIM, that
    points at something wrong in the geometry reconstruction or scoring
    pipeline itself (leg1 has no known issue with retro-check/culling, and
    should look close to the historically-good ~0.98 whole-image SSIM this
    project measured before any of this session's leg2 work started).
  - If leg1-only looks fine but leg2-only or combined is where it breaks,
    that's consistent with the already-flagged (but not yet investigated)
    leg1/leg2_ground degradation at this scale, or something specific to
    how leg2's retro-check/culled-search population interacts with leg1
    when summed.

Same cache format as plot_tier2_from_cache.py -- reuses its geometry
reconstruction verbatim, just doesn't sum the legs before backprojecting.

Usage:
    python3 diagnose_tier2_leg_split.py --footprint 1000 --pulses 200 \\
        --sbr-cache sbr1000_p200.npz --asc-cache asc1000_p200.npz
"""
import argparse
import os
import sys

import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from validation_tier2_compare import (backproject, to_numpy, compute_ref_ranges, complex_coherence,
                                       required_freq_samples, per_building_ssim)

ap = argparse.ArgumentParser()
ap.add_argument('--footprint', type=float, default=1000.0)
ap.add_argument('--density', type=float, default=200.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dr-max', type=float, default=2.0)
ap.add_argument('--delta-cr', type=float, default=2.0)
ap.add_argument('--standoff', type=float, default=8000.0)
ap.add_argument('--altitude', type=float, default=3000.0)
ap.add_argument('--fc', type=float, default=10e9)
ap.add_argument('--bandwidth', type=float, default=600e6)
ap.add_argument('--az-res', type=float, default=2.0)
ap.add_argument('--pulses', type=int, default=100)
ap.add_argument('--img-size', type=int, default=200)
ap.add_argument('--sbr-cache', type=str, required=True)
ap.add_argument('--asc-cache', type=str, required=True)
ap.add_argument('--ssim-floor-db', type=float, default=30.0,
                 help='clip both dB images to [-floor, 0] before scoring SSIM -- see '
                      'plot_tier2_from_cache.py\'s docstring for why (measured: unclipped whole-'
                      'image SSIM 0.83 -> 0.947 at 30dB on the 1000m/200-building cache, mostly '
                      'noise-floor comparison, not real disagreement).')
ap.add_argument('--phase-rms-floor-db', type=float, default=30.0,
                 help='hard floor for phase_rms/phase_bias/coherence (whole-image and per-building) -- '
                      'separate from --ssim-floor-db, see per_building_ssim\'s docstring in '
                      'validation_tier2_compare.py and plot_tier2_from_cache.py\'s own --phase-rms-'
                      'floor-db help for the full measurement and why 30dB (not the more dramatic-'
                      'looking 20dB) is the default: 20dB keeps only 1.20% of whole-image pixels and '
                      'leaves 95/200 buildings under 1% of their crop kept -- a small-sample problem, '
                      'not a cleaner comparison. Pass a negative number to restore the old soft-'
                      'weighting-only behavior.')
ap.add_argument('--no-gtd-phase-correction', dest='gtd_phase_correction', action='store_false',
                 help='ON BY DEFAULT -- pass this to disable it. Multiplies cached SBR arrays by '
                      '1j*(freqs/f0) before backprojection, matching ASC\'s own (j*f/f0)^alpha=1.0 GTD '
                      'term -- see plot_tier2_from_cache.py\'s docstring for the full derivation/'
                      'validation. Applied to order1_sbr and order2_sbr identically (both legs use the '
                      'same alpha=1.0 in ASC). Validated: collapses phase bias from -79.39deg to '
                      '-0.13deg on the real 1000m/200-building cache.')
ap.set_defaults(gtd_phase_correction=True)
args = ap.parse_args()

xp, on_gpu = get_backend(False)
print("Backend: numpy (CPU)")

standoff, altitude = args.standoff, args.altitude
o_broadside = np.array([0.0, -standoff, altitude])
wavelength = C / args.fc

facets_b = make_multi_building_scene_adaptive(
    xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
    dR_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets (rebuilt scene, seed={args.seed})")

d_theta = wavelength / (2.0 * args.az_res)
squint_len = d_theta * standoff
n_pulses = int(np.ceil(standoff * wavelength / (args.az_res ** 2)))
if n_pulses < args.pulses:
    n_pulses = args.pulses
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = xp.asarray(np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

min_freq, maxdr, needed_window = required_freq_samples(
    to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
n_freq = int(np.ceil(min_freq * 1.05))
freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, n_freq))
print(f"n_pulses={n_pulses}, n_freq={n_freq}")

sbr_cached = np.load(args.sbr_cache, allow_pickle=True)
order1_sbr = sbr_cached['order1']
order2_sbr = sbr_cached['order2']
asc_cached = np.load(args.asc_cache, allow_pickle=True)
leg1_asc = asc_cached['leg1']
leg2_asc = asc_cached['leg2']
for name, arr in [('order1_sbr', order1_sbr), ('order2_sbr', order2_sbr),
                   ('leg1_asc', leg1_asc), ('leg2_asc', leg2_asc)]:
    if arr.shape != (n_pulses, n_freq):
        raise SystemExit(f"shape mismatch: {name} is {arr.shape}, expected {(n_pulses, n_freq)}")

if args.gtd_phase_correction:
    f0 = float(np.mean(to_numpy(freqs)))
    gtd_correction = 1j * (to_numpy(freqs) / f0)
    order1_sbr = order1_sbr * gtd_correction[None, :]
    order2_sbr = order2_sbr * gtd_correction[None, :]
    print(f"applied GTD phase correction (1j * freqs/f0, f0={f0/1e9:.4f}GHz) to order1_sbr/order2_sbr")


def score(sbr_arr, asc_arr, label):
    img_sbr = backproject(xp, xp.asarray(sbr_arr), plat, freqs, grid, grid, args.fc, R_ref)
    img_asc = backproject(xp, xp.asarray(asc_arr), plat, freqs, grid, grid, args.fc, R_ref)
    mag_sbr, mag_asc = np.abs(to_numpy(img_sbr)), np.abs(to_numpy(img_asc))
    phase_sbr, phase_asc = np.angle(to_numpy(img_sbr)), np.angle(to_numpy(img_asc))
    db_sbr = 20 * np.log10(mag_sbr / (mag_sbr.max() + 1e-12) + 1e-6)
    db_asc = 20 * np.log10(mag_asc / (mag_asc.max() + 1e-12) + 1e-6)
    db_sbr_clip = np.clip(db_sbr, -args.ssim_floor_db, 0.0)
    db_asc_clip = np.clip(db_asc, -args.ssim_floor_db, 0.0)
    s = ssim(db_sbr_clip, db_asc_clip, data_range=args.ssim_floor_db)
    dphase = np.angle(np.exp(1j * (phase_sbr - phase_asc)))
    if args.phase_rms_floor_db >= 0:
        floor_mask = db_sbr > -args.phase_rms_floor_db
    else:
        floor_mask = mag_sbr > 0.05 * (mag_sbr.max() + 1e-12)
    w_mag = np.where(floor_mask, mag_sbr, 0.0) if floor_mask.any() else mag_sbr
    w = w_mag / (w_mag.sum() + 1e-12)
    phase_rms = np.degrees(np.sqrt(np.sum(w * dphase ** 2)))
    phase_bias = np.degrees(np.sum(w * dphase))
    # coherence: EXCLUDE noise-floor pixels (zero both branches at masked-
    # out locations) rather than let them dilute the ratio -- same fix as
    # plot_tier2_from_cache.py's whole-image coherence and per_building_
    # ssim's coherence_b, previously the only one of the three left unmasked.
    img_sbr_np, img_asc_np = to_numpy(img_sbr), to_numpy(img_asc)
    img_sbr_masked = np.where(floor_mask, img_sbr_np, 0.0)
    img_asc_masked = np.where(floor_mask, img_asc_np, 0.0)
    coherence = complex_coherence(img_sbr_masked, img_asc_masked)
    print(f"\n=== {label} ===")
    print(f"whole-image SSIM: {s:.4f} (clipped to -{args.ssim_floor_db:.0f}dB)   phase RMS: {phase_rms:.2f}deg   "
          f"phase bias: {phase_bias:+.2f}deg   coherence: {coherence:.4f}")
    grid_np = to_numpy(grid)
    per_bldg = per_building_ssim(facets_b, db_sbr_clip, db_asc_clip, mag_sbr, mag_asc,
                                  phase_sbr, phase_asc, grid_np, standoff, altitude,
                                  data_range=args.ssim_floor_db,
                                  phase_rms_floor_db=(args.phase_rms_floor_db
                                                       if args.phase_rms_floor_db >= 0 else None))
    scored = [r for r in per_bldg if not r['skipped']]
    if scored:
        b_ssim = np.array([r['ssim'] for r in scored])
        n_pass = int((b_ssim >= 0.95).sum())
        print(f"per-building SSIM: mean={b_ssim.mean():.4f} median={np.median(b_ssim):.4f} "
              f"min={b_ssim.min():.4f}  pass={n_pass}/{len(scored)} ({100*n_pass/len(scored):.0f}%)")
    return dict(ssim=s, phase_rms=phase_rms, phase_bias=phase_bias, coherence=coherence)


print("\nScoring leg1/order1 ALONE, leg2/order2 ALONE, and combined -- to isolate where the "
      "phase bias / SSIM collapse actually comes from...")
r1 = score(order1_sbr, leg1_asc, "leg1 / order1 ONLY (direct, single-bounce)")
r2 = score(order2_sbr, leg2_asc, "leg2 / order2 ONLY (double-bounce)")
rc = score(order1_sbr + order2_sbr, leg1_asc + leg2_asc, "COMBINED (order1+2 vs leg1+2)")

print("\n=== summary ===")
for label, r in [("leg1 only", r1), ("leg2 only", r2), ("combined", rc)]:
    print(f"{label:12s}: SSIM={r['ssim']:.4f}  phase_bias={r['phase_bias']:+7.2f}deg  "
          f"phase_rms={r['phase_rms']:6.2f}deg  coherence={r['coherence']:.4f}")
print("\nIf leg1-only already shows a big phase bias / poor SSIM, that's a reconstruction/scoring "
      "pipeline problem, not a leg2-specific physics issue -- leg1 has no retro-check or culling "
      "involved and should look close to this project's historical ~0.98 whole-image SSIM baseline.")
