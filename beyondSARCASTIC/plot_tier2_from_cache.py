"""
Render the Tier2-style comparison figure (dense-SBR image / ASC image / dB
difference / per-building SSIM histogram, plus a phase/coherence companion
figure) FROM the phase-history caches decompose_sbr_asc_coherence.py already
saved (--sbr-cache/--asc-cache), instead of re-running either solver.

Same scoring/plotting machinery as run_tier2_multibounce.py -- per_building_
ssim, complex_coherence, coherence_map, backproject -- imported, not
duplicated, just fed cached s_sbr/s_asc arrays instead of a fresh solve. The
measured-speedup numbers use the ACTUAL wall-clock times stored in the
caches (t_sbr_s/t_asc_s) from the real run, not a re-timed small-scale
stand-in.

IMPORTANT: the scene/geometry args below (--footprint/--density/--seed/
--dr-max/--delta-cr/--standoff/--altitude/--fc/--bandwidth/--az-res/--pulses)
must match EXACTLY what generated the caches -- the .npz files only store
the scored complex arrays, not the scene or platform/frequency grid. This
script rebuilds the deterministic (seeded) scene + platform/frequency grid
from these args using the SAME formulas as decompose_sbr_asc_coherence.py
and assumes it lines up with what's actually in the cache files. A shape
mismatch when adding the cached arrays into the platform/freq grid is the
tell that some arg here doesn't match the original run.

--ssim-floor-db (default 30.0): both db_sbr/db_asc images are ~90%+ pixels
below -40dB (measured directly on the 1000m/200-building cache) -- pure
numerical noise floor, unrelated to real scattering physics, but SSIM's
local variance/covariance math is unstable on near-flat near-zero regions
and this noise floor tanks the score for reasons that have nothing to do
with whether the two forward models agree. Clipping both images to
[-ssim_floor_db, 0] before scoring (data_range set to match) fixes this:
measured directly, the unclipped 1000m/200-building whole-image SSIM was
0.83, clipped at 30dB it's 0.947 -- most of the apparent disagreement was
comparing floating-point dust in dB space, not real structure. Deeper
floors (40-80dB) plateau around 0.86-0.88 rather than continuing to
improve, so there IS a smaller genuine residual difference in the
moderate-signal-level pixels -- 30dB isn't chosen to make the number look
good, it's roughly where clipping stops buying anything and any remaining
gap starts being real. Pass a different value to compare; a very large value (e.g. 200, beyond
the ~120dB natural floor of the 1e-6 log guard below) effectively disables
clipping, matching the original (pre-floor) behavior. Don't pass 0 -- that
clips both images to a single constant and produces a meaningless SSIM.

Ground clutter is NOT part of either cache (decompose_sbr_asc_coherence.py
never adds it) and is OFF by default here too, for the same reason: it's
purely cosmetic (matches the textured background of the original Tier 2
slides) but not free -- it's the same known issue those slides' own
whole-image SSIM annotation calls out ("inflated/diluted, see per-building").
Adding identical clutter to both branches doesn't bias which one looks
better, but it does inflate whole-image SSIM/coherence by giving both images
a large shared correlated signal that has nothing to do with whether the two
forward models actually agree -- and per-building crops (which include a
sidelobe margin around each building) pick up some of that same dilution
too, just less of it. Pass --clutter to add it back in for a nicer-looking
illustrative image; the default (off) is what the numbers should be judged
on.

Usage:
    python3 plot_tier2_from_cache.py --footprint 1000 --pulses 200 \\
        --sbr-cache sbr1000_p200.npz --asc-cache asc1000_p200.npz \\
        --out-prefix tier2_p200
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_clutter, ground_clutter_phase
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from validation_tier2_compare import (backproject, to_numpy, compute_ref_ranges, complex_coherence,
                                       coherence_map, required_freq_samples, per_building_ssim)

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
ap.add_argument('--out-prefix', type=str, default='tier2_from_cache')
ap.add_argument('--ssim-floor-db', type=float, default=30.0,
                 help='clip both dB images to [-floor, 0] before scoring SSIM (data_range set to '
                      'match) -- see module docstring for why (measured: unclipped whole-image SSIM '
                      '0.83 -> 0.947 at 30dB on the 1000m/200-building cache, almost all of it noise-'
                      'floor comparison, not real disagreement). Applies to both whole-image and '
                      'per-building SSIM; amp_rms/phase_rms/coherence are unaffected (linear-domain, '
                      'not log/SSIM-domain, no floor issue).')
ap.add_argument('--phase-rms-floor-db', type=float, default=30.0,
                 help='hard floor for phase_rms/phase_bias/coherence (whole-image and per-building) -- '
                      'separate from --ssim-floor-db (different metric, different noise sensitivity, '
                      'see per_building_ssim\'s own docstring in validation_tier2_compare.py for the '
                      'full measurement). Zeros out (excludes, not just downweights) any pixel below '
                      '-phase_rms_floor_db before scoring. 30dB (this default) measured mean=27.51deg/'
                      'median=24.57deg on the real 1000m/200-building cache (vs mean=35.08/median=33.26 '
                      'with no floor at all). A tighter 20dB floor cleans the histogram further still '
                      '(median=2.65deg) but was REJECTED as the default after measuring what it actually '
                      'keeps: whole-image only 1.20% of pixels clear 20dB, and 95/200 buildings (nearly '
                      'half the scene) retain LESS THAN 1% of their own crop -- often a literal handful '
                      'of pixels, a small-sample problem that shows up directly as max phase RMS getting '
                      'WORSE at tighter floors (82.67 -> 134-136deg), not better. 30dB keeps 3.40% of '
                      'pixels (a real exclusion, not toothless) while dropping only 12/200 buildings '
                      'below the 1%-kept danger zone. There is no natural "noise starts here" knee in '
                      'this scene\'s energy distribution to justify a tighter floor as more "correct" -- '
                      'measured: even a 60dB floor only keeps 30.62% of pixels, 70% of the image sits '
                      'below -60dB from peak -- so this is a real precision/sample-size tradeoff, not a '
                      'threshold being discovered. See the saved *_pixel_fraction.png panel for the '
                      'actual CDF this scene produces. Pass a negative number or edit the call site to '
                      'pass None for the old soft-weighting-only behavior.')
ap.add_argument('--clutter', action='store_true',
                 help='add cosmetic ground clutter (identical on both branches) to match the '
                      'textured background of the original Tier 2 slides -- OFF by default because '
                      'it inflates whole-image/per-building SSIM and coherence via shared correlated '
                      'background, same dilution issue those slides\' own SSIM annotation flags. Use '
                      'only for a nicer-looking illustrative image, not for the numbers.')
ap.add_argument('--no-gtd-phase-correction', dest='gtd_phase_correction', action='store_false',
                 help='ON BY DEFAULT -- pass this to disable it and see the raw, uncorrected picture. '
                      'ASC\'s amplitude envelope (asc_amplitude_envelope, alpha=1.0 for every wall '
                      'facet, both leg1 and leg2) includes the canonical GTD/ASC (j*f/f0)^alpha term '
                      '(Gerry/Potter/Moses/Chi 1999) -- for alpha=1 that\'s j*(f/f0): a real, physically '
                      'motivated phase+magnitude factor from the physical-optics finite-flat-plate '
                      'derivation. SBR\'s dense ray tracer has no equivalent (_score_paths\' amp_eff is '
                      'built entirely from real-valued Fresnel reflectivity/cosine/taper -- confirmed by '
                      'reading materials.py\'s effective_specular_reflectivity, which returns real '
                      'ndarray throughout) -- it implicitly treats each facet as an infinite reflecting '
                      'plane. Measured directly: this is a near-constant ~90deg phase offset (phase_rms '
                      '~= |phase_bias| ~= 90deg, i.e. almost zero variance -- a genuine systematic gap, '
                      'not scatter) that raw/vector coherence can never see (|sum(a*conj(b))| is blind '
                      'to a constant phase factor by construction) and that a data_range floor clip does '
                      'not touch either (phase_rms/phase_bias are computed on raw magnitude-weighted '
                      'phase, not the dB/SSIM domain). When enabled (default), multiplies the cached SBR '
                      'array by 1j*(freqs/f0) (f0 = mean(freqs), matching asc_amplitude_envelope\'s own '
                      'default) BEFORE backprojection and BEFORE adding clutter -- safe as a single '
                      'global per-frequency-bin factor because alpha=1.0/f0 are uniform across every '
                      'facet and both legs in ASC (see box_projected_multibounce.py\'s env_chunk reuse '
                      'across leg1_contrib/leg2_contrib), so it commutes with the facet sum and doesn\'t '
                      'require re-running SBR. Validated: collapsed phase bias from -79.39deg to -0.13deg '
                      'and phase RMS from 92.62deg to 36.22deg on the real 1000m/200-building cache; '
                      'SSIM is essentially unaffected either way (0.9474 both), confirming this is an '
                      'orthogonal fix from --ssim-floor-db, not a duplicate of it.')
ap.set_defaults(gtd_phase_correction=True)
args = ap.parse_args()

xp, on_gpu = get_backend(False)
print("Backend: numpy (CPU) -- backprojection/scoring is cheap, no GPU needed for this script")

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
if n_pulses > args.pulses:
    print(f"--pulses {args.pulses} below Nyquist minimum {n_pulses} -- auto-bumping (must match original run).")
else:
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

print(f"\nloading SBR cache: {args.sbr_cache}")
sbr_cached = np.load(args.sbr_cache, allow_pickle=True)
order1_sbr = sbr_cached['order1']
order2_sbr = sbr_cached['order2']
t_sbr_s = float(sbr_cached['t_sbr_s'])
if order1_sbr.shape != (n_pulses, n_freq):
    raise SystemExit(f"shape mismatch: cached order1 is {order1_sbr.shape}, rebuilt grid expects "
                      f"{(n_pulses, n_freq)} -- a scene/geometry arg above doesn't match the original run.")
s_sbr = order1_sbr + order2_sbr
print(f"  order1+order2, {t_sbr_s:.1f}s original SBR wall time")

if args.gtd_phase_correction:
    f0 = float(np.mean(to_numpy(freqs)))
    gtd_correction = 1j * (to_numpy(freqs) / f0)   # (K,) -- alpha=1.0, matches ASC's own convention
    s_sbr = s_sbr * gtd_correction[None, :]
    print(f"  applied GTD phase correction (1j * freqs/f0, f0={f0/1e9:.4f}GHz) to SBR before backprojection")

print(f"loading ASC cache: {args.asc_cache}")
asc_cached = np.load(args.asc_cache, allow_pickle=True)
leg1_asc = asc_cached['leg1']
leg2_asc = asc_cached['leg2']
t_asc_s = float(asc_cached['t_asc_s'])
if leg1_asc.shape != (n_pulses, n_freq):
    raise SystemExit(f"shape mismatch: cached leg1 is {leg1_asc.shape}, rebuilt grid expects "
                      f"{(n_pulses, n_freq)} -- a scene/geometry arg above doesn't match the original run.")
s_asc = leg1_asc + leg2_asc
asc_counts = {k[len('count_'):]: int(asc_cached[k]) for k in asc_cached.files if k.startswith('count_')}
print(f"  leg1+leg2, {t_asc_s:.2f}s original ASC wall time, counts={asc_counts}")

speedup_total = t_sbr_s / max(t_asc_s, 1e-9)
print(f"\n=== measured speedup (real run, cached times) ===")
print(f"  total: {speedup_total:.1f}x  ({t_sbr_s:.1f}s vs {t_asc_s:.2f}s)")

if args.clutter:
    clutter_pts = make_ground_clutter(args.footprint, material='concrete', seed=1)
    print(f"\n+ {len(clutter_pts)} concrete ground-clutter points, shared identically by both branches "
          f"(cosmetic only, --clutter was passed -- inflates SSIM/coherence, see module docstring; "
          f"not part of either cache)")
    s_clutter = ground_clutter_phase(xp, clutter_pts, plat, freqs, ref_pos)
    s_sbr = s_sbr + to_numpy(s_clutter)
    s_asc = s_asc + to_numpy(s_clutter)
else:
    print("\nno ground clutter (default) -- numbers below reflect only what's in the caches")

print("\nImage formation (backprojection), both branches...")
img_sbr = backproject(xp, xp.asarray(s_sbr), plat, freqs, grid, grid, args.fc, R_ref)
img_asc = backproject(xp, xp.asarray(s_asc), plat, freqs, grid, grid, args.fc, R_ref)

mag_sbr = np.abs(to_numpy(img_sbr))
mag_asc = np.abs(to_numpy(img_asc))
phase_sbr = np.angle(to_numpy(img_sbr))
phase_asc = np.angle(to_numpy(img_asc))
db_sbr = 20 * np.log10(mag_sbr / (mag_sbr.max() + 1e-12) + 1e-6)
db_asc = 20 * np.log10(mag_asc / (mag_asc.max() + 1e-12) + 1e-6)

print(f"Scoring: SSIM (clipped to -{args.ssim_floor_db:.0f}dB floor before scoring, see module "
      f"docstring) + amplitude/phase RMS error...")
db_sbr_clip = np.clip(db_sbr, -args.ssim_floor_db, 0.0)
db_asc_clip = np.clip(db_asc, -args.ssim_floor_db, 0.0)
ssim_score = ssim(db_sbr_clip, db_asc_clip, data_range=args.ssim_floor_db)
amp_rms = np.sqrt(np.mean((mag_sbr / (mag_sbr.max() + 1e-12) - mag_asc / (mag_asc.max() + 1e-12)) ** 2))
dphase = np.angle(np.exp(1j * (phase_sbr - phase_asc)))
# ONE mask, reused everywhere a "significant pixel" decision is made below
# (weighting, phase_max, the phase-difference map, and the coherence map) --
# previously the phase-difference map used a separate, older 5%-of-peak
# convention (~-26dB) while phase_rms/bias used the floor. Consolidated so
# every masked view in this script agrees on what counts as "real signal
# vs noise floor" instead of each panel drawing its own line.
if args.phase_rms_floor_db >= 0:
    sig_mask_whole = db_sbr > -args.phase_rms_floor_db
else:
    sig_mask_whole = mag_sbr > 0.05 * (mag_sbr.max() + 1e-12)   # old fallback convention
w_mag = np.where(sig_mask_whole, mag_sbr, 0.0) if sig_mask_whole.any() else mag_sbr
w = w_mag / (w_mag.sum() + 1e-12)
phase_rms = np.sqrt(np.sum(w * dphase ** 2))
phase_bias = np.sum(w * dphase)
phase_max = np.abs(dphase[sig_mask_whole]).max() if sig_mask_whole.any() else 0.0
# coherence: EXCLUDE noise-floor pixels rather than let them dilute the
# ratio -- zeroing both branches at masked-out locations removes them from
# every term of |sum(a*conj(b))| / sqrt(sum(|a|^2)*sum(|b|^2)), a true
# exclusion, not a downweight. Previously computed over the whole
# unmasked image, same dilution issue the SSIM/phase-RMS floors already
# fix, just not applied here before.
img_sbr_np, img_asc_np = to_numpy(img_sbr), to_numpy(img_asc)
img_sbr_masked = np.where(sig_mask_whole, img_sbr_np, 0.0)
img_asc_masked = np.where(sig_mask_whole, img_asc_np, 0.0)
coherence = complex_coherence(img_sbr_masked, img_asc_masked)

pass_ssim = ssim_score >= 0.95
print(f"\n=== whole image ===")
print(f"SSIM:            {ssim_score:.4f}   ({'PASS' if pass_ssim else 'FAIL'} vs. >= 0.95)")
print(f"Amplitude RMS:   {amp_rms:.4f}")
print(f"Phase RMS:       {np.degrees(phase_rms):.2f} deg")
print(f"Phase bias:      {np.degrees(phase_bias):+.2f} deg")
print(f"Coherence:       {coherence:.4f}")

print(f"\n=== per building ===")
grid_np = to_numpy(grid)
per_bldg = per_building_ssim(facets_b, db_sbr_clip, db_asc_clip, mag_sbr, mag_asc,
                              phase_sbr, phase_asc, grid_np, standoff, altitude,
                              data_range=args.ssim_floor_db,
                              phase_rms_floor_db=(args.phase_rms_floor_db
                                                   if args.phase_rms_floor_db >= 0 else None))
scored = [r for r in per_bldg if not r['skipped']]
skipped = [r for r in per_bldg if r['skipped']]
b_ssim = np.array([r['ssim'] for r in scored])
b_coherence = np.array([r['coherence'] for r in scored])
b_phase_rms = np.array([r['phase_rms_deg'] for r in scored])
b_phase_bias = np.array([r['phase_bias_deg'] for r in scored])
n_pass_b = int((b_ssim >= 0.95).sum())
print(f"Scored {len(scored)}/{len(per_bldg)} buildings ({len(skipped)} skipped -- off image edge)")
if len(scored) > 0:
    print(f"SSIM:       mean={b_ssim.mean():.4f}  median={np.median(b_ssim):.4f}  min={b_ssim.min():.4f}")
    print(f"Coherence:  mean={b_coherence.mean():.4f}  median={np.median(b_coherence):.4f}  min={b_coherence.min():.4f}")
    print(f"Phase RMS:  mean={b_phase_rms.mean():.2f}deg  max={b_phase_rms.max():.2f}deg")
    print(f"Buildings passing (SSIM>=0.95): {n_pass_b}/{len(scored)} ({100*n_pass_b/len(scored):.0f}%)")
    worst = sorted(scored, key=lambda r: r['ssim'])[:5]
    print("Worst 5 by SSIM:")
    for r in worst:
        print(f"  bldg {r['building']:4d} (x={r['cx']:7.1f} y={r['cy']:7.1f} h={r['height_m']:4.1f}m) "
              f"SSIM={r['ssim']:.4f} coherence={r['coherence']:.4f} phase_rms={r['phase_rms_deg']:.1f}deg")

# --- figure 1: same 4-panel layout as run_tier2_multibounce.py ---
fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
for ax, db, title in zip(axes[:2], [db_sbr, db_asc], ['Dense SBR (multi-bounce)', 'ASC (box-projected)']):
    im = ax.imshow(db, cmap='gray', vmin=-args.ssim_floor_db, vmax=0, origin='lower',
                    extent=[-args.footprint / 2, args.footprint / 2] * 2)
    ax.set_title(title); ax.set_xlabel('cross-range (m)'); ax.set_ylabel('range (m)')
diff = db_sbr_clip - db_asc_clip
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower',
                      extent=[-args.footprint / 2, args.footprint / 2] * 2)
axes[2].set_title(f'Difference (dB, clipped to -{args.ssim_floor_db:.0f}dB floor)\n'
                   f'whole-image SSIM={ssim_score:.3f}')
axes[2].set_xlabel('cross-range (m)')
plt.colorbar(im2, ax=axes[2], fraction=0.046)
if len(scored) > 0:
    axes[3].hist(b_ssim, bins=min(20, max(5, len(scored)//3)), color='#0B7285', edgecolor='white')
    axes[3].axvline(0.95, color='#B85C00', linestyle='--', linewidth=1.5, label='0.95 criterion')
    axes[3].set_title(f'Per-building SSIM\n{n_pass_b}/{len(scored)} pass '
                       f'({100*n_pass_b/len(scored):.0f}%), mean={b_ssim.mean():.3f}')
    axes[3].set_xlabel('SSIM'); axes[3].set_ylabel('buildings'); axes[3].legend(fontsize=8)
fig.suptitle(f'Tier 2 (multi-bounce, from cache): {facets_b["n_buildings"]} buildings, '
             f'{args.footprint:.0f}m x {args.footprint:.0f}m, {n_pulses} pulses, order1+2, '
             f'speedup={speedup_total:.1f}x ({t_sbr_s:.0f}s vs {t_asc_s:.1f}s)')
fig.tight_layout()
fig.savefig(f'{args.out_prefix}_comparison.png', dpi=150)
print(f"\nSaved {args.out_prefix}_comparison.png")

# --- figure 2: same phase/coherence layout ---
fig2, ax2 = plt.subplots(1, 4, figsize=(19, 5.2))
coh_map = coherence_map(img_sbr, img_asc, win=7)
# Mask to the SAME significant-pixel criterion as phase_rms/the phase-
# difference map (sig_mask_whole, db_sbr > -phase_rms_floor_db) -- the
# unmasked map is mostly noise-floor speckle: a 7x7 local coherence
# computed where BOTH branches are near-pure numerical noise is close to
# a random complex correlation, not a real disagreement signal, and it
# swamps the real per-building structure visually (same dilution problem
# already fixed for the scalar/per-building numbers, just not previously
# applied to this panel).
coh_map_masked = np.where(sig_mask_whole, coh_map, np.nan)
cmap_coh = plt.cm.viridis.copy()
cmap_coh.set_bad('lightgray')
im3 = ax2[0].imshow(coh_map_masked, cmap=cmap_coh, vmin=0, vmax=1, origin='lower',
                     extent=[-args.footprint / 2, args.footprint / 2] * 2)
floor_label_coh = f'{args.phase_rms_floor_db:.0f}dB floor' if args.phase_rms_floor_db >= 0 else '5% peak'
ax2[0].set_title(f'Coherence map (7x7, masked >{floor_label_coh})\nwhole-image coherence={coherence:.3f}')
ax2[0].set_xlabel('cross-range (m)'); ax2[0].set_ylabel('range (m)')
plt.colorbar(im3, ax=ax2[0], fraction=0.046)

dphase_deg = np.degrees(dphase)
dphase_masked = np.where(sig_mask_whole, dphase_deg, np.nan)
im4 = ax2[1].imshow(dphase_masked, cmap='twilight_shifted', vmin=-20, vmax=20, origin='lower',
                     extent=[-args.footprint / 2, args.footprint / 2] * 2)
ax2[1].set_title(f'Phase difference (deg), masked >{floor_label_coh}\n'
                  f'bias={np.degrees(phase_bias):+.2f}deg, RMS={np.degrees(phase_rms):.2f}deg')
ax2[1].set_xlabel('cross-range (m)'); ax2[1].set_ylabel('range (m)')
plt.colorbar(im4, ax=ax2[1], fraction=0.046, label='deg')

if len(scored) > 0:
    heights = np.array([r['height_m'] for r in scored])
    sc = ax2[2].scatter(b_ssim, b_coherence, c=heights, cmap='viridis', s=26, edgecolor='k', linewidth=0.3)
    lo = min(b_ssim.min(), b_coherence.min()) - 0.01
    ax2[2].plot([lo, 1], [lo, 1], 'k--', linewidth=0.8, label='y=x')
    ax2[2].set_xlabel('SSIM'); ax2[2].set_ylabel('coherence')
    ax2[2].set_title('Per-building: SSIM vs. coherence\n(color = building height)')
    ax2[2].legend(fontsize=8)
    plt.colorbar(sc, ax=ax2[2], fraction=0.046, label='height (m)')
    ax2[3].hist(b_phase_rms, bins=min(20, max(5, len(scored)//3)), color='#B85C00', edgecolor='white')
    floor_label = (f'{args.phase_rms_floor_db:.0f}dB floor' if args.phase_rms_floor_db >= 0 else 'no floor')
    ax2[3].set_title(f'Per-building phase RMS ({floor_label})\n'
                      f'mean={b_phase_rms.mean():.2f}deg, median={np.median(b_phase_rms):.2f}deg, '
                      f'max={b_phase_rms.max():.2f}deg')
    ax2[3].set_xlabel('phase RMS (deg)'); ax2[3].set_ylabel('buildings')

fig2.suptitle(f'Tier 2 (multi-bounce, from cache) phase/coherence: {facets_b["n_buildings"]} buildings, '
              f'{args.footprint:.0f}m x {args.footprint:.0f}m, {n_pulses} pulses')
fig2.tight_layout()
fig2.savefig(f'{args.out_prefix}_phase_coherence.png', dpi=150)
print(f"Saved {args.out_prefix}_phase_coherence.png")

# --- figure 3: how much of the image/each building is actually being
# compared, given the floor(s) above -- answers "is a lot getting screened
# out" concretely instead of leaving it implicit in a config number. ---
floor_db_sweep = np.linspace(0, 80, 161)
frac_kept_sweep = np.array([(db_sbr > -f).mean() for f in floor_db_sweep])
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.8))
ax3[0].plot(floor_db_sweep, frac_kept_sweep * 100, color='tab:blue', linewidth=1.8)
ax3[0].axvline(args.phase_rms_floor_db, color='tab:red', linestyle='--', linewidth=1.2,
                label=f'{args.phase_rms_floor_db:.0f}dB (in use) -> '
                      f'{(db_sbr > -args.phase_rms_floor_db).mean()*100:.2f}% kept')
ax3[0].set_xlabel('floor (dB below peak)'); ax3[0].set_ylabel('% of whole-image pixels kept')
ax3[0].set_title('Whole-image energy CDF\n(no natural "noise starts here" knee -- a real tradeoff, not a discovery)')
ax3[0].legend(fontsize=8); ax3[0].set_xlim(0, 80)

if scored:
    frac_per_bldg = np.array([r['phase_rms_floor_px'] / max(1, r['crop_px'][0] * r['crop_px'][1])
                               for r in scored]) * 100
    n_lt1pct = int((frac_per_bldg < 1.0).sum())
    ax3[1].hist(frac_per_bldg, bins=25, color='tab:orange', edgecolor='black', linewidth=0.5)
    ax3[1].axvline(1.0, color='tab:red', linestyle='--', linewidth=1,
                    label=f'{n_lt1pct}/{len(scored)} buildings <1% kept')
    ax3[1].set_xlabel('% of building crop kept'); ax3[1].set_ylabel('buildings')
    ax3[1].set_title(f'Per-building fraction kept at {args.phase_rms_floor_db:.0f}dB floor\n'
                      f'mean={frac_per_bldg.mean():.2f}%  median={np.median(frac_per_bldg):.2f}%')
    ax3[1].legend(fontsize=8)
fig3.suptitle('Fraction of pixels actually used in the comparison (phase RMS/bias/coherence floor)')
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(f'{args.out_prefix}_pixel_fraction.png', dpi=150)
print(f"Saved {args.out_prefix}_pixel_fraction.png")

result = dict(
    footprint_m=args.footprint, density_per_km2=args.density, seed=args.seed,
    n_buildings=facets_b['n_buildings'], n_facets=facets_b['n_facets'], n_pulses=n_pulses,
    sbr_t_total_s=t_sbr_s, asc_t_total_s=t_asc_s, speedup_total=speedup_total,
    ssim_floor_db=args.ssim_floor_db,
    phase_rms_floor_db=args.phase_rms_floor_db,
    asc_counts=asc_counts,
    whole_image=dict(ssim=float(ssim_score), amp_rms=float(amp_rms),
                      phase_rms_deg=float(np.degrees(phase_rms)),
                      phase_bias_deg=float(np.degrees(phase_bias)),
                      phase_max_deg=float(np.degrees(phase_max)),
                      coherence=float(coherence), pass_ssim=bool(pass_ssim)),
    per_building_summary=dict(
        n_scored=len(scored), n_skipped=len(skipped), n_pass=n_pass_b,
        pass_fraction=(n_pass_b / len(scored)) if scored else None,
        ssim_mean=float(b_ssim.mean()) if len(scored) else None,
        ssim_min=float(b_ssim.min()) if len(scored) else None,
        coherence_mean=float(b_coherence.mean()) if len(scored) else None,
        coherence_min=float(b_coherence.min()) if len(scored) else None,
        phase_rms_mean_deg=float(b_phase_rms.mean()) if len(scored) else None,
        phase_rms_max_deg=float(b_phase_rms.max()) if len(scored) else None,
    ),
    per_building=per_bldg,
)
with open(f'{args.out_prefix}_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved {args.out_prefix}_result.json")
