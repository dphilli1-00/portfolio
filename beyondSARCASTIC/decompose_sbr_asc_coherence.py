"""Decompose SBR-vs-ASC coherence PER BOUNCE ORDER/LEG, instead of only the
combined number run_direct_to_facet_validation.py reports.

Motivation: that script measured raw/image coherence of 0.9515/0.9356 on a
500m/50-building/38,779-facet scene, but only 0.8982/0.8403 on a
1000m/200-building/146,053-facet scene -- a real, non-trivial fidelity drop
at the bigger/denser scale. The combined number conflates order1(leg1) and
order2(leg2); this script computes coherence for each SEPARATELY so we can
tell which one actually degraded, instead of guessing.

Leading hypothesis going in (NOT yet confirmed -- that's what this script is
for): box_projected_multibounce.py's leg2/leg3 run with occlusion checking
OFF (a known, documented gap -- see that module's own docstring), so at
higher building density there are more opportunities for a spuriously-
unoccluded ground/wall-to-wall bounce to get scored that a real ray would
have found blocked. If that's the whole story, leg2's coherence should be
visibly worse than leg1's, and the gap should widen with density.

IMPORTANT, worth checking rather than assuming: box-projected leg1 does NOT
check occlusion by other buildings either -- visible1 there is a pure
backface cull (cos_inc1 > 0), nothing more. SBR's order1, by contrast, gets
occlusion for free, structurally: ray_facet_intersect finds the NEAREST
facet a ray actually hits, so a facet blocked by another building simply
never registers as a hit at all. So leg1 could ALSO be over-counting
occluded facets at high density -- this script checks that explicitly
instead of assuming leg1 is clean just because it wasn't named in the
original bug report.

Usage:
    python3 decompose_sbr_asc_coherence.py --gpu
        # fast sanity check: matches the KNOWN-GOOD 500m/50-building scene
        # (should reproduce ~0.95 raw coherence per leg) -- confirms the
        # script itself is correct before trusting it on the degraded case.

    python3 decompose_sbr_asc_coherence.py --gpu --footprint 1000 --density 200
        # matches the scene that showed the coherence drop. SLOW: this is
        # the same ~50min SBR run as run_direct_to_facet_validation.py's
        # full-scale run, because it needs return_components=True on the
        # SBR side (order1/order2 kept separate), which the existing
        # --sbr-cache checkpoints from that script do NOT store. Use
        # --sbr-cache here (a script-local cache, not interchangeable with
        # run_direct_to_facet_validation.py's) to avoid re-paying this on
        # a second run.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import run_multibounce_sbr
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import backproject, to_numpy, compute_ref_ranges, complex_coherence, required_freq_samples

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--footprint', type=float, default=500.0)
ap.add_argument('--density', type=float, default=200.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dr-max', type=float, default=2.0)
ap.add_argument('--delta-cr', type=float, default=2.0)
ap.add_argument('--standoff', type=float, default=8000.0)
ap.add_argument('--altitude', type=float, default=3000.0)
ap.add_argument('--fc', type=float, default=10e9)
ap.add_argument('--bandwidth', type=float, default=600e6)
ap.add_argument('--az-res', type=float, default=2.0)
ap.add_argument('--pulses', type=int, default=100, help='auto-bumped to Nyquist minimum, same as run_direct_to_facet_validation.py')
ap.add_argument('--img-size', type=int, default=200)
ap.add_argument('--ground-material', type=str, default='dry_soil')
ap.add_argument('--sbr-cache', type=str, default=None,
                 help='script-local cache (stores per-order components) -- NOT interchangeable with '
                      "run_direct_to_facet_validation.py's own --sbr-cache files, which don't keep components")
ap.add_argument('--out', type=str, default='decompose_sbr_asc_coherence.png')
ap.add_argument('--leg2-occlusion', action='store_true',
                 help='ablation: enable box_projected_multibounce.py\'s leg2_occlusion_check '
                      '(cheap AABB-only occlusion test on the leg2 hop) to test whether the '
                      'unoccluded-bounce hypothesis explains a measured leg2 coherence collapse')
ap.add_argument('--split-leg2', action='store_true',
                 help='diagnostic: split leg2 into leg2_ground (wall->ground, a true 90-degree '
                      'dihedral) vs leg2_building (wall->another building, generally NOT a true '
                      'corner) and report each against the combined SBR order2 signal, to test '
                      'whether the missing retroreflection-validity check on wall-to-wall bounces '
                      'explains the leg2 coherence collapse')
ap.add_argument('--leg2-retro-check', action='store_true',
                 help='ASC-side fix: gate leg2 on whether the second bounce surface\'s own law '
                      'of reflection actually sends the ray back toward the sensor, within a '
                      'diffraction-limited tolerance -- should leave wall->ground unaffected '
                      '(exact for a 90-degree corner) while suppressing spurious wall->building '
                      'pairs that do not happen to form a near-90-degree corner')
ap.add_argument('--sbr-leg2-retro-check', action='store_true',
                 help='SBR-side counterpart, separate flag from --leg2-retro-check (which only '
                      'gates ASC): run_multibounce_sbr never checks whether bounce-2\'s own law of '
                      'reflection sends the ray back to the sensor, only that a geometric sightline '
                      'is unoccluded -- measured directly that only 8.5% of SBR\'s real building-'
                      'target order2 hits are actually retroreflective, so SBR\'s own reference '
                      'population is 91.5% non-physical noise scored at full coherent strength. '
                      'This is a VALIDITY gate on SBR\'s own population (see run_multibounce_sbr\'s '
                      'leg2_retro_check docstring), not just another ASC-side ablation -- pass this '
                      'alongside --leg2-retro-check for a real apples-to-apples comparison, since '
                      'without it SBR\'s order2_building reference is still the un-gated, mostly-'
                      'non-physical population.')
ap.add_argument('--retro-beamwidth-mult', type=float, default=3.0,
                 help='misalignment tolerance in beamwidths for --leg2-retro-check/--sbr-leg2-retro-check '
                      '(wider = more permissive)')
ap.add_argument('--leg2-ground-only', action='store_true',
                 help='comparison: drop wall->building leg2 entirely, fall back to the '
                      'originally-validated ground-only leg2 case, to check whether the '
                      'wall-to-building generalization is earning its complexity at this scale')
ap.add_argument('--leg2-retro-taper', action='store_true',
                 help='the real fix, supersedes --leg2-retro-check: continuous frequency-dependent '
                      'sinc taper on leg2 (sinc(2*pi*f*L_wall*sin(misalignment)/c)) instead of a '
                      'hard cutoff -- leg2 currently reuses leg1\'s own wall-illumination taper for '
                      'its amplitude, which has nothing to do with dihedral corner directivity; this '
                      'gives leg2 its own persistence taper based on the actual corner misalignment angle')
args = ap.parse_args()

xp, on_gpu = get_backend(args.gpu)
print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

standoff, altitude = args.standoff, args.altitude
o_broadside = np.array([0.0, -standoff, altitude])
wavelength = C / args.fc

facets_b = make_multi_building_scene_adaptive(
    xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
    dR_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, args.footprint)
n_facets = facets_b['n_facets']
print(f"{facets_b['n_buildings']} buildings, {n_facets} facets")

d_theta = wavelength / (2.0 * args.az_res)
squint_len = d_theta * standoff
n_pulses = int(np.ceil(standoff * wavelength / (args.az_res ** 2)))
if n_pulses > args.pulses:
    print(f"--pulses {args.pulses} below Nyquist minimum {n_pulses} -- auto-bumping.")
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

aim_pts = facets_b['center']

need_split_order2 = args.split_leg2
have_order2_split = False

if args.sbr_cache and os.path.exists(args.sbr_cache):
    print(f"loading cached SBR components from {args.sbr_cache}...")
    cached = np.load(args.sbr_cache, allow_pickle=True)
    order1_sbr = xp.asarray(cached['order1'])
    order2_sbr = xp.asarray(cached['order2'])
    t_sbr = float(cached['t_sbr_s'])
    print(f"  loaded ({t_sbr:.1f}s original wall time)")
    # Staleness guard: a cache file's mere EXISTENCE says nothing about
    # which flags it was generated with -- we've already been burned twice
    # this session by a stale cache silently reusing pre-fix SBR results
    # while only the requested flag changed. Every cache now stores which
    # leg2_retro_check/retro_beamwidth_mult it was built with; mismatch is
    # a loud warning, not a silent reuse.
    cached_retro_check = bool(cached['sbr_leg2_retro_check']) if 'sbr_leg2_retro_check' in cached.files else False
    cached_retro_mult = float(cached['retro_beamwidth_mult']) if 'retro_beamwidth_mult' in cached.files else None
    if cached_retro_check != args.sbr_leg2_retro_check or (
            args.sbr_leg2_retro_check and cached_retro_mult is not None and cached_retro_mult != args.retro_beamwidth_mult):
        print(f"  *** WARNING: this cache was built with sbr_leg2_retro_check={cached_retro_check}"
              f"{f', retro_beamwidth_mult={cached_retro_mult}' if cached_retro_mult is not None else ' (predates this flag)'}, "
              f"but this run requested sbr_leg2_retro_check={args.sbr_leg2_retro_check}"
              f"{f', retro_beamwidth_mult={args.retro_beamwidth_mult}' if args.sbr_leg2_retro_check else ''}. "
              f"Loading it anyway would silently score against the WRONG SBR population. Delete this "
              f"cache file or pass a new --sbr-cache name to get a fresh, correctly-flagged run. ***")
        raise SystemExit(1)
    if need_split_order2 and 'order2_ground' in cached.files:
        order2_ground_sbr = xp.asarray(cached['order2_ground'])
        order2_building_sbr = xp.asarray(cached['order2_building'])
        have_order2_split = True
    elif need_split_order2:
        print("  WARNING: this cache predates split_order2_by_target -- leg2_ground/leg2_building "
              "will be compared against the FULL mixed order2 signal (comparison artifact, see "
              "module docstring), not a real per-target SBR split. Delete this cache file (or pass "
              "a new --sbr-cache name) to pay for a fresh SBR run that includes the split.")
else:
    print(f"running dense SBR (return_components=True, max_bounces=2, split_order2_by_target={need_split_order2}, "
          f"leg2_retro_check={args.sbr_leg2_retro_check})...")
    t0 = time.perf_counter()
    s_sbr, sbr_stats = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                            max_bounces=2, aim_is_direction=False,
                                            return_components=True, progress=True,
                                            split_order2_by_target=need_split_order2,
                                            leg2_retro_check=args.sbr_leg2_retro_check,
                                            retro_beamwidth_mult=args.retro_beamwidth_mult)
    t_sbr = time.perf_counter() - t0
    print(f"  {t_sbr:.1f}s wall, counts={sbr_stats['counts']}")
    order1_sbr = sbr_stats['s_by_order']['order1']
    order2_sbr = sbr_stats['s_by_order']['order2']
    save_kwargs = dict(order1=to_numpy(order1_sbr), order2=to_numpy(order2_sbr), t_sbr_s=t_sbr,
                        sbr_leg2_retro_check=args.sbr_leg2_retro_check,
                        retro_beamwidth_mult=args.retro_beamwidth_mult)
    if need_split_order2:
        order2_ground_sbr = sbr_stats['s_by_order']['order2_ground']
        order2_building_sbr = sbr_stats['s_by_order']['order2_building']
        have_order2_split = True
        save_kwargs['order2_ground'] = to_numpy(order2_ground_sbr)
        save_kwargs['order2_building'] = to_numpy(order2_building_sbr)
    if args.sbr_cache:
        np.savez(args.sbr_cache, **save_kwargs)
        print(f"  cached to {args.sbr_cache}")

if on_gpu:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("  freed cupy memory pool before closed-form stage")

print(f"running closed form (box-projected, return_components=True, leg2_occlusion_check={args.leg2_occlusion}, "
      f"split_leg2_by_target={args.split_leg2}, leg2_retroreflection_check={args.leg2_retro_check}, "
      f"leg2_retro_taper={args.leg2_retro_taper}, leg2_building_enabled={not args.leg2_ground_only})...")
t0 = time.perf_counter()
s_asc, asc_stats = run_asc_box_projected_multibounce(
    xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
    ground_material=args.ground_material, return_components=True, include_order3=False,
    progress=True, leg2_occlusion_check=args.leg2_occlusion, split_leg2_by_target=args.split_leg2,
    leg2_retroreflection_check=args.leg2_retro_check, retro_beamwidth_mult=args.retro_beamwidth_mult,
    leg2_building_enabled=not args.leg2_ground_only, leg2_retro_taper=args.leg2_retro_taper)
t_asc = time.perf_counter() - t0
print(f"  {t_asc:.2f}s wall, counts={asc_stats['counts']}")
leg1_asc = asc_stats['s_by_leg']['leg1']
leg2_asc = asc_stats['s_by_leg']['leg2']


def raw_coherence(a, b):
    a, b = to_numpy(a), to_numpy(b)
    return float(np.abs(np.vdot(a.ravel(), b.ravel())) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def img_coherence(a, b):
    img_a = backproject(xp, a, plat, freqs, grid, grid, args.fc, R_ref)
    img_b = backproject(xp, b, plat, freqs, grid, grid, args.fc, R_ref)
    return float(complex_coherence(to_numpy(img_a), to_numpy(img_b))), img_a, img_b


print("\n=== per-leg/order coherence (this is the actual diagnostic) ===")
raw1 = raw_coherence(order1_sbr, leg1_asc)
raw2 = raw_coherence(order2_sbr, leg2_asc)
raw_combined = raw_coherence(order1_sbr + order2_sbr, leg1_asc + leg2_asc)
print(f"leg1/order1 (direct, single-bounce):  raw coherence = {raw1:.4f}")
print(f"leg2/order2 (double-bounce):           raw coherence = {raw2:.4f}")
print(f"combined (order1+2 vs leg1+2):         raw coherence = {raw_combined:.4f}")

img_coh1, img1_sbr, img1_asc = img_coherence(order1_sbr, leg1_asc)
img_coh2, img2_sbr, img2_asc = img_coherence(order2_sbr, leg2_asc)
img_coh_combined, img_c_sbr, img_c_asc = img_coherence(order1_sbr + order2_sbr, leg1_asc + leg2_asc)
print(f"leg1/order1 image coherence: {img_coh1:.4f}")
print(f"leg2/order2 image coherence: {img_coh2:.4f}")
print(f"combined image coherence:    {img_coh_combined:.4f}")

if args.split_leg2:
    leg2_ground = asc_stats['s_by_leg']['leg2_ground']
    leg2_building = asc_stats['s_by_leg']['leg2_building']
    n_ground = asc_stats['counts'].get('leg2_ground', 0)
    n_building = asc_stats['counts'].get('leg2_building', 0)
    e_ground = float(np.linalg.norm(to_numpy(leg2_ground)) ** 2)
    e_building = float(np.linalg.norm(to_numpy(leg2_building)) ** 2)
    e_total = e_ground + e_building + 1e-300
    print(f"\n=== leg2 split: ground (true 90-deg corner) vs building (arbitrary pair) ===")
    if have_order2_split:
        ref_ground, ref_building = order2_ground_sbr, order2_building_sbr
        print("comparing against SBR's OWN ground/building order2 split (apples-to-apples, not the mixed signal)")
    else:
        ref_ground, ref_building = order2_sbr, order2_sbr
        print("comparing against the FULL/mixed order2 signal (SBR-side split unavailable this run -- "
              "see cache warning above if any) -- imperfect: SBR's order2 includes real wall-to-wall "
              "energy leg2_ground never attempts to represent, and real wall-to-ground energy "
              "leg2_building never attempts to represent, so both numbers are understated relative "
              "to a true like-for-like comparison.")
    print(f"leg2_ground:   {n_ground} contributions, {100*e_ground/e_total:.1f}% of leg2 energy, "
          f"raw coherence vs {'order2_ground' if have_order2_split else 'order2 (mixed)'} = "
          f"{raw_coherence(ref_ground, leg2_ground):.4f}")
    print(f"leg2_building: {n_building} contributions, {100*e_building/e_total:.1f}% of leg2 energy, "
          f"raw coherence vs {'order2_building' if have_order2_split else 'order2 (mixed)'} = "
          f"{raw_coherence(ref_building, leg2_building):.4f}")

if raw2 < raw1 - 0.02:
    print("\n-> leg2 is measurably worse than leg1: consistent with the leg2/leg3 "
          "missing-occlusion-check hypothesis (but see leg1 number above before concluding "
          "leg1 is clean -- it has no occlusion check either, it just may matter less at this density/seed).")
elif raw1 < raw2 - 0.02:
    print("\n-> leg1 is measurably worse than leg2 -- NOT what the occlusion-check hypothesis "
          "predicts. Worth looking at leg1's own missing occlusion check, or something else entirely "
          "(e.g. the ground-clutter term, or a scene-density-dependent geometry issue).")
else:
    print("\n-> leg1 and leg2 degraded roughly TOGETHER -- doesn't cleanly implicate leg2 specifically. "
          "Consider whether both share a cause (e.g. both lack occlusion checking against other "
          "buildings), or whether the drop is coming from somewhere this decomposition doesn't isolate "
          "(ground clutter, backprojection grid/resolution effects, etc).")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for row, (sbr_img, asc_img, label, coh) in enumerate([
        (img1_sbr, img1_asc, 'leg1/order1 (direct)', img_coh1),
        (img2_sbr, img2_asc, 'leg2/order2 (double-bounce)', img_coh2)]):
    db_sbr = 20 * np.log10(np.abs(to_numpy(sbr_img)) + 1e-12); db_sbr -= db_sbr.max()
    db_asc = 20 * np.log10(np.abs(to_numpy(asc_img)) + 1e-12); db_asc -= db_asc.max()
    db_diff = db_sbr - db_asc
    for col, (db, title) in enumerate([(db_sbr, f'SBR {label}'), (db_asc, f'ASC {label}'),
                                        (db_diff, f'diff (coh={coh:.3f})')]):
        ax = axes[row, col]
        vmin, vmax = (-40, 0) if col < 2 else (-10, 10)
        cmap = 'gray' if col < 2 else 'RdBu_r'
        im = ax.imshow(db, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                        extent=[-args.footprint/2, args.footprint/2, -args.footprint/2, args.footprint/2])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"SBR vs ASC, decomposed by leg/order -- {facets_b['n_buildings']} buildings, "
             f"{n_facets} facets, footprint={args.footprint}m, density={args.density}/km2")
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(f"\nsaved {args.out}")
