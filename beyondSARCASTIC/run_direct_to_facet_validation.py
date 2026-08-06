"""
Dense-SBR-vs-closed-form validation, direct-to-facet edition.

Same comparison run_tier2_multibounce.py does (run_multibounce_sbr vs
run_asc_box_projected_multibounce, backprojected + scored), but with the
two fixes from this session's work swapped in:

  - Dense SBR aims one ray directly at every known facet's own center
    (aim_pts=facets_b['center'], aim_is_direction=False) instead of an
    exploratory ground-plane or angular grid. make_aim_grid plateaued at
    ~14% facet-discovery coverage on a height-subdivided scene regardless
    of ray density (a real, measured ceiling -- grazing-incidence walls
    couple azimuth and elevation together, so no independent az x el grid
    samples them efficiently). Aiming directly at each known facet makes
    coverage exhaustive by construction (measured 99.9% on a 6-building
    test scene) and gives a REAL ray-traced occlusion verdict per facet
    (was this facet's own straight-line sightline blocked by some other
    facet first?) -- independent of the closed form's own analytic-
    geometry occlusion path, which is the whole point of a "ground truth"
    reference.

  - make_multi_building_scene_adaptive is called with --delta-cr and
    --wavelength (derived from --fc) in addition to --dr-max, so walls
    get subdivided along their LENGTH too, not just height -- the joint
    range + cross-range + Fresnel criterion (adaptive_facet_partition.py).
    run_tier2_multibounce.py doesn't pass these, so it's still running
    height-only partitioning. That's *why* direct-to-facet aiming is
    valid here: "one ray, scored at that facet's center" is only a
    resolution-cell-accurate approximation once the facet is actually
    sized to its own resolution cell on every axis, not just height.

Real cost tradeoff, budget for it: direct-to-facet SBR cost scales with
facet count (O(facets) rays x AABB-culled intersection tests), not a
fixed ray budget. Measured on this session's hardware: ~0.43s/pulse
(order1+2+3) at 981 facets, ~22s/pulse (order1 ALONE) at 13,227 facets.
That's why this script's default scene is much smaller than
run_tier2_multibounce.py's 1000m/200-density production default --
start small, check the printed cost estimate, then scale --footprint/
--density up deliberately. This script also writes its own output
filename (not tier2_multibounce_result.json) specifically so it can
never collide with or overwrite the original slide numbers again.

Usage:
    python3 run_direct_to_facet_validation.py
    python3 run_direct_to_facet_validation.py --gpu --footprint 300 --density 200
    python3 run_direct_to_facet_validation.py --footprint 60 --density 600 --pulses 5   # quick check
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_facet, make_ground_clutter, ground_clutter_phase
from multibounce_demo import run_multibounce_sbr
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import (backproject, to_numpy, compute_ref_ranges, complex_coherence,
                                       required_freq_samples)

plt.close('all')


def estimate_sbr_cost(n_facets):
    """Rough linear extrapolation from two measured points this session:
    ~0.43s/pulse (order1+2+3) @ 981 facets, ~22s/pulse (order1 ONLY)
    @ 13,227 facets. Order1-only is the dominant, most comparable term at
    scale (order2/3 add on top but are smaller since they only continue
    from a subset of order1's hits) -- treated as a floor, not the whole
    story. This is a heads-up, not a precise model: actual cost also
    depends on how clustered the buildings are (AABB culling works better
    when a ray-chunk's angular footprint only spans a few buildings)."""
    per_facet = 22.0 / 13227
    return per_facet * n_facets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=150.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dr-max', type=float, default=2.0,
                     help='range-resolution partition target (m), criterion 1')
    ap.add_argument('--delta-cr', type=float, default=2.0,
                     help='cross-range-resolution partition target (m), criterion 2 -- '
                          'match to --az-res for a self-consistent partition')
    ap.add_argument('--pulses', type=int, default=8)
    ap.add_argument('--img-size', type=int, default=150)
    ap.add_argument('--standoff', type=float, default=8000.0)
    ap.add_argument('--altitude', type=float, default=3000.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    ap.add_argument('--az-res', type=float, default=2.0)
    ap.add_argument('--ground-material', type=str, default='dry_soil')
    ap.add_argument('--order3', action='store_true')
    ap.add_argument('--no-clutter', action='store_true')
    ap.add_argument('--skip-confirm', action='store_true',
                     help='skip the printed cost estimate + Enter-to-continue gate')
    ap.add_argument('--out', type=str, default='direct_to_facet_validation_result.json')
    ap.add_argument('--sbr-cache', type=str, default=None,
                     help='path to a .npz previously saved by this script (see the auto-saved '
                          '"<out>.sbr_cache.npz") -- if given and the file exists, SKIPS the '
                          'expensive dense-SBR stage entirely and loads s_sbr/sbr_stats from disk '
                          'instead. Use this to re-run just the closed-form comparison (e.g. after '
                          'a crash there) without paying for SBR again.')
    ap.add_argument('--skip-sbr', action='store_true',
                     help='skip the dense-SBR stage entirely -- no run, no cache lookup. Runs '
                          'closed form alone: no coherence/speedup numbers (nothing to compare '
                          'against), just the closed-form image and its own timing. Useful for '
                          'checking the closed form runs cleanly / getting its own timing on a '
                          'big scene without paying for SBR at all. Takes priority over --sbr-cache '
                          'if both are given.')
    ap.add_argument('--low-precision-envelope', action='store_true',
                     help='float32/complex64 for the closed-form envelope instead of float64/'
                          'complex128 -- see asc_amplitude_envelope\'s low_precision docstring. '
                          'Coherence-verified (~1e-7 relative error) but NOT re-checked against '
                          'this codebase\'s float64-measured canonical-shape validation numbers. '
                          'Measured no speedup on CPU; GPU (memory-bandwidth-bound) is the case '
                          'this is actually for -- unverified on real GPU hardware, check it '
                          'yourself before trusting it.')
    ap.add_argument('--profile', action='store_true',
                     help='time pulse 0 of the SBR run stage-by-stage (ray cast vs scoring, '
                          'per bounce order) and print a breakdown before continuing -- see '
                          'run_multibounce_sbr\'s profile_first_pulse docstring. Use this on a '
                          'small scene first if a run is much slower than expected, instead of '
                          'guessing where the time is going.')
    args = ap.parse_args()

    t_start = time.perf_counter()
    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

    standoff, altitude = args.standoff, args.altitude
    o_broadside = np.array([0.0, -standoff, altitude])
    wavelength = C / args.fc

    t0 = time.perf_counter()
    facets_b = make_multi_building_scene_adaptive(
        xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
        dR_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
        min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
    facets_g = make_ground_facet(xp, args.footprint, material=args.ground_material)
    n_facets = facets_b['n_facets']
    print(f"{facets_b['n_buildings']} buildings, {n_facets} facets "
          f"(2-D length x height adaptive partition -- see adaptive_facet_partition.py)")
    print(f"scene build: {time.perf_counter()-t0:.2f}s")

    # Nyquist pulse-count bump happens BEFORE the cost estimate now --
    # it used to happen after, so the estimate (and the confirm gate
    # built on it) quoted whatever --pulses the user passed in, not the
    # actual bumped count the run below uses. Measured impact: a real
    # run printed "~8.6 min" for 8 pulses, then silently bumped to 60
    # and actually took ~7.5x longer than advertised -- exactly the kind
    # of thing this gate exists to prevent.
    if args.az_res:
        d_theta = wavelength / (2.0 * args.az_res)
        squint_len = d_theta * standoff
        min_pulses = int(np.ceil(standoff * wavelength / (args.az_res ** 2)))
        if args.pulses < min_pulses:
            print(f"--pulses {args.pulses} below Nyquist minimum {min_pulses} -- auto-bumping.")
            args.pulses = min_pulses
    else:
        squint_len = 400.0

    est_per_pulse = estimate_sbr_cost(n_facets)
    print(f"\nRough dense-SBR cost estimate: ~{est_per_pulse:.1f}s/pulse x {args.pulses} pulses "
          f"(post-Nyquist-bump count) = ~{est_per_pulse*args.pulses/60:.1f} min for SBR alone "
          f"(closed form is ~2 orders of magnitude faster, negligible by comparison). This rate "
          f"is extrapolated from CPU-only measurements (see estimate_sbr_cost's docstring) -- "
          f"if you're on {'GPU (cupy)' if on_gpu else 'CPU'}, treat this as a rough prior, not a "
          f"calibrated number; no GPU timing data has gone into it.")
    if not args.skip_confirm:
        input("Press Enter to continue, or Ctrl+C to abort and re-run with a smaller "
              "--footprint/--density first... ")

    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([u, np.full(args.pulses, -standoff), np.full(args.pulses, altitude)]))
    ref_pos = xp.zeros(3)
    grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))
    R_ref = compute_ref_ranges(xp, plat, ref_pos)

    min_freq, maxdr, needed_window = required_freq_samples(
        to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
    n_freq = int(np.ceil(min_freq * 1.05))
    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, n_freq))
    print(f"n_freq={n_freq}, n_pulses={args.pulses}\nsetup total: {time.perf_counter()-t_start:.2f}s\n")

    # ---- direct-to-facet aim points: one ray per known facet, not an
    # exploratory grid (see module docstring) ----
    aim_pts = facets_b['center']

    sbr_cache_path = args.sbr_cache or (os.path.splitext(args.out)[0] + '.sbr_cache.npz')

    max_bounces = 3 if args.order3 else 2
    if args.skip_sbr:
        print("[1/2] --skip-sbr: not running dense SBR. No coherence/speedup numbers this run.")
        s_sbr, sbr_stats, t_sbr = None, None, None
    elif args.sbr_cache and os.path.exists(args.sbr_cache):
        print(f"[1/2] loading cached dense-SBR result from {args.sbr_cache} (skipping SBR run)...")
        cached = np.load(args.sbr_cache, allow_pickle=True)
        s_sbr = xp.asarray(cached['s_sbr'])
        sbr_stats = dict(counts=cached['counts'].item())
        t_sbr = float(cached['t_sbr_s'])
        print(f"  loaded ({t_sbr:.1f}s original wall time), counts={sbr_stats['counts']}")
    else:
        print(f"[1/2] dense SBR, direct-to-facet (max_bounces={max_bounces})...")
        t0 = time.perf_counter()
        s_sbr, sbr_stats = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                                max_bounces=max_bounces, aim_is_direction=False,
                                                return_components=False, progress=True,
                                                profile_first_pulse=args.profile)
        t_sbr = time.perf_counter() - t0
        print(f"  {t_sbr:.1f}s wall ({1000*t_sbr/args.pulses:.1f} ms/pulse), counts={sbr_stats['counts']}, "
              f"coverage={sbr_stats['counts']['order1']/(n_facets*args.pulses)*100:.1f}% of "
              f"(facets x pulses) for order1")

        # Checkpoint immediately -- an SBR run at this scale can take
        # 10s of minutes; a crash in the NEXT stage (closed form, or
        # backprojection) shouldn't cost you that work a second time.
        # Real motivating case: a GPU OOM in run_asc_box_projected_
        # multibounce killed a run right after a 20-minute SBR stage
        # finished cleanly, because cupy's default memory pool was still
        # holding SBR's cached (but unused) allocations and left no
        # headroom for the closed form's own (F, K)-shaped arrays.
        np.savez(sbr_cache_path, s_sbr=to_numpy(s_sbr), counts=sbr_stats['counts'], t_sbr_s=t_sbr)
        print(f"  checkpointed to {sbr_cache_path} (--sbr-cache {sbr_cache_path} to reuse)")

    if on_gpu:
        # Release cupy's cached (but unused) device memory before the
        # next stage allocates its own large arrays -- the pool doesn't
        # give memory back to the driver on its own between calls, so a
        # memory-heavy stage (SBR, here) can starve a later one even
        # after its own tensors have gone out of scope. See the
        # checkpointing comment above for the OOM this caused.
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("  freed cupy memory pool before closed-form stage")

    print(f"[2/2] closed form (box-projected{' + order3' if args.order3 else ''})...")
    t0 = time.perf_counter()
    s_asc, asc_stats = run_asc_box_projected_multibounce(
        xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
        ground_material=args.ground_material, return_components=False, include_order3=args.order3,
        progress=True, low_precision_envelope=args.low_precision_envelope)
    t_asc = time.perf_counter() - t0
    print(f"  {t_asc:.2f}s wall ({1000*t_asc/args.pulses:.3f} ms/pulse)")

    have_sbr = s_sbr is not None
    speedup_total = (t_sbr / max(t_asc, 1e-9)) if have_sbr else None
    raw_coherence = None
    if have_sbr:
        print(f"\n=== measured speedup: closed form is {speedup_total:.0f}x faster than dense SBR ===")
        raw_coherence = float(np.abs(np.vdot(to_numpy(s_sbr).ravel(), to_numpy(s_asc).ravel()))
                               / (np.linalg.norm(to_numpy(s_sbr)) * np.linalg.norm(to_numpy(s_asc))))
        print(f"raw phase-history coherence (pre-backprojection): {raw_coherence:.4f}")

    if not args.no_clutter:
        clutter_pts = make_ground_clutter(args.footprint, material='concrete', seed=1)
        print(f"+ {len(clutter_pts)} concrete ground-clutter points, shared identically by both branches")
        s_clutter = ground_clutter_phase(xp, clutter_pts, plat, freqs, ref_pos)
        if have_sbr:
            s_sbr = s_sbr + s_clutter
        s_asc = s_asc + s_clutter

    print("\nbackprojecting...")
    t0 = time.perf_counter()
    img_asc = backproject(xp, s_asc, plat, freqs, grid, grid, args.fc, R_ref)
    img_sbr = backproject(xp, s_sbr, plat, freqs, grid, grid, args.fc, R_ref) if have_sbr else None
    print(f"  {time.perf_counter()-t0:.2f}s")

    img_coh = None
    if have_sbr:
        img_coh = float(complex_coherence(to_numpy(img_sbr), to_numpy(img_asc)))
        print(f"backprojected-image complex coherence: {img_coh:.4f}")

    if have_sbr:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        panels = [(axes[0], img_sbr, 'dense SBR (direct-to-facet)'),
                  (axes[1], img_asc, 'closed form (box-projected ASC)')]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        panels = [(ax, img_asc, 'closed form (box-projected ASC) -- SBR skipped, no comparison')]
    for ax, img, title in panels:
        db = 20 * np.log10(np.abs(to_numpy(img)) + 1e-12)
        db -= db.max()
        ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower',
                  extent=[-args.footprint/2, args.footprint/2, -args.footprint/2, args.footprint/2])
        ax.set_title(title)
    if have_sbr:
        fig.suptitle(f"direct-to-facet validation: raw coherence={raw_coherence:.3f}, "
                     f"image coherence={img_coh:.3f}, {speedup_total:.0f}x speedup")
    else:
        fig.suptitle(f"closed form only (--skip-sbr): {t_asc:.2f}s / {args.pulses} pulses")
    fig.tight_layout()
    png_path = os.path.splitext(args.out)[0] + '.png'
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")

    result = dict(
        footprint=args.footprint, density=args.density, seed=args.seed,
        dr_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
        n_buildings=facets_b['n_buildings'], n_facets=n_facets,
        n_pulses=args.pulses, n_freq=n_freq,
        sbr_counts=(sbr_stats['counts'] if have_sbr else None), t_sbr_s=t_sbr, t_asc_s=t_asc,
        speedup=speedup_total, raw_coherence=raw_coherence, image_coherence=img_coh,
    )
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
