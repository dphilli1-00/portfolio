"""
Tier 2 validation, multi-bounce edition: same real collection geometry,
same scene size/density, and the same scoring/plotting infrastructure as
validation_tier2_compare.py (imported, not duplicated) -- but the OLD
script's forward models were single-bounce only (make_building_scene's
undivided 5-facet-per-building mesh, run_dense_sbr_timed, run_asc_cached).
This one swaps in the multi-bounce machinery built since then:

  - make_multi_building_scene_adaptive instead of make_building_scene --
    each wall is height-partitioned per its own range sensitivity, not one
    undivided facet. Facet count will be noticeably HIGHER than the old
    slide's "1000 facets" for the same 200 buildings/1km^2 as a direct
    result (that's the fix, not a regression -- see
    adaptive_facet_partition.py for why the undivided facet was wrong).
  - run_multibounce_sbr (max_bounces=2, order3 optional via --order3)
    instead of run_dense_sbr_timed -- real ray-traced wall-ground AND
    wall-wall double bounces, not single-bounce only.
  - run_asc_box_projected_multibounce instead of run_asc_cached -- the
    closed-form counterpart: reflects each facet's own ray off ITS
    normal and intersects the nearest of {ground plane, another
    building's box} analytically, no ray tracing.

Default args reproduce the exact geometry behind the original two Tier 2
slides: --standoff 60000 --altitude 22500 --fc 10e9 --az-res 2 (which
auto-bumps --pulses to the Nyquist minimum for this standoff/wavelength/
resolution -- 450, matching the slide's "450-pulse Nyquist-sampled
aperture" exactly) --footprint 1000 --density 200 --seed 0 (same building
layout RNG seed as the original slide's make_building_scene(..., seed=0)
call, though the adaptive generator's own per-wall partitioning means the
facet-level geometry still differs even at the same seed).

This is a big run (200+ buildings, a wall-ground/wall-wall double-bounce
combinatorial search per pulse) -- meant to run locally on your own GPU
(pass --gpu), not in a sandboxed environment. Start with --rays and
--pulses well below the defaults to sanity-check the pipeline runs
end-to-end before committing to the full-size run; timings are printed
at every stage so you can extrapolate before waiting on the big one.

Usage:
    python3 run_tier2_multibounce.py --gpu
    python3 run_tier2_multibounce.py --gpu --rays 40 --pulses 60   # quick smoke test
    python3 run_tier2_multibounce.py --gpu --order3                # add the (slow, weakly
                                                                     # validated -- see
                                                                     # box_projected_multibounce.py)
                                                                     # third bounce
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import (get_backend, C, make_aim_grid, make_ground_facet,
                             make_ground_clutter, ground_clutter_phase, compute_layover_margin)
from multibounce_demo import run_multibounce_sbr
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import (backproject, to_numpy, compute_ref_ranges, complex_coherence,
                                       coherence_map, required_freq_samples, per_building_ssim)

plt.close('all')
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=1000.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dr-max', type=float, default=3.0,
                     help='adaptive height-partition target (m of equivalent-range change '
                          'between adjacent sub-facets) -- smaller = more facets = finer but slower')
    ap.add_argument('--rays', type=int, default=80, help='SBR aim-grid side length (rays = this^2)')
    ap.add_argument('--pulses', type=int, default=100)
    ap.add_argument('--img-size', type=int, default=200)
    ap.add_argument('--standoff', type=float, default=60000.0)
    ap.add_argument('--altitude', type=float, default=22500.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    ap.add_argument('--az-res', type=float, default=2.0,
                     help='desired cross-range resolution (m); aperture length and the Nyquist '
                          '--pulses floor are both derived from this + standoff + wavelength, same '
                          'as validation_tier2_compare.py. Pass 0/omit to keep a fixed 400m aperture '
                          'and use --pulses as-is instead.')
    ap.add_argument('--ground-material', type=str, default='dry_soil')
    ap.add_argument('--order3', action='store_true',
                     help='include the third bounce (slow, and only weakly validated against SBR '
                          'so far -- see box_projected_multibounce.py docstring. Confirmed to move '
                          'combined-image SSIM by ~0.002 on a smaller test scene, i.e. immaterial to '
                          'the numbers this script reports -- default off to save the runtime.')
    ap.add_argument('--no-clutter', action='store_true')
    args = ap.parse_args()

    t_start = time.perf_counter()
    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

    standoff, altitude = args.standoff, args.altitude
    o_broadside = np.array([0.0, -standoff, altitude])

    t0 = time.perf_counter()
    facets_b = make_multi_building_scene_adaptive(
        xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
        dR_max=args.dr_max, min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
    facets_g = make_ground_facet(xp, args.footprint, material=args.ground_material)
    print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} building facets "
          f"(vs. the original single-bounce slide's undivided ~{facets_b['n_buildings']*5} -- "
          f"adaptive height partitioning legitimately adds facets, see module docstring)")
    print(f"scene build: {time.perf_counter()-t0:.2f}s")

    wavelength = C / args.fc
    if args.az_res:
        d_theta = wavelength / (2.0 * args.az_res)
        squint_len = d_theta * standoff
        min_pulses = int(np.ceil(standoff * wavelength / (args.az_res ** 2)))
        edge_deg = (args.footprint / standoff) * 180.0 / np.pi
        print(f"--az-res {args.az_res}m -> aperture={squint_len:.0f}m, "
              f"edge-to-edge incidence-angle swing over the scene ~{edge_deg:.2f} deg")
        if args.pulses < min_pulses:
            print(f"--pulses {args.pulses} below Nyquist minimum {min_pulses} for this aperture -- "
                  f"auto-bumping.")
            args.pulses = min_pulses
    else:
        squint_len = 400.0

    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([u, np.full(args.pulses, -standoff), np.full(args.pulses, altitude)]))

    t0 = time.perf_counter()
    aim_pts, margin = make_aim_grid(xp, args.footprint, args.rays, standoff, altitude, max_height_m=40.0)
    print(f"aim grid: {aim_pts.shape[0]} rays ({args.rays}x{args.rays}), "
          f"padded +/-{margin:.1f}m beyond footprint for roof layover ({time.perf_counter()-t0:.2f}s)")

    ref_pos = xp.zeros(3)
    grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))
    R_ref = compute_ref_ranges(xp, plat, ref_pos)

    min_freq, maxdr, needed_window = required_freq_samples(
        to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
    n_freq = int(np.ceil(min_freq * 1.05))
    print(f"required_freq_samples: min_freq={min_freq} -> n_freq={n_freq} "
          f"(range window {needed_window:.0f}m at {args.bandwidth/1e6:.0f}MHz)")
    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, n_freq))
    print(f"setup total: {time.perf_counter()-t_start:.2f}s\n")

    max_bounces = 3 if args.order3 else 2
    print(f"[1/2] dense SBR (max_bounces={max_bounces})...")
    t0 = time.perf_counter()
    s_sbr, sbr_stats = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                            max_bounces=max_bounces, return_components=False,
                                            progress=True)
    print(f"  {time.perf_counter()-t0:.1f}s wall ({sbr_stats['t_per_pulse_ms']:.2f} ms/pulse), "
          f"counts={sbr_stats['counts']}")

    print(f"[2/2] point-ASC (box-projected{' + order3' if args.order3 else ''})...")
    t0 = time.perf_counter()
    s_asc, asc_stats = run_asc_box_projected_multibounce(
        xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
        ground_material=args.ground_material, return_components=False, include_order3=args.order3)
    print(f"  {time.perf_counter()-t0:.1f}s wall ({asc_stats['t_per_pulse_ms']:.2f} ms/pulse)")

    speedup_per_pulse = sbr_stats['t_per_pulse_ms'] / max(asc_stats['t_per_pulse_ms'], 1e-9)
    speedup_total = sbr_stats['t_total_s'] / max(asc_stats['t_total_s'], 1e-9)
    print(f"\n=== measured speedup ===")
    print(f"  per-pulse: {speedup_per_pulse:.0f}x  ({sbr_stats['t_per_pulse_ms']:.2f} ms vs "
          f"{asc_stats['t_per_pulse_ms']:.3f} ms)")
    print(f"  total:     {speedup_total:.0f}x  ({sbr_stats['t_total_s']:.1f}s vs {asc_stats['t_total_s']:.2f}s)")

    if not args.no_clutter:
        clutter_pts = make_ground_clutter(args.footprint, material='concrete', seed=1)
        print(f"\n+ {len(clutter_pts)} concrete ground-clutter points, shared identically by both branches")
        s_clutter = ground_clutter_phase(xp, clutter_pts, plat, freqs, ref_pos)
        s_sbr = s_sbr + s_clutter
        s_asc = s_asc + s_clutter

    print("\n[3/4] Image formation (backprojection), both branches...")
    t0 = time.perf_counter()
    img_sbr = backproject(xp, s_sbr, plat, freqs, grid, grid, args.fc, R_ref)
    img_asc = backproject(xp, s_asc, plat, freqs, grid, grid, args.fc, R_ref)
    print(f"  {time.perf_counter()-t0:.2f}s for both images")

    mag_sbr = np.abs(to_numpy(img_sbr))
    mag_asc = np.abs(to_numpy(img_asc))
    phase_sbr = np.angle(to_numpy(img_sbr))
    phase_asc = np.angle(to_numpy(img_asc))
    db_sbr = 20 * np.log10(mag_sbr / (mag_sbr.max() + 1e-12) + 1e-6)
    db_asc = 20 * np.log10(mag_asc / (mag_asc.max() + 1e-12) + 1e-6)

    print("[4/4] Scoring: SSIM + amplitude/phase RMS error...")
    ssim_score = ssim(db_sbr, db_asc, data_range=40.0)
    amp_rms = np.sqrt(np.mean((mag_sbr / (mag_sbr.max() + 1e-12) - mag_asc / (mag_asc.max() + 1e-12)) ** 2))
    dphase = np.angle(np.exp(1j * (phase_sbr - phase_asc)))
    w = mag_sbr / (mag_sbr.sum() + 1e-12)
    phase_rms = np.sqrt(np.sum(w * dphase ** 2))
    phase_bias = np.sum(w * dphase)
    sig_mask_whole = mag_sbr > 0.05 * (mag_sbr.max() + 1e-12)
    phase_max = np.abs(dphase[sig_mask_whole]).max() if sig_mask_whole.any() else 0.0
    coherence = complex_coherence(to_numpy(img_sbr), to_numpy(img_asc))

    pass_ssim = ssim_score >= 0.95
    print(f"\n=== Tier 2 (multi-bounce) result -- whole image ===")
    print(f"SSIM:            {ssim_score:.4f}   ({'PASS' if pass_ssim else 'FAIL'} vs. >= 0.95)")
    print(f"Amplitude RMS:   {amp_rms:.4f}")
    print(f"Phase RMS:       {np.degrees(phase_rms):.2f} deg")
    print(f"Phase bias:      {np.degrees(phase_bias):+.2f} deg")
    print(f"Coherence:       {coherence:.4f}")

    print(f"\n=== Tier 2 (multi-bounce) result -- per building ===")
    grid_np = to_numpy(grid)
    per_bldg = per_building_ssim(facets_b, db_sbr, db_asc, mag_sbr, mag_asc,
                                  phase_sbr, phase_asc, grid_np, standoff, altitude)
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

    # --- figure 1: same 4-panel layout as validation_tier2_compare.py ---
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, db, title in zip(axes[:2], [db_sbr, db_asc], ['Dense SBR (multi-bounce)', 'ASC (box-projected)']):
        im = ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower',
                        extent=[-args.footprint / 2, args.footprint / 2] * 2)
        ax.set_title(title); ax.set_xlabel('cross-range (m)'); ax.set_ylabel('range (m)')
    diff = db_sbr - db_asc
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower',
                          extent=[-args.footprint / 2, args.footprint / 2] * 2)
    axes[2].set_title(f'Difference (dB)\nwhole-image SSIM={ssim_score:.3f}')
    axes[2].set_xlabel('cross-range (m)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    if len(scored) > 0:
        axes[3].hist(b_ssim, bins=min(20, max(5, len(scored)//3)), color='#0B7285', edgecolor='white')
        axes[3].axvline(0.95, color='#B85C00', linestyle='--', linewidth=1.5, label='0.95 criterion')
        axes[3].set_title(f'Per-building SSIM\n{n_pass_b}/{len(scored)} pass '
                           f'({100*n_pass_b/len(scored):.0f}%), mean={b_ssim.mean():.3f}')
        axes[3].set_xlabel('SSIM'); axes[3].set_ylabel('buildings'); axes[3].legend(fontsize=8)
    fig.suptitle(f'Tier 2 (multi-bounce): {facets_b["n_buildings"]} buildings, '
                 f'{args.footprint:.0f}m x {args.footprint:.0f}m, {args.pulses} pulses, '
                 f'{"order1+2+3" if args.order3 else "order1+2"}')
    fig.tight_layout()
    fig.savefig('tier2_multibounce_comparison.png', dpi=150)
    print("\nSaved tier2_multibounce_comparison.png")

    # --- figure 2: same phase/coherence layout ---
    fig2, ax2 = plt.subplots(1, 4, figsize=(19, 5.2))
    coh_map = coherence_map(img_sbr, img_asc, win=7)
    im3 = ax2[0].imshow(coh_map, cmap='viridis', vmin=0, vmax=1, origin='lower',
                            extent=[-args.footprint / 2, args.footprint / 2] * 2)
    ax2[0].set_title(f'Coherence map (7x7)\nwhole-image coherence={coherence:.3f}')
    ax2[0].set_xlabel('cross-range (m)'); ax2[0].set_ylabel('range (m)')
    plt.colorbar(im3, ax=ax2[0], fraction=0.046)

    dphase_deg = np.degrees(dphase)
    dphase_masked = np.where(sig_mask_whole, dphase_deg, np.nan)
    im4 = ax2[1].imshow(dphase_masked, cmap='twilight_shifted', vmin=-20, vmax=20, origin='lower',
                            extent=[-args.footprint / 2, args.footprint / 2] * 2)
    ax2[1].set_title(f'Phase difference (deg), masked >5% peak\n'
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
        ax2[3].set_title(f'Per-building phase RMS\nmean={b_phase_rms.mean():.2f}deg, max={b_phase_rms.max():.2f}deg')
        ax2[3].set_xlabel('phase RMS (deg)'); ax2[3].set_ylabel('buildings')

    fig2.suptitle(f'Tier 2 (multi-bounce) phase/coherence: {facets_b["n_buildings"]} buildings, '
                  f'{args.footprint:.0f}m x {args.footprint:.0f}m, {args.pulses} pulses')
    fig2.tight_layout()
    fig2.savefig('tier2_multibounce_phase_coherence.png', dpi=150)
    print("Saved tier2_multibounce_phase_coherence.png")

    result = dict(
        footprint_m=args.footprint, density_per_km2=args.density, seed=args.seed, dr_max=args.dr_max,
        n_buildings=facets_b['n_buildings'], n_facets=facets_b['n_facets'],
        n_pulses=args.pulses, n_rays=aim_pts.shape[0], max_bounces=max_bounces, order3=args.order3,
        sbr_t_total_s=sbr_stats['t_total_s'], sbr_ms_per_pulse=sbr_stats['t_per_pulse_ms'],
        sbr_counts=sbr_stats['counts'],
        asc_t_total_s=asc_stats['t_total_s'], asc_ms_per_pulse=asc_stats['t_per_pulse_ms'],
        speedup_per_pulse=speedup_per_pulse, speedup_total=speedup_total,
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
    with open('tier2_multibounce_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved tier2_multibounce_result.json")
    print(f"\ntotal script time: {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
