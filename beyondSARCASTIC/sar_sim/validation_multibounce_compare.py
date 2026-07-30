"""
Tier-2-style validation, but for the DOUBLE-BOUNCE term specifically, not
the already-validated single bounce.

Why this exists: run_asc_cached_multibounce's closed-form wall-ground
bounce has no occlusion/shadow-ray check at all (it can't, and stay
O(facets) -- see its docstring), while run_multibounce_sbr's ray-traced
version explicitly checks return-path visibility before counting a path.
A raw path-COUNT comparison already showed ASC finding ~3x more candidate
double-bounce paths than the ray-traced ground truth on the same scene/
pulses -- this script quantifies what that gap actually costs in image
fidelity (SSIM / amplitude-phase RMS / complex coherence), the same way
the original Tier 2 slide quantified the single-bounce ASC-cache gap,
instead of leaving "~3x more paths" as an ungrounded number.

Isolation, not the combined image: comparing the FULL (order1+order2)
image would bury this under the single-bounce term, which already agrees
almost perfectly (that's the existing, already-validated Tier 2 result)
-- the same "whole-image SSIM is diluted" lesson that pushed the original
validation toward per-building scoring. Both run_multibounce_sbr and
run_asc_cached_multibounce now support return_components=True specifically
so the order-2-only / leg2-only phase history can be pulled out and
imaged on its own.

No order-3 here: ASC has no closed-form triple-bounce term (see
run_asc_cached_multibounce's docstring for why), so there's nothing on
the ASC side to compare a ray-traced order-3 image against.

Usage:
    python3 validation_multibounce_compare.py --footprint 1000 --density 200 \
        --rays 50 --pulses 60 --freq 64 --img-size 150 --ground-material dry_soil
"""

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend, make_aim_grid, C
from multibounce_demo import run_multibounce_sbr, run_asc_cached_multibounce
from validation_tier2_compare import (backproject, complex_coherence, required_freq_samples,
                                       compute_ref_ranges, to_numpy)


def score_whole_image(img_a, img_b, data_range=40.0):
    mag_a = np.abs(to_numpy(img_a))
    mag_b = np.abs(to_numpy(img_b))
    phase_a = np.angle(to_numpy(img_a))
    phase_b = np.angle(to_numpy(img_b))

    db_a = 20 * np.log10(mag_a / (mag_a.max() + 1e-12) + 1e-6)
    db_b = 20 * np.log10(mag_b / (mag_b.max() + 1e-12) + 1e-6)
    ssim_score = ssim(db_a, db_b, data_range=data_range)

    amp_rms = float(np.sqrt(np.mean((mag_a / (mag_a.max() + 1e-12) - mag_b / (mag_b.max() + 1e-12)) ** 2)))
    dphase = np.angle(np.exp(1j * (phase_a - phase_b)))
    w = mag_a / (mag_a.sum() + 1e-12)
    phase_rms = float(np.degrees(np.sqrt(np.sum(w * dphase ** 2))))
    phase_bias = float(np.degrees(np.sum(w * dphase)))
    sig_mask = mag_a > 0.05 * (mag_a.max() + 1e-12)
    phase_max = float(np.degrees(np.abs(dphase[sig_mask]).max())) if sig_mask.any() else 0.0
    coherence = complex_coherence(to_numpy(img_a), to_numpy(img_b))

    return dict(ssim=float(ssim_score), amp_rms=amp_rms, phase_rms_deg=phase_rms,
                phase_bias_deg=phase_bias, phase_max_deg=phase_max, coherence=coherence), db_a, db_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=1000.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--rays', type=int, default=50)
    ap.add_argument('--pulses', type=int, default=60)
    ap.add_argument('--freq', type=int, default=64)
    ap.add_argument('--img-size', type=int, default=150)
    ap.add_argument('--standoff', type=float, default=8000.0)
    ap.add_argument('--altitude', type=float, default=3000.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    ap.add_argument('--ground-material', type=str, default='dry_soil',
                     choices=['dry_soil', 'concrete', 'metal'])
    ap.add_argument('--no-occlusion-check', action='store_true',
                     help='disable the cheap AABB occlusion check on the ASC ground bounce, '
                          'to compare against the occlusion-aware version')
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

    facets_b = make_building_scene(xp, args.footprint, args.density, seed=0)
    facets_g = make_ground_facet(xp, args.footprint, material=args.ground_material)
    print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} building facets "
          f"+ 1 ground facet ({args.ground_material})")

    squint_len = 400.0
    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([
        u, np.full(args.pulses, -args.standoff), np.full(args.pulses, args.altitude)
    ]))
    aim_pts, margin = make_aim_grid(xp, args.footprint, args.rays, args.standoff, args.altitude,
                                     max_height_m=40.0)

    ref_pos = xp.zeros(3)
    grid = xp.asarray(np.linspace(-args.footprint / 2, args.footprint / 2, args.img_size))

    min_freq, maxdr, needed_window = required_freq_samples(
        to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), args.bandwidth)
    if args.freq < min_freq:
        print(f"--freq {args.freq} below range-window minimum {min_freq} for this scene -- "
              f"auto-bumping (undersized K silently aliases instead of erroring).")
        args.freq = min_freq

    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, args.freq))
    R_ref = compute_ref_ranges(xp, plat, ref_pos)

    print("\n[1/4] Ray-traced ground truth: dense-SBR multi-bounce (order<=2)...")
    t0 = time.perf_counter()
    s_sbr, stats_sbr = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                            max_bounces=2, return_components=True)
    print(f"  {stats_sbr['t_per_pulse_ms']:.1f} ms/pulse, {time.perf_counter()-t0:.1f}s total, "
          f"counts={stats_sbr['counts']}")
    s_sbr_order2 = stats_sbr['s_by_order']['order2']

    occl = not args.no_occlusion_check
    print(f"[2/4] Closed-form: ASC-cached wall-ground bounce (occlusion_check={occl})...")
    t0 = time.perf_counter()
    s_asc, stats_asc = run_asc_cached_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                                   ground_material=args.ground_material,
                                                   return_components=True, occlusion_check=occl)
    if occl:
        print(f"  AABB occlusion check blocked {stats_asc['n_ground_bounce_occl_blocked']} candidate paths")
    print(f"  {stats_asc['t_per_pulse_ms']:.2f} ms/pulse, {time.perf_counter()-t0:.2f}s total, "
          f"n_ground_bounce_paths={stats_asc['n_ground_bounce_paths']}")
    s_asc_leg2 = stats_asc['s_by_leg']['leg2']

    n_sbr_order2 = stats_sbr['counts']['order2']
    n_asc_leg2 = stats_asc['n_ground_bounce_paths']
    overcount_ratio = n_asc_leg2 / max(n_sbr_order2, 1)
    print(f"\nPath count (same scene, same {args.pulses} pulses): "
          f"ray-traced order2={n_sbr_order2}  vs  ASC closed-form={n_asc_leg2}  "
          f"(ASC finds {overcount_ratio:.2f}x as many candidates)")

    print("\n[3/4] Image formation (backprojection) -- double-bounce term ONLY, isolated from order1...")
    img_sbr2 = backproject(xp, s_sbr_order2, plat, freqs, grid, grid, args.fc, R_ref)
    img_asc2 = backproject(xp, s_asc_leg2, plat, freqs, grid, grid, args.fc, R_ref)

    print("[4/4] Scoring...")
    scores, db_sbr2, db_asc2 = score_whole_image(img_sbr2, img_asc2)
    print(f"SSIM:          {scores['ssim']:.4f}")
    print(f"Amplitude RMS: {scores['amp_rms']:.4f} (normalized, 0=identical)")
    print(f"Phase RMS:     {scores['phase_rms_deg']:.2f} deg")
    print(f"Phase bias:    {scores['phase_bias_deg']:+.2f} deg")
    print(f"Phase max err: {scores['phase_max_deg']:.2f} deg")
    print(f"Coherence:     {scores['coherence']:.4f}")
    print(f"\nFor context, the ALREADY-VALIDATED single-bounce Tier 2 result (different script/run) "
          f"was SSIM=0.993 whole-image, coherence=0.989 whole-image -- compare against that scale, "
          f"not against 1.0, when judging whether {scores['ssim']:.3f}/{scores['coherence']:.3f} here "
          f"is 'good'.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    extent = [-args.footprint / 2, args.footprint / 2] * 2
    for ax, db, title in zip(axes[:2], [db_sbr2, db_asc2],
                              ['Dense-SBR order-2 only\n(ray-traced, occlusion-checked)',
                               'ASC leg2 only\n(closed-form, NO occlusion check)']):
        im = ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower', extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('cross-range (m)')
        ax.set_ylabel('range (m)')
    diff = db_sbr2 - db_asc2
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20, origin='lower', extent=extent)
    axes[2].set_title(f'Difference (dB)\nSSIM={scores["ssim"]:.3f}, coherence={scores["coherence"]:.3f}',
                       fontsize=10)
    axes[2].set_xlabel('cross-range (m)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f'Double-bounce fidelity: ray-traced (occlusion-checked) vs. ASC closed-form '
                 f'(AABB occlusion={"on" if occl else "off"}) -- {facets_b["n_buildings"]} buildings, '
                 f'{args.footprint:.0f}m x {args.footprint:.0f}m, {args.pulses} pulses, '
                 f'ground={args.ground_material}')
    fig.tight_layout()
    fig.savefig('multibounce_validation.png', dpi=150)
    print("\nSaved multibounce_validation.png")

    result = dict(
        footprint_m=args.footprint, density_per_km2=args.density, n_buildings=facets_b['n_buildings'],
        n_pulses=args.pulses, ground_material=args.ground_material, occlusion_check=occl,
        path_counts=dict(sbr_order2_raytraced=n_sbr_order2, asc_leg2_closed_form=n_asc_leg2,
                          overcount_ratio=overcount_ratio,
                          asc_aabb_blocked=stats_asc.get('n_ground_bounce_occl_blocked')),
        image_scores=scores,
        timing=dict(sbr_ms_per_pulse=stats_sbr['t_per_pulse_ms'], asc_ms_per_pulse=stats_asc['t_per_pulse_ms']),
    )
    with open('multibounce_validation_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved multibounce_validation_result.json")


if __name__ == '__main__':
    main()
