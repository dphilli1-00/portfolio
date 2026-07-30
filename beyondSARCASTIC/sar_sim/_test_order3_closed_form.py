"""Test the new closed-form wall-ground-wall order-3 term
(run_asc_cached_order3_wall_ground_wall) against the ray-traced order-3
ground truth (max_bounces=3), same scene/validation pattern as every
other A/B test this session. First check: does the closed-form side even
find any order-3 candidates in this scene (buildings are sparse boxes --
plausible that facing walls are rare), then score against the ray-traced
order3 image if it does.
"""
import time
import numpy as np

from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend, make_aim_grid
import multibounce_demo as mb
from validation_tier2_compare import backproject, required_freq_samples, compute_ref_ranges, to_numpy
from validation_multibounce_compare import score_whole_image

xp, on_gpu = get_backend(False)

footprint, density, rays, pulses, freq_arg, img_size = 300.0, 200.0, 100, 14, 24, 64
standoff, altitude, fc, bandwidth = 8000.0, 3000.0, 10e9, 600e6

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

squint_len = 400.0
u = np.linspace(-squint_len / 2, squint_len / 2, pulses)
plat = xp.asarray(np.column_stack([u, np.full(pulses, -standoff), np.full(pulses, altitude)]))
aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, img_size))

min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
freq = max(freq_arg, min_freq)
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, freq))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

t0 = time.perf_counter()
s_sbr, stats_sbr = mb.run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                           max_bounces=3, return_components=True)
print(f"ray-traced (max_bounces=3): {time.perf_counter()-t0:.1f}s, counts={stats_sbr['counts']}")

t0 = time.perf_counter()
s_asc3, stats_asc3 = mb.run_asc_cached_order3_wall_ground_wall(
    xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
    ground_material='dry_soil', occlusion_check=True, return_components=True)
dt = time.perf_counter() - t0
print(f"closed-form order3 (wall-ground-wall): {dt:.2f}s  n_order3_paths={stats_asc3['n_order3_paths']} "
      f"(occl_blocked={stats_asc3['n_order3_occl_blocked']})")

if stats_sbr['counts']['order3'] == 0 and stats_asc3['n_order3_paths'] == 0:
    print("\nNeither side found any order-3 paths in this scene -- can't score, need a scene with facing walls.")
else:
    img_sbr3 = backproject(xp, stats_sbr['s_by_order']['order3'], plat, freqs, grid, grid, fc, R_ref)
    img_asc3 = backproject(xp, stats_asc3['s_by_leg']['order3'], plat, freqs, grid, grid, fc, R_ref)
    scores, _, _ = score_whole_image(img_sbr3, img_asc3)
    print(f"\nray-traced order3 paths: {stats_sbr['counts']['order3']}   closed-form order3 paths (post-occl): "
          f"{stats_asc3['n_order3_paths']}")
    print(f"SSIM={scores['ssim']:.4f}  amp_rms={scores['amp_rms']:.4f}  "
          f"phase_rms={scores['phase_rms_deg']:.2f}deg  coherence={scores['coherence']:.4f}")

    # also: what does occlusion_check=False give, to see how much the
    # occlusion filtering itself matters here (mirrors the double-bounce
    # story: unchecked closed-form massively overcounts)
    s_asc3_noocc, stats_asc3_noocc = mb.run_asc_cached_order3_wall_ground_wall(
        xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
        ground_material='dry_soil', occlusion_check=False, return_components=True)
    img_asc3_noocc = backproject(xp, stats_asc3_noocc['s_by_leg']['order3'], plat, freqs, grid, grid, fc, R_ref)
    scores_noocc, _, _ = score_whole_image(img_sbr3, img_asc3_noocc)
    print(f"\nWITHOUT occlusion check: n_order3_paths={stats_asc3_noocc['n_order3_paths']}  "
          f"SSIM={scores_noocc['ssim']:.4f}  coherence={scores_noocc['coherence']:.4f}")
