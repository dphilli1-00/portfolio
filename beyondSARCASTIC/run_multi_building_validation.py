"""Multi-building validation: adaptive-mesh dense-SBR vs. closed-form
point-ASC, on a randomly generated multi-building scene (not just the
single-building toy case).

Known tuning notes from getting this to run in reasonable time:
- Pulse count must satisfy azimuth-ambiguity Nyquist against the FULL
  illuminated scene footprint (SCENE_FOOTPRINT_M), not just the display
  window -- any scatterer anywhere in the scene can alias into the image
  if pulses are too sparse. Rule of thumb used here:
      pulses_min > squint_len * 2 * (SCENE_FOOTPRINT_M/2) / (R * wavelength)
- Facet count (and thus runtime) is dominated by TALL buildings viewed
  near broadside: order-1 needs fine height subdivision there (dR_max
  controls this). max_wh height and dR_max are the two knobs that trade
  fidelity for runtime -- shrink either if this is too slow.
- Ray density (aim grid resolution) matters a lot for order-2 fidelity
  specifically: SBR only credits a (wall-facet, ground-patch) combo if
  some cast ray actually lands on it, while point-ASC enumerates every
  geometrically valid combo analytically regardless of ray density. Too
  sparse a ray grid makes SBR under-sample the combination space and
  diverge from ASC even though both are "correct" -- this was measured
  directly: going from 80x80 to 120x120 rays on the single-building case
  took phase bias from 27.5 deg to 0.0 deg. If SBR-vs-ASC agreement looks
  worse than expected, increase the aim grid before suspecting a model bug.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import make_ground_facet, get_backend, make_aim_grid
from multibounce_demo import run_multibounce_sbr, run_asc_cached_multibounce
from validation_tier2_compare import (backproject, compute_ref_ranges, to_numpy,
                                       required_freq_samples)
from validation_multibounce_compare import score_whole_image
from scene_multi_building_adaptive import make_multi_building_scene_adaptive

t_start = time.perf_counter()
xp, on_gpu = get_backend(False)

SCENE_FOOTPRINT_M = 100.0
IMG_WINDOW_M = 100.0
GROUND_MATERIAL = 'dry_soil'
standoff, altitude = 8000.0, 3000.0
fc, bandwidth = 9.6e9, 6.0e8
pulses, img_size = 150, 120
squint_len = 400.0

o_broadside = np.array([0.0, -standoff, altitude])
facets_b = make_multi_building_scene_adaptive(
    xp, SCENE_FOOTPRINT_M, density_per_km2=350.0, seed=1, o=o_broadside, dR_max=3.0,
    min_wh=(8.0, 8.0, 6.0), max_wh=(18.0, 18.0, 16.0))
facets_g = make_ground_facet(xp, SCENE_FOOTPRINT_M, material=GROUND_MATERIAL)
print(f"n_facets building={facets_b['n_facets']}  ground={facets_g['center'].shape[0]}", flush=True)

u = np.linspace(-squint_len / 2, squint_len / 2, pulses)
plat = xp.asarray(np.column_stack([u, np.full(pulses, -standoff), np.full(pulses, altitude)]))
aim_pts, margin = make_aim_grid(xp, SCENE_FOOTPRINT_M, 300, standoff, altitude, max_height_m=45.0)
print(f"aim grid: 90x90={aim_pts.shape[0]} rays over {SCENE_FOOTPRINT_M}m scene", flush=True)

ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-IMG_WINDOW_M / 2, IMG_WINDOW_M / 2, img_size))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
n_freq = int(min_freq * 1.05)
print(f"required_freq_samples: min_freq={min_freq} -> n_freq={n_freq}", flush=True)
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
print(f"setup: {time.perf_counter() - t_start:.1f}s", flush=True)

t0 = time.perf_counter()
s_sbr, stats_sbr = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                        max_bounces=2, return_components=True)
s_sbr_order2 = stats_sbr['s_by_order']['order2']
s_sbr_order1 = stats_sbr['s_by_order']['order1']
print(f"[1/2] SBR (adaptive mesh): {time.perf_counter() - t0:.1f}s  counts={stats_sbr['counts']}")

t0 = time.perf_counter()
s_asc, stats_asc = run_asc_cached_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                               ground_material=GROUND_MATERIAL, return_components=True,
                                               occlusion_check=False)
s_asc_leg2 = stats_asc['s_by_leg']['leg2']
s_asc_leg1 = stats_asc['s_by_leg']['leg1']
print(f"[2/2] point-ASC: {time.perf_counter() - t0:.1f}s")

img_sbr2 = backproject(xp, s_sbr_order2, plat, freqs, grid, grid, fc, R_ref)
img_asc2 = backproject(xp, s_asc_leg2, plat, freqs, grid, grid, fc, R_ref)
img_sbr_full = backproject(xp, s_sbr_order1 + s_sbr_order2, plat, freqs, grid, grid, fc, R_ref)
img_asc_full = backproject(xp, s_asc_leg1 + s_asc_leg2, plat, freqs, grid, grid, fc, R_ref)

scores2, _, _ = score_whole_image(img_sbr2, img_asc2)
scores_full, _, _ = score_whole_image(img_sbr_full, img_asc_full)

print("\n=== point-ASC vs adaptive-mesh SBR (order2 only) ===")
for k in ['ssim', 'amp_rms', 'phase_rms_deg', 'phase_bias_deg', 'coherence']:
    print(f"  {k:<15} {scores2[k]:.4f}")
print("\n=== point-ASC vs adaptive-mesh SBR (order1+order2 combined) ===")
for k in ['ssim', 'amp_rms', 'phase_rms_deg', 'phase_bias_deg', 'coherence']:
    print(f"  {k:<15} {scores_full[k]:.4f}")

np.savez('multi_building_images.npz',
         img_sbr2=to_numpy(img_sbr2), img_asc2=to_numpy(img_asc2),
         img_sbr_full=to_numpy(img_sbr_full), img_asc_full=to_numpy(img_asc_full),
         grid=to_numpy(grid))
print(f"\ntotal: {time.perf_counter() - t_start:.1f}s")
