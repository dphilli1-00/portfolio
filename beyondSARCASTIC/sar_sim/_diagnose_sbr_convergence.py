"""Is dense-SBR itself converged, or is it a moving target?

make_aim_grid builds a uniform grid on the GROUND PLANE (x,y), then rays go
from platform to each ground-plane point -- wall hits happen wherever those
rays cross a vertical facet en route. That means wall-height sampling density
is an artifact of ground-plane grid spacing projected through an oblique
viewing angle, NOT a uniform or physically-motivated sampling of the wall.
If that artifact matters, the SBR "ground truth" image itself should still
be changing as ray density increases -- i.e. it isn't ground truth, it's one
particular discretization, same category of thing as the ASC model it's
being compared against.

Test: same scene/geometry, sweep ray density, compare each order-2 SBR image
to the next-denser one (self-consistency / convergence), not to ASC at all.
"""
import time
import numpy as np

from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend, make_aim_grid
import multibounce_demo as mb
from validation_tier2_compare import backproject, required_freq_samples, compute_ref_ranges, to_numpy
from validation_multibounce_compare import score_whole_image

xp, on_gpu = get_backend(False)

footprint, density, pulses, freq_arg, img_size = 300.0, 200.0, 12, 24, 60
standoff, altitude, fc, bandwidth = 8000.0, 3000.0, 10e9, 600e6

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')

squint_len = 400.0
u = np.linspace(-squint_len / 2, squint_len / 2, pulses)
plat = xp.asarray(np.column_stack([u, np.full(pulses, -standoff), np.full(pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, img_size))

min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
freq = max(freq_arg, min_freq)
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, freq))
R_ref = compute_ref_ranges(xp, plat, ref_pos)

ray_counts = [60, 90, 120, 180]
imgs = {}
for rays in ray_counts:
    aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)
    t0 = time.perf_counter()
    s_sbr, stats_sbr = mb.run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                               max_bounces=2, return_components=True)
    dt = time.perf_counter() - t0
    s2 = stats_sbr['s_by_order']['order2']
    img2 = backproject(xp, s2, plat, freqs, grid, grid, fc, R_ref)
    imgs[rays] = img2
    n_ground_hits = stats_sbr['counts'].get('order2', '?')
    print(f"rays={rays} ({rays*rays} aim pts): {dt:.2f}s  order2_paths={n_ground_hits}")

print("\nSelf-consistency: each ray density vs. the next denser one (order-2 image only)")
for a, b in zip(ray_counts[:-1], ray_counts[1:]):
    scores, _, _ = score_whole_image(imgs[a], imgs[b])
    print(f"  rays={a:>3} vs rays={b:>3}:  SSIM={scores['ssim']:.4f}  coherence={scores['coherence']:.4f}  "
          f"phase_rms={scores['phase_rms_deg']:.2f}deg")

print("\nSelf-consistency: lowest density vs. highest density (order-2 image only)")
scores, _, _ = score_whole_image(imgs[ray_counts[0]], imgs[ray_counts[-1]])
print(f"  rays={ray_counts[0]} vs rays={ray_counts[-1]}:  SSIM={scores['ssim']:.4f}  coherence={scores['coherence']:.4f}  "
      f"phase_rms={scores['phase_rms_deg']:.2f}deg")
