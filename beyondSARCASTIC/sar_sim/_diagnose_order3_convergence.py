"""Is the ray-traced order3 count (7, on this scene) actually converged,
or is it a severe undercount from too sparse an aim-grid discovering a
narrow, rare population? The closed-form chain (after fixing the
retroreflection-pointing bug) still finds ~28 candidates after occlusion
that pointing-check to within 1.4 degrees -- tightening the tolerance
further made zero difference, suggesting those 28 are already
near-exact geometric trihedrals, not loose false positives. Test whether
increasing SBR's own ray density moves the order3 count/energy, the same
convergence check already run for order2 earlier this session."""
import time
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend, make_aim_grid
import multibounce_demo as mb
from validation_tier2_compare import backproject, required_freq_samples, compute_ref_ranges, to_numpy

xp, on_gpu = get_backend(False)
footprint, density, pulses, freq_arg = 300.0, 200.0, 14, 24
standoff, altitude, fc, bandwidth = 8000.0, 3000.0, 10e9, 600e6

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')

squint_len = 400.0
u = np.linspace(-squint_len / 2, squint_len / 2, pulses)
plat = xp.asarray(np.column_stack([u, np.full(pulses, -standoff), np.full(pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, 60))
min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
freq = max(freq_arg, min_freq)
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, freq))

for rays in [80, 130, 190]:
    aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)
    t0 = time.perf_counter()
    s_sbr, stats_sbr = mb.run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                               max_bounces=3, return_components=True)
    dt = time.perf_counter() - t0
    o3 = stats_sbr['s_by_order']['order3']
    energy = float(to_numpy(xp.sum(xp.abs(o3)**2)))
    print(f"rays={rays} ({rays*rays} aim pts): {dt:.1f}s  order3 unique paths={stats_sbr['counts']['order3']}  "
          f"order3 total energy={energy:.4e}")
