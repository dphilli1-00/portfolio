"""Regime-adaptive test: sweep escalate_threshold_cells and check
fidelity (SSIM/coherence vs ray-traced ground truth), escalated
fraction, and cost -- same scene/rays as the earlier taper tests."""
import time
import numpy as np

from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend, make_aim_grid, C
import multibounce_demo as mb
from validation_tier2_compare import backproject, required_freq_samples, compute_ref_ranges, to_numpy
from validation_multibounce_compare import score_whole_image

xp, on_gpu = get_backend(False)

footprint, density, rays, pulses, freq_arg, img_size = 300.0, 200.0, 120, 20, 32, 80
standoff, altitude, fc, bandwidth = 8000.0, 3000.0, 10e9, 600e6

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')

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
                                           max_bounces=2, return_components=True)
print(f"ray-traced: {time.perf_counter()-t0:.1f}s, counts={stats_sbr['counts']}")
s_sbr_order2 = stats_sbr['s_by_order']['order2']
img_sbr2 = backproject(xp, s_sbr_order2, plat, freqs, grid, grid, fc, R_ref)

configs = [
    ('baseline: all single-point', dict(regime_adaptive=False)),
    ('all decomposed: n_sub=32', dict(regime_adaptive=False)),  # handled specially below
    ('regime-adaptive: threshold=1 cell', dict(regime_adaptive=True, escalate_threshold_cells=1.0, escalate_n_sub=16)),
    ('regime-adaptive: threshold=3 cells', dict(regime_adaptive=True, escalate_threshold_cells=3.0, escalate_n_sub=16)),
    ('regime-adaptive: threshold=10 cells', dict(regime_adaptive=True, escalate_threshold_cells=10.0, escalate_n_sub=16)),
    ('regime-adaptive: threshold=30 cells', dict(regime_adaptive=True, escalate_threshold_cells=30.0, escalate_n_sub=16)),
]

for label, kwargs in configs:
    t0 = time.perf_counter()
    if label == 'all decomposed: n_sub=32':
        s_asc, stats_asc = mb.run_asc_cached_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                                          ground_material='dry_soil', return_components=True,
                                                          occlusion_check=True, occlusion_method='aabb',
                                                          ground_bounce_n_sub=32)
    else:
        s_asc, stats_asc = mb.run_asc_cached_multibounce(xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
                                                          ground_material='dry_soil', return_components=True,
                                                          occlusion_check=True, occlusion_method='aabb', **kwargs)
    dt = time.perf_counter() - t0
    s_asc_leg2 = stats_asc['s_by_leg']['leg2']
    img_asc2 = backproject(xp, s_asc_leg2, plat, freqs, grid, grid, fc, R_ref)
    scores, _, _ = score_whole_image(img_sbr2, img_asc2)
    esc_frac = stats_asc.get('n_escalated_fraction')
    esc_str = f", escalated={esc_frac*100:.1f}%" if esc_frac is not None else ""
    print(f"\n{label} [{dt:.2f}s{esc_str}]:")
    print(f"  SSIM={scores['ssim']:.4f}  amp_rms={scores['amp_rms']:.4f}  "
          f"phase_rms={scores['phase_rms_deg']:.2f}deg  coherence={scores['coherence']:.4f}")
