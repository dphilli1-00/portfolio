"""Internal-only CPU validation (real production code, both sides) of the
new leg2_retro_check validity gate just added to run_multibounce_sbr. Runs
the actual production functions (not a reimplementation) with the gate on
vs off, apples-to-apples against ASC's own (already-gated) leg2_building, to
confirm the fix moves in the right direction before paying for a real GPU
run."""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import run_multibounce_sbr
from box_projected_multibounce import run_asc_box_projected_multibounce
from validation_tier2_compare import to_numpy, required_freq_samples

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 150.0
density = 300.0
fc = 10e9
bandwidth = 300e6
az_res = 3.0
standoff, altitude = 8000.0, 3000.0
wavelength = C / fc
o_broadside = np.array([0.0, -standoff, altitude])

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=o_broadside,
    dR_max=5.0, delta_cr=5.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 8
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = xp.asarray(np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, 40))
min_freq, maxdr, needed_window = required_freq_samples(to_numpy(plat), to_numpy(ref_pos), to_numpy(grid), bandwidth)
n_freq = int(np.ceil(min_freq * 1.05))
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
print(f"n_pulses={n_pulses}, n_freq={n_freq}")

aim_pts = facets_b['center']


def raw_coherence(a, b):
    a, b = to_numpy(a), to_numpy(b)
    return float(np.abs(np.vdot(a.ravel(), b.ravel())) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


for retro_on in (False, True):
    print(f"\n=== leg2_retro_check={retro_on} ===")
    s_sbr, sbr_stats = run_multibounce_sbr(xp, on_gpu, facets_b, facets_g, plat, aim_pts, freqs, ref_pos,
                                            max_bounces=2, aim_is_direction=False,
                                            return_components=True, progress=False,
                                            split_order2_by_target=True,
                                            leg2_retro_check=retro_on, retro_beamwidth_mult=3.0)
    print(f"SBR counts: {sbr_stats['counts']}")
    order2_ground_sbr = sbr_stats['s_by_order']['order2_ground']
    order2_building_sbr = sbr_stats['s_by_order']['order2_building']

    s_asc, asc_stats = run_asc_box_projected_multibounce(
        xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
        ground_material='dry_soil', return_components=True, include_order3=False,
        progress=False, leg2_occlusion_check=False, split_leg2_by_target=True,
        leg2_retroreflection_check=True, retro_beamwidth_mult=3.0,
        leg2_building_enabled=True, leg2_retro_taper=False)
    print(f"ASC counts: {asc_stats['counts']}")
    leg2_ground_asc = asc_stats['s_by_leg']['leg2_ground']
    leg2_building_asc = asc_stats['s_by_leg']['leg2_building']

    coh_ground = raw_coherence(order2_ground_sbr, leg2_ground_asc)
    coh_building = raw_coherence(order2_building_sbr, leg2_building_asc)
    print(f"apples-to-apples raw coherence: leg2_ground={coh_ground:.4f}  leg2_building={coh_building:.4f}")
