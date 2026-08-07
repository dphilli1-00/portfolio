"""Internal-only CPU validation: run the ACTUAL production
run_asc_box_projected_multibounce (not the standalone prototype) with
leg2_culled_search=False vs True on the SAME real scene/pulses, and confirm
the outputs are identical -- this is the real end-to-end check, since the
wiring into the per-pulse loop could have bugs even if the standalone
prototype was correct."""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import run_asc_box_projected_multibounce, to_numpy
from validation_tier2_compare import required_freq_samples

xp, on_gpu = get_backend(False)
print(f"backend: {'cupy' if on_gpu else 'numpy (CPU)'}")

footprint = 150.0
density = 250.0
fc = 10e9
bandwidth = 300e6
az_res = 3.0
standoff, altitude = 8000.0, 3000.0
wavelength = C / fc
o_broadside = np.array([0.0, -standoff, altitude])

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=o_broadside,
    dR_max=2.0, delta_cr=2.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets")

d_theta = wavelength / (2.0 * az_res)
squint_len = d_theta * standoff
n_pulses = 6
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = xp.asarray(np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)]))
ref_pos = xp.zeros(3)
grid = xp.asarray(np.linspace(-footprint / 2, footprint / 2, 40))
min_freq, maxdr, needed_window = required_freq_samples(to_numpy(xp, plat), to_numpy(xp, ref_pos), to_numpy(xp, grid), bandwidth)
n_freq = int(np.ceil(min_freq * 1.05))
freqs = xp.asarray(fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq))
print(f"n_pulses={n_pulses}, n_freq={n_freq}")

results = {}
for culled in (False, True):
    s_asc, asc_stats = run_asc_box_projected_multibounce(
        xp, on_gpu, facets_b, facets_g, plat, freqs, ref_pos,
        ground_material='dry_soil', return_components=True, include_order3=False,
        progress=True, leg2_occlusion_check=False, split_leg2_by_target=True,
        leg2_retroreflection_check=True, retro_beamwidth_mult=3.0,
        leg2_building_enabled=True, leg2_retro_taper=False,
        leg2_culled_search=culled)
    results[culled] = (s_asc, asc_stats)
    print(f"culled={culled}: counts={asc_stats['counts']}")

s_off, stats_off = results[False]
s_on, stats_on = results[True]

diff_total = float(np.max(np.abs(to_numpy(xp, s_off) - to_numpy(xp, s_on))))
diff_leg1 = float(np.max(np.abs(to_numpy(xp, stats_off['s_by_leg']['leg1']) - to_numpy(xp, stats_on['s_by_leg']['leg1']))))
diff_leg2 = float(np.max(np.abs(to_numpy(xp, stats_off['s_by_leg']['leg2']) - to_numpy(xp, stats_on['s_by_leg']['leg2']))))
diff_leg2g = float(np.max(np.abs(to_numpy(xp, stats_off['s_by_leg']['leg2_ground']) - to_numpy(xp, stats_on['s_by_leg']['leg2_ground']))))
diff_leg2b = float(np.max(np.abs(to_numpy(xp, stats_off['s_by_leg']['leg2_building']) - to_numpy(xp, stats_on['s_by_leg']['leg2_building']))))

print(f"\n=== production end-to-end diff, culled vs uncalled ===")
print(f"max |s_total| diff:   {diff_total:.3e}")
print(f"max |leg1| diff:      {diff_leg1:.3e}")
print(f"max |leg2| diff:      {diff_leg2:.3e}")
print(f"max |leg2_ground| diff:   {diff_leg2g:.3e}")
print(f"max |leg2_building| diff: {diff_leg2b:.3e}")
print(f"counts match: {stats_off['counts'] == stats_on['counts']}")

all_ok = diff_total < 1e-9 and stats_off['counts'] == stats_on['counts']
print(f"\n-> {'PASS' if all_ok else 'FAIL'}")
