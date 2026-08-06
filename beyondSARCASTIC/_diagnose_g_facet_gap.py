"""Internal-only diagnostic (CPU, small scene) -- does NOT modify
box_projected_multibounce.py. Tests one specific hypothesis: for the
building-target subset of leg2 that SURVIVES the retro-reflection gate
(the physically-plausible corner candidates), how far is G (the continuous
ray/box intersection point on the target building's surface) from the
NEAREST actual facet center on that same target building?

Motivation: leg2's building-target coherence is still only ~0.63 after the
retro-gate fix (vs leg1's ~0.96). One candidate explanation was a position
mismatch -- SBR scores a building-target bounce at the real, discrete
facet center the ray actually hit; box_projected_multibounce computes G as
a continuous point on the target box's surface that doesn't necessarily
land on any specific facet's center, and this codebase already validated
(for ground) that a few meters of position mismatch is enough to fully
decorrelate X-band phase. Pushback (correct to check before assuming):
task #32's Fresnel/resolution-cell facet partitioning was specifically
built so "one point within a facet is a fine stand-in for that facet's own
center" -- if that guarantee also holds for target buildings (not just
source walls), G landing anywhere within a small facet's footprint should
already be close to that facet's own center, and this hypothesis is wrong.
This script measures the actual G-to-nearest-facet-center distance instead
of assuming either way.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, make_ground_facet
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import (_building_boxes_from_facets, _reflect_and_intersect_scene)

xp, on_gpu = get_backend(False)   # CPU only, this is a quick internal check

footprint = 300.0
density = 200.0
fc = 10e9
wavelength = C / fc
standoff, altitude = 8000.0, 3000.0
o = np.array([50.0, -standoff, altitude])   # one representative off-broadside pulse position

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=np.array([0.0, -standoff, altitude]),
    dR_max=2.0, delta_cr=2.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
facets_g = make_ground_facet(xp, footprint)
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets, wavelength={wavelength*100:.2f}cm")

Cb, Nb = facets_b['center'], facets_b['normal']
fbid = facets_b['facet_building_id']
box_min, box_max = _building_boxes_from_facets(xp, facets_b)
half_extent_g = float(facets_g.get('footprint_half_extent', footprint / 2.0))

look = Cb - o[None, :]
R_asc = np.linalg.norm(look, axis=1)
d_in = look / R_asc[:, None]
cos_inc1 = np.sum(-d_in * Nb, axis=1)
visible1 = cos_inc1 > 0

hit = _reflect_and_intersect_scene(xp, Cb, Nb, d_in, half_extent_g, box_min, box_max, fbid)
G, valid_geom2, d_out = hit['G'], hit['valid'], hit['d_out']
surf_normal, is_ground, building_hit = hit['surf_normal'], hit['is_ground'], hit['building_hit']

valid2 = visible1 & valid_geom2 & (~is_ground)   # building-target subset only
print(f"building-target candidates (pre retro-gate): {int(valid2.sum())}")

# Same retro-reflection misalignment test as leg2_retroreflection_check --
# replicated here standalone rather than importing/modifying the real
# function, to keep this script fully independent of the code being tested.
return_vec = o[None, :] - G
return_len = np.linalg.norm(return_vec, axis=1)
retro_dir = d_out - 2.0 * np.sum(d_out * surf_normal, axis=1, keepdims=True) * surf_normal
return_len_safe = np.where(return_len > 1e-9, return_len, 1.0)
to_sensor_dir = return_vec / return_len_safe[:, None]
retro_cos = np.clip(np.sum(retro_dir * to_sensor_dir, axis=1), -1.0, 1.0)
retro_angle = np.arccos(retro_cos)
L_wall = 2.0 * facets_b['half_u']
beamwidth = wavelength / np.maximum(L_wall, wavelength)
retro_valid = retro_angle < (3.0 * beamwidth)

survivors = valid2 & retro_valid
print(f"building-target survivors (post retro-gate, what leg2_building actually scores): {int(survivors.sum())}")

idx_survivors = np.nonzero(survivors)[0]
if idx_survivors.shape[0] == 0:
    print("no survivors at this single pulse -- try a different o / seed")
    sys.exit(0)

dists = []
for i in idx_survivors:
    tgt_building = int(building_hit[i])
    mask = fbid == tgt_building
    tgt_centers = Cb[mask]
    d = np.linalg.norm(tgt_centers - G[i][None, :], axis=1)
    dists.append(d.min())
dists = np.array(dists)

print(f"\nG-to-nearest-target-facet-center distance, over {len(dists)} surviving building-target hits:")
print(f"  min:    {dists.min():.4f} m  ({dists.min()/wavelength:.2f} wavelengths)")
print(f"  median: {np.median(dists):.4f} m  ({np.median(dists)/wavelength:.2f} wavelengths)")
print(f"  mean:   {dists.mean():.4f} m  ({dists.mean()/wavelength:.2f} wavelengths)")
print(f"  max:    {dists.max():.4f} m  ({dists.max()/wavelength:.2f} wavelengths)")
print(f"  90th pct: {np.percentile(dists, 90):.4f} m  ({np.percentile(dists, 90)/wavelength:.2f} wavelengths)")
frac_over_quarter_wave = (dists > wavelength / 4.0).mean()
print(f"  fraction exceeding lambda/4 ({wavelength/4*100:.2f}cm, enough for a ~90-degree phase error): {frac_over_quarter_wave*100:.1f}%")
