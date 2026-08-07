"""
Test whether ASC's leg1 (box_projected_multibounce.py / multibounce_demo.py's
run_asc_box_projected_multibounce leg1 term) is over-counting facets that are
actually occluded by ANOTHER BUILDING -- the gap decompose_sbr_asc_coherence.py's
own module docstring flagged from the start but this project never directly
measured: ASC's leg1 visibility is a pure backface cull (cos_inc1 > 0, see
that function's amp_eff1_geom line), with no check for whether some OTHER
facet sits between the platform and this one. SBR's dense ray tracer gets
occlusion for free -- ray_facet_intersect finds the nearest hit along each
ray, so a blocked facet simply never registers.

Motivation for running this NOW specifically: diagnose_tier2_leg_split.py on
the real 1000m/200-building cache showed leg1 (single-bounce, the SIMPLEST
term -- no retro-check, no culling, no roof/wall dilution concerns) is what's
dragging the combined per-building SSIM pass rate down to 44%, while leg2
(double-bounce, all this session's complexity) actually does BETTER at 66%.
An earlier version of this test on a single-building toy scene (where
building-to-building occlusion is geometrically impossible) found ZERO
self-occlusion -- this scene has 200 buildings, so it can actually exercise
the hypothesis the toy scene couldn't.

For each of a handful of sampled pulses: cast one ray per facet, aimed
directly at that facet's own center (the same direct-to-facet convention
order1_sbr itself uses) -- SAME convention ASC uses to decide visible1 via
cos_inc1 > 0. Compare ASC's naive visibility mask against SBR's own
ray_facet_intersect (with the same building-level AABB culling
run_multibounce_sbr uses internally, via multibounce_demo._building_aabbs)
to see whether the ray aimed at facet i's center actually LANDS on facet i,
or on something else (nearest_idx != i) -- occluded.

This is compute-heavy at full scene scale (146,053 facets) -- ~30s/pulse
even WITH building-level AABB culling, measured on CPU. Runs fine on GPU;
--pulses controls how many (evenly spaced across the aperture) to sample --
you don't need all 200 to get a representative read, but pass a larger
number if you want more confidence.

Usage:
    python3 diagnose_leg1_occlusion.py --gpu --pulses 10
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sar_sim'))

from dense_sbr_demo import get_backend, C, ray_facet_intersect
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from multibounce_demo import _building_aabbs
from validation_tier2_compare import to_numpy

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--footprint', type=float, default=1000.0)
ap.add_argument('--density', type=float, default=200.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dr-max', type=float, default=2.0)
ap.add_argument('--delta-cr', type=float, default=2.0)
ap.add_argument('--standoff', type=float, default=8000.0)
ap.add_argument('--altitude', type=float, default=3000.0)
ap.add_argument('--fc', type=float, default=10e9)
ap.add_argument('--az-res', type=float, default=2.0)
ap.add_argument('--aperture-pulses', type=int, default=200,
                 help='total aperture size (must match the cache this is meant to explain -- '
                      'only affects the platform sweep geometry, not how many are SAMPLED below)')
ap.add_argument('--pulses', type=int, default=10,
                 help='how many pulses, evenly spaced across the aperture, to actually test -- '
                      'each one costs a full (F,F)-scale culled ray-facet intersection pass')
args = ap.parse_args()

xp, on_gpu = get_backend(args.gpu)
print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

standoff, altitude = args.standoff, args.altitude
o_broadside = np.array([0.0, -standoff, altitude])
wavelength = C / args.fc

t0 = time.perf_counter()
facets_b = make_multi_building_scene_adaptive(
    xp, args.footprint, density_per_km2=args.density, seed=args.seed, o=o_broadside,
    dR_max=args.dr_max, delta_cr=args.delta_cr, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
print(f"{facets_b['n_buildings']} buildings, {facets_b['n_facets']} facets ({time.perf_counter()-t0:.1f}s)")

Cb, Nb = facets_b['center'], facets_b['normal']
Cb_np, Nb_np = to_numpy(Cb), to_numpy(Nb)
F = Cb_np.shape[0]

box_min, box_max, building_id_of_facet = _building_aabbs(xp, facets_b)
fbid_np = to_numpy(building_id_of_facet)

d_theta = wavelength / (2.0 * args.az_res)
squint_len = d_theta * standoff
n_pulses = max(args.aperture_pulses, int(np.ceil(standoff * wavelength / (args.az_res ** 2))))
u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
plat = np.column_stack([u, np.full(n_pulses, -standoff), np.full(n_pulses, altitude)])

sample_idx = np.linspace(0, n_pulses - 1, args.pulses).astype(int)
sample_idx = np.unique(sample_idx)
print(f"sampling {len(sample_idx)} of {n_pulses} pulses: {sample_idx.tolist()}")

n_asc_visible_total = 0
n_occluded_total = 0
occluded_by_building = np.zeros(facets_b['n_buildings'], dtype=int)
visible_by_building = np.zeros(facets_b['n_buildings'], dtype=int)

for pi in sample_idx:
    t0 = time.perf_counter()
    o = plat[pi]
    look = Cb_np - o[None, :]
    R = np.linalg.norm(look, axis=1)
    d_in = look / R[:, None]
    cos_inc1 = np.sum(-d_in * Nb_np, axis=1)
    asc_visible = cos_inc1 > 0

    hit = ray_facet_intersect(xp, xp.asarray(o), xp.asarray(d_in), facets_b,
                               box_min=box_min, box_max=box_max,
                               building_id_of_facet=building_id_of_facet)
    nearest_idx = to_numpy(hit[4])
    hit_mask = to_numpy(hit[3])
    self_idx = np.arange(F)
    truly_visible = hit_mask & (nearest_idx == self_idx)

    occluded = asc_visible & ~truly_visible
    n_asc_visible_total += int(asc_visible.sum())
    n_occluded_total += int(occluded.sum())
    for b in np.unique(fbid_np[asc_visible]):
        mask_b = asc_visible & (fbid_np == b)
        visible_by_building[b] += int(mask_b.sum())
        occluded_by_building[b] += int((mask_b & ~truly_visible).sum())
    print(f"  pulse {pi}: ASC-visible={int(asc_visible.sum())}, occluded={int(occluded.sum())} "
          f"({100*occluded.sum()/max(1,asc_visible.sum()):.1f}%)  [{time.perf_counter()-t0:.1f}s]")

print(f"\n=== across {len(sample_idx)} sampled pulses ===")
print(f"total ASC-visible facet-pulses: {n_asc_visible_total}")
print(f"actually occluded per SBR's own ray trace: {n_occluded_total} "
      f"({100*n_occluded_total/max(1,n_asc_visible_total):.1f}%)")

frac_occluded_by_building = np.divide(occluded_by_building, np.maximum(visible_by_building, 1))
worst = np.argsort(-frac_occluded_by_building)[:10]
print("\nworst 10 buildings by occlusion fraction (of their own ASC-visible facets):")
for b in worst:
    if visible_by_building[b] > 0:
        print(f"  building {b}: {occluded_by_building[b]}/{visible_by_building[b]} occluded "
              f"({100*frac_occluded_by_building[b]:.1f}%)")
