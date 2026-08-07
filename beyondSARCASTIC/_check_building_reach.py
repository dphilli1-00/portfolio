"""Internal-only CPU check (does NOT modify production code): for a
realistic 1000m/200-building scene, how many OTHER buildings does a given
building actually need to test for leg2's box search, if we cull by a
generous max double-bounce distance -- vs the current unculled O(buildings)
per facet?

If most buildings are within range of most other buildings anyway, culling
won't help much (the O(F x B) box search is close to irreducible here). If
most buildings only have a small fraction of the total within realistic
double-bounce range, a static per-building candidate list computed ONCE
(building positions don't change pulse to pulse, only platform position
does) could cut the box search by a large, real factor with zero per-pulse
cost for the culling itself.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim')
sys.path.insert(0, '/sessions/inspiring-amazing-goodall/mnt/sar_sim/sar_sim')

from dense_sbr_demo import get_backend, C
from scene_multi_building_adaptive import make_multi_building_scene_adaptive
from box_projected_multibounce import _building_boxes_from_facets

xp, on_gpu = get_backend(False)

footprint = 1000.0
density = 200.0
fc = 10e9
wavelength = C / fc
standoff, altitude = 8000.0, 3000.0
o_broadside = np.array([0.0, -standoff, altitude])

facets_b = make_multi_building_scene_adaptive(
    xp, footprint, density_per_km2=density, seed=0, o=o_broadside,
    dR_max=2.0, delta_cr=2.0, wavelength=wavelength,
    min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0))
n_buildings = facets_b['n_buildings']
print(f"{n_buildings} buildings, {facets_b['n_facets']} facets, footprint={footprint}m")

box_min, box_max = _building_boxes_from_facets(xp, facets_b)
centers = 0.5 * (box_min + box_max)
centers_xy = centers[:, :2]

# pairwise building-center xy distances (n_buildings x n_buildings) -- tiny,
# 200x200 = 40,000 entries, negligible cost regardless of backend
dx = centers_xy[:, None, 0] - centers_xy[None, :, 0]
dy = centers_xy[:, None, 1] - centers_xy[None, :, 1]
dist = np.sqrt(dx * dx + dy * dy)

for max_range in [50.0, 100.0, 150.0, 200.0, 300.0]:
    reachable = dist < max_range
    counts = reachable.sum(axis=1) - 1  # exclude self
    counts = np.maximum(counts, 0)
    print(f"max_bounce_range={max_range:.0f}m: median reachable buildings = {int(np.median(counts))}, "
          f"mean = {counts.mean():.1f}, max = {int(counts.max())}, "
          f"(out of {n_buildings-1} possible) -- "
          f"effective box-search reduction ~{(n_buildings-1)/max(1,counts.mean()):.1f}x")
