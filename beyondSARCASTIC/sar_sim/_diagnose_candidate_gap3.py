"""Check: is the aggregate path-count gap actually a counting-convention
mismatch? Ray tracing dedupes valid order-2 candidates by the (idx1,
idx2) PAIR -- a single wall facet (idx1) whose rays, spread across its
own finite extent, land on several different ground PATCHES (idx2,
since make_ground_facet tiles the ground into patch_m-sized cells)
gets counted as multiple distinct "paths". ASC's closed-form check
produces exactly ONE ground point per wall facet (the center's own
analytic reflection), so it can never report more than one path per
wall. If ray tracing's path count is inflated by many-ground-patches-
per-wall while its WALL count matches or trails ASC's, that's a
counting-convention artifact, not a visibility/occlusion difference."""
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, ray_facet_intersect, get_backend, make_aim_grid, C
import multibounce_demo as mb

xp, on_gpu = get_backend(False)
footprint, density, rays = 300.0, 200.0, 150
standoff, altitude = 8000.0, 3000.0

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')
print(f"ground patches: {facets_g['n_ground_facets']}, patch_m={facets_g['patch_m']}")
facets_combined = mb.concat_facets(xp, facets_b, facets_g)
F_b = facets_b['center'].shape[0]
F_c = facets_combined['center'].shape[0]
Cb, Nb = facets_b['center'], facets_b['normal']
Cc, Nc = facets_combined['center'], facets_combined['normal']

aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)
eps = 1e-3

print(f"\n{'u (m)':>8} {'distinct (idx1,idx2) paths':>28} {'distinct idx1 walls':>20} {'paths/wall':>12}")
for u in [-200, -100, 0, 100, 200]:
    o = xp.asarray([float(u), -standoff, altitude])
    d1 = aim_pts - o[None, :]
    d1 = d1 / xp.linalg.norm(d1, axis=1, keepdims=True)
    hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(xp, o, d1, facets_b)
    hit_normal1 = Nb[idx1]
    d2 = d1 - 2.0 * xp.sum(d1 * hit_normal1, axis=1, keepdims=True) * hit_normal1
    o2 = hit_pt1 + eps * hit_normal1
    hit_pt2, hit_amp2, cos_inc2, hit_mask2_raw, idx2 = ray_facet_intersect(xp, o2, d2, facets_combined)
    hit_mask2 = hit_mask2_raw & hit_mask1
    hit_normal2 = Nc[idx2]
    clear2 = mb._return_visible(xp, hit_pt2, hit_normal2, o, facets_combined, eps)
    valid2 = hit_mask2 & clear2

    key2 = idx1[valid2].astype(xp.int64) * F_c + idx2[valid2].astype(xp.int64)
    n_paths = int(np.unique(mb.to_numpy(key2)).shape[0])
    n_walls = int(np.unique(mb.to_numpy(idx1[valid2])).shape[0])
    print(f"{u:>8} {n_paths:>28} {n_walls:>20} {n_paths/max(n_walls,1):>12.2f}")
