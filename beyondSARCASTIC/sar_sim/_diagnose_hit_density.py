"""How is ray tracing's actual bounce-1 hit density distributed along a
wall's height (z-fraction from base=0 to roof=1)? My _ground_bounce_
subpoint_contrib assumed UNIFORM density -- check whether that's true,
since the convergent-but-wrong coherence result suggests it might not be."""
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, ray_facet_intersect, get_backend, make_aim_grid
import multibounce_demo as mb

xp, on_gpu = get_backend(False)
footprint, density, rays = 300.0, 200.0, 200   # denser rays for a smoother histogram
standoff, altitude = 8000.0, 3000.0

facets_b = make_building_scene(xp, footprint, density, seed=0)
Cb, Nb, HVb = facets_b['center'], facets_b['normal'], facets_b['half_v']

aim_pts, margin = make_aim_grid(xp, footprint, rays, standoff, altitude, max_height_m=40.0)

all_zfrac = []
for u in [-200, -100, 0, 100, 200]:
    o = xp.asarray([float(u), -standoff, altitude])
    d1 = aim_pts - o[None, :]
    d1 = d1 / xp.linalg.norm(d1, axis=1, keepdims=True)
    hit_pt1, hit_amp1, cos_inc1, hit_mask1, idx1 = ray_facet_intersect(xp, o, d1, facets_b)
    hit = mb.to_numpy(hit_mask1)
    hp = mb.to_numpy(hit_pt1)[hit]
    idx_np = mb.to_numpy(idx1)[hit]
    z_center = mb.to_numpy(Cb[:, 2])[idx_np]
    half_v = mb.to_numpy(HVb)[idx_np]
    z_bottom = z_center - half_v
    z_top = z_center + half_v
    zfrac = (hp[:, 2] - z_bottom) / (z_top - z_bottom)   # 0=base, 1=roof
    all_zfrac.append(zfrac)

zfrac_all = np.concatenate(all_zfrac)
print(f"{zfrac_all.shape[0]} wall hits total across 5 platform positions")
hist, edges = np.histogram(zfrac_all, bins=10, range=(0, 1))
print("\nz-fraction (0=base, 1=roof) histogram, 10 bins:")
for i in range(10):
    bar = '#' * int(80 * hist[i] / max(hist.max(), 1))
    print(f"  [{edges[i]:.1f}-{edges[i+1]:.1f}] {hist[i]:>5}  {bar}")
print(f"\nmean z-fraction: {zfrac_all.mean():.3f} (0.5 = uniform/centered)")
