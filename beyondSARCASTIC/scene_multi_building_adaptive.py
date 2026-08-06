"""Multi-building scene, same random layout generator as dense_sbr_demo's
make_building_scene, but each wall is adaptively partitioned in BOTH
height and length (merged order1+order2, joint range+cross-range+Fresnel
partition, visibility-gated) instead of being one undivided facet or a
height-only stack -- the fix validated on the single-building case,
generalized to N buildings at their own (random) positions, sizes,
heights, and viewing angles relative to the sensor.

Why this matters (see adaptive_facet_partition.py for the derivation):
_score_paths (in multibounce_demo.py) always evaluates a bounce at that
facet's CENTER -- a fine approximation for facets a few meters across,
not for a whole undivided multi-meter-tall, multi-meter-wide wall.
Height was already partitioned (range-resolution accuracy); length was
NOT -- a broadside wall has ~zero range change along its own length so
the old height-only dR_max check never subdivided it, even though it
can span many cross-range resolution cells and should image as an
extended streak, not a point. Each wall now gets a 2-D (length x height)
grid of sub-facets via the Cartesian product of the two per-axis
partitions.
"""
import numpy as np
from adaptive_facet_partition import merged_wall_partition


def make_multi_building_scene_adaptive(xp, footprint_m, density_per_km2, seed, o, dR_max=1.0,
                                        min_wh=(8.0, 8.0, 6.0), max_wh=(25.0, 25.0, 40.0),
                                        roof_refl_scale=1.3, delta_cr=2.0, wavelength=0.03):
    """
    xp: numpy or cupy module (backend-agnostic, matches the rest of the project)
    footprint_m: side length of the square scene (m)
    density_per_km2: buildings per km^2
    o: platform position (3,) numpy array used to evaluate per-wall visibility
       and range-sensitivity for the adaptive partition (use a representative
       pulse position, e.g. broadside)
    dR_max: target max equivalent-range change (m) between adjacent sub-facet
            centers, for whichever bounce order is locally more sensitive
            (criterion 1: range-resolution accuracy)
    delta_cr: target max cross-range (perpendicular-to-LOS) displacement (m)
              between adjacent sub-facet centers (criterion 2: cross-range
              resolution shape fidelity) -- match to the imaging system's
              actual cross-range resolution, e.g. --az-res
    wavelength: radar wavelength (m), used for the Fresnel/far-field bound
                sqrt(R*wavelength) (criterion 3)
    """
    rng = np.random.default_rng(seed)
    area_km2 = (footprint_m / 1000.0) ** 2
    n_buildings = max(1, int(round(density_per_km2 * area_km2)))

    cx = rng.uniform(-footprint_m / 2, footprint_m / 2, n_buildings)
    cy = rng.uniform(-footprint_m / 2, footprint_m / 2, n_buildings)
    w = rng.uniform(min_wh[0], max_wh[0], n_buildings)
    d = rng.uniform(min_wh[1], max_wh[1], n_buildings)
    h = rng.uniform(min_wh[2], max_wh[2], n_buildings)
    refl = rng.uniform(0.6, 1.0, n_buildings)

    centers, u_hats, v_hats, normals, half_u, half_v, amps = [], [], [], [], [], [], []
    facet_building_id = []
    total_facets_per_building = []

    for i in range(n_buildings):
        hx, hy, hz = w[i] / 2, d[i] / 2, h[i] / 2
        bcx, bcy = cx[i], cy[i]
        wall_defs = [
            ((hx, 0), (0, 1, 0), (1, 0, 0), hy),
            ((-hx, 0), (0, 1, 0), (-1, 0, 0), hy),
            ((0, hy), (1, 0, 0), (0, 1, 0), hx),
            ((0, -hy), (1, 0, 0), (0, -1, 0), hx),
        ]
        n_facets_this = 0
        for (xy_off, u_h, n_h, hu) in wall_defs:
            center_xy = np.array([bcx + xy_off[0], bcy + xy_off[1]])
            normal = np.array(n_h, dtype=float)

            # visibility gate: the mirror-reflection math used by the
            # order-2 range function gets numerically ill-conditioned for
            # a wall whose normal doesn't face the platform at all
            # (near-grazing/backfacing), which spuriously inflates the
            # measured range sensitivity there and over-subdivides a wall
            # that's actually invisible/contributing ~nothing.
            look = np.array([center_xy[0], center_xy[1], hz]) - o
            look = look / np.linalg.norm(look)
            visible = np.dot(-look, normal) > 0

            u_hat_arr = np.array(u_h, dtype=float)
            if visible:
                b_z, b_u = merged_wall_partition(o, center_xy, normal, u_hat_arr, h[i], 2 * hu,
                                                  dR_max=dR_max, delta_cr=delta_cr, wavelength=wavelength)
            else:
                b_z = np.array([1e-4, h[i] - 1e-4])       # single coarse facet, not illuminated anyway
                b_u = np.array([-hu + 1e-4, hu - 1e-4])

            n_facets_this += (len(b_z) - 1) * (len(b_u) - 1)
            for j in range(len(b_z) - 1):
                z_lo, z_hi = b_z[j], b_z[j + 1]
                z_c = 0.5 * (z_lo + z_hi)
                for k in range(len(b_u) - 1):
                    u_lo, u_hi = b_u[k], b_u[k + 1]
                    u_c = 0.5 * (u_lo + u_hi)
                    wall_center = np.array([center_xy[0], center_xy[1], 0.0]) + u_c * u_hat_arr
                    centers.append([wall_center[0], wall_center[1], z_c])
                    u_hats.append(u_h); v_hats.append((0, 0, 1)); normals.append(n_h)
                    half_u.append(0.5 * (u_hi - u_lo)); half_v.append(0.5 * (z_hi - z_lo)); amps.append(refl[i])
                    facet_building_id.append(i)

        # roof: single facet, unsubdivided (order-1 only, near-nadir --
        # out of scope for this fix, which targets vertical wall facets)
        centers.append([bcx, bcy, h[i]])
        u_hats.append((1, 0, 0)); v_hats.append((0, 1, 0)); normals.append((0, 0, 1))
        half_u.append(hx); half_v.append(hy); amps.append(refl[i] * roof_refl_scale)
        facet_building_id.append(i)
        total_facets_per_building.append(n_facets_this + 1)

    to_xp = lambda a: xp.asarray(np.array(a, dtype=np.float64))
    print(f"multi-building adaptive scene: {n_buildings} buildings, "
          f"facets/building={total_facets_per_building}, total={len(centers)}")
    return dict(
        center=to_xp(centers), u_hat=to_xp(u_hats), v_hat=to_xp(v_hats),
        normal=to_xp(normals), half_u=to_xp(half_u), half_v=to_xp(half_v),
        amp=to_xp(amps), n_buildings=n_buildings, n_facets=len(centers),
        building_cx=cx, building_cy=cy, building_w=w, building_d=d, building_h=h,
        facet_building_id=xp.asarray(np.array(facet_building_id, dtype=np.int64)),
    )
