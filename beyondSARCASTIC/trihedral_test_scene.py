"""3 explicit towers positioned to form a right-angle courtyard, purpose-
built to stress-test order-3 (triple bounce) instead of hoping a random
dense scene happens to contain a usable one.

Layout: towers at (0,gap), (gap,0), (gap,gap) -- an L/triangle courtyard
opening toward the origin (and toward the sensor, south of it). Every
adjacent pair's inward-facing walls meet near 90 degrees; together with the
ground plane, each pair is a textbook trihedral corner (2 perpendicular
vertical panels + 1 horizontal panel retroreflects across a wide angle
range, the same principle real SAR corner-reflector calibration targets
use). With 3 towers instead of 2, there are three such corners (A-B, A-C,
B-C) plus straight-through paths across all three, giving order-3 many
more chances to actually occur than a random scene where wall pairs rarely
line up square to each other.

Reuses the exact per-wall construction from scene_multi_building_adaptive
(adaptive height partition, per-wall visibility gate, roof facet) --
deliberately not re-derived -- just with hand-picked cx/cy/w/d/h instead of
random ones.
"""
import numpy as np
from adaptive_facet_partition import merged_wall_partition


def make_trihedral_test_scene(xp, o, dR_max=1.0, gap=20.0, tower_wh=(10.0, 10.0, 25.0),
                               refl=0.9, roof_refl_scale=1.3):
    w, d, h = tower_wh
    cx = np.array([0.0, gap, gap])
    cy = np.array([gap, 0.0, gap])
    ww = np.full(3, w); dd = np.full(3, d); hh = np.full(3, h); rf = np.full(3, refl)

    centers, u_hats, v_hats, normals, half_u, half_v, amps = [], [], [], [], [], [], []
    facet_building_id = []
    total_facets_per_building = []
    for i in range(3):
        hx, hy, hz = ww[i] / 2, dd[i] / 2, hh[i] / 2
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
            look = np.array([center_xy[0], center_xy[1], hz]) - o
            look = look / np.linalg.norm(look)
            visible = np.dot(-look, normal) > 0
            if visible:
                boundaries, _, _ = merged_wall_partition(o, center_xy, normal, hh[i], dR_max=dR_max)
            else:
                boundaries = np.array([1e-4, hh[i] - 1e-4])
            n_facets_this += len(boundaries) - 1
            for j in range(len(boundaries) - 1):
                z_lo, z_hi = boundaries[j], boundaries[j + 1]
                z_c = 0.5 * (z_lo + z_hi)
                centers.append([center_xy[0], center_xy[1], z_c])
                u_hats.append(u_h); v_hats.append((0, 0, 1)); normals.append(n_h)
                half_u.append(hu); half_v.append(0.5 * (z_hi - z_lo)); amps.append(rf[i])
                facet_building_id.append(i)
        centers.append([bcx, bcy, hh[i]])
        u_hats.append((1, 0, 0)); v_hats.append((0, 1, 0)); normals.append((0, 0, 1))
        half_u.append(hx); half_v.append(hy); amps.append(rf[i] * roof_refl_scale)
        facet_building_id.append(i)
        total_facets_per_building.append(n_facets_this + 1)

    to_xp = lambda a: xp.asarray(np.array(a, dtype=np.float64))
    print(f"trihedral test scene: 3 towers, facets/tower={total_facets_per_building}, "
          f"total={len(centers)}")
    return dict(
        center=to_xp(centers), u_hat=to_xp(u_hats), v_hat=to_xp(v_hats),
        normal=to_xp(normals), half_u=to_xp(half_u), half_v=to_xp(half_v),
        amp=to_xp(amps), n_buildings=3, n_facets=len(centers),
        building_cx=cx, building_cy=cy, building_w=ww, building_d=dd, building_h=hh,
        facet_building_id=xp.asarray(np.array(facet_building_id, dtype=np.int64)),
    )
