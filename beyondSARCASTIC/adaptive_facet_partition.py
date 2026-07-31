"""General adaptive facet partitioning: subdivide a wall's HEIGHT axis
so that consecutive sub-facet centers differ by at most dR_max in
EQUIVALENT ROUND-TRIP RANGE, for a GIVEN bounce order's range function --
not a fixed physical patch size. Order-1 (direct single bounce) and
order-2 (wall-ground double bounce) have wildly different range-vs-height
sensitivity on the same wall (measured: order-1 needs ~3m steps, order-2
at broadside barely needs any), so this builds each order's own
partition and MERGES the breakpoints -- the resulting mesh is provably
fine enough for BOTH orders simultaneously, using no more facets than
necessary for whichever order is locally more sensitive.

Self-contained: includes _wall_L_theta (the exact 3-leg wall-ground
round-trip path length) directly, rather than importing it from the
edge-closed-form module, since that module's boundary-term formula has
a known unresolved singularity near broadside and isn't a dependency
this should drag in.
"""
import numpy as np


def _wall_L_theta(o, center_xy, normal, z):
    """Exact 3-leg round-trip path length and ground incidence angle for a
    ray: platform o -> wall point (center_xy, z) -> mirror off `normal` ->
    ground plane z=0 -> (retroreflection) -> platform."""
    z = np.atleast_1d(z).astype(float)
    n = len(z)
    C1 = np.tile(np.array([center_xy[0], center_xy[1], 0.0]), (n, 1))
    C1[:, 2] = z
    d_in = C1 - o[None, :]
    d_in = d_in / np.linalg.norm(d_in, axis=1, keepdims=True)
    d_out = d_in - 2.0 * np.sum(d_in * normal[None, :], axis=1, keepdims=True) * normal[None, :]
    t = -C1[:, 2] / d_out[:, 2]
    G = C1 + t[:, None] * d_out
    L = (np.linalg.norm(o[None, :] - C1, axis=1)
         + np.linalg.norm(C1 - G, axis=1)
         + np.linalg.norm(o[None, :] - G, axis=1))
    cos_wall = np.abs(np.sum(-d_in * normal[None, :], axis=1))
    cos_ground = np.abs(d_out[:, 2])
    theta_ground = np.arccos(np.clip(cos_ground, 0.0, 1.0))
    return L, cos_wall, theta_ground, cos_ground


def order1_range_fn(o, center_xy):
    def f(z):
        p = np.array([center_xy[0], center_xy[1], z])
        return np.linalg.norm(o - p)
    return f


def order2_range_fn(o, center_xy, normal):
    def f(z):
        return _wall_L_theta(o, center_xy, normal, np.array([z]))[0][0] / 2.0
    return f


def adaptive_range_partition(range_fn, H, dR_max=1.0, z_eps=1e-4):
    """z-boundaries [z_eps, ..., H-z_eps] such that |R(z_i+1)-R(z_i)| <= dR_max."""
    boundaries = [z_eps]
    z_cur = z_eps
    R_cur = range_fn(z_cur)
    guard = 0
    while z_cur < H - z_eps and guard < 5000:
        guard += 1
        lo, hi = z_cur, H - z_eps
        R_hi = range_fn(hi)
        if abs(R_hi - R_cur) <= dR_max:
            boundaries.append(H - z_eps)
            break
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            R_mid = range_fn(mid)
            if abs(R_mid - R_cur) > dR_max:
                hi = mid
            else:
                lo = mid
        z_next = lo
        boundaries.append(z_next)
        z_cur = z_next
        R_cur = range_fn(z_cur)
    return np.array(sorted(set(boundaries)))


def merged_wall_partition(o, center_xy, normal, H, dR_max=1.0):
    """Union of the order-1 and order-2 partitions -- fine enough for
    whichever bounce order is locally more sensitive, at every height."""
    b1 = adaptive_range_partition(order1_range_fn(o, center_xy), H, dR_max)
    b2 = adaptive_range_partition(order2_range_fn(o, center_xy, normal), H, dR_max)
    merged = np.array(sorted(set(np.concatenate([b1, b2, [1e-4, H - 1e-4]]))))
    return merged, b1, b2


if __name__ == "__main__":
    standoff, altitude = 8000.0, 3000.0
    o = np.array([0.0, -standoff, altitude])
    center_xy = np.array([0.0, -10.0])
    normal = np.array([0.0, -1.0, 0.0])

    for H in [40.0, 400.0]:
        for dR_max in [1.0, 0.25]:
            merged, b1, b2 = merged_wall_partition(o, center_xy, normal, H, dR_max=dR_max)
            print(f"H={H:6.1f}m dR_max={dR_max:.2f}m  order1->{len(b1)-1:3d} facets  "
                  f"order2->{len(b2)-1:3d} facets  MERGED->{len(merged)-1:3d} facets")
