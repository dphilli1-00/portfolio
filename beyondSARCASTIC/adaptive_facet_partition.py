"""General adaptive facet partitioning: subdivide a wall's HEIGHT axis and
LENGTH axis so that consecutive sub-facet centers satisfy a JOINT
resolution-cell criterion, for a GIVEN bounce order's range function --
not a fixed physical patch size, and not range accuracy alone.

Three independent criteria (identified/derived across this session, see
the project's own conversation history):

1. Range-resolution accuracy (the original, only-implemented-until-now
   check): consecutive sub-facet centers must differ by no more than
   delta_R in EQUIVALENT ROUND-TRIP RANGE, delta_R = c/(2*bandwidth) --
   otherwise a facet's own internal range spread exceeds one range-
   resolution cell and phase decorrelates across it.

2. Cross-range-resolution shape fidelity (NEW): consecutive sub-facet
   centers must ALSO differ by no more than delta_cr = lambda*R/(2*
   L_aperture) in CROSS-RANGE (perpendicular-to-line-of-sight)
   displacement -- otherwise a facet physically wider than one cross-
   range resolution cell still collapses to a single point-scatterer
   response in the image, which focuses to one resolution-cell-sized
   blob regardless of taper. That's a real object mis-rendered as a dot
   instead of the extended streak it should be -- an image STRUCTURE
   defect, not an amplitude/phase error, and criterion 1 alone does not
   catch it: a perfectly broadside-facing wall has ~zero range change
   along its own length (dR_max never triggers) while still spanning many
   cross-range resolution cells.

   These two are a JOINT check, not independent per-axis ones: a grazing
   (non-broadside) wall's length axis couples into BOTH range and cross-
   range simultaneously (the same coupling that made a single-axis
   angular ray grid fail to discover grazing-wall facets earlier this
   session) -- so both components are checked on WHICHEVER axis (height
   or length) is being partitioned, not range-only for height and cross-
   range-only for length.

3. Fresnel/far-field validity (NEW, independent of 1+2): a sub-facet's
   own physical size must stay under sqrt(R*lambda), the distance below
   which the far-field (Fraunhofer) taper formula this codebase's ASC
   closed form uses stops being a valid approximation at all -- same
   scale used earlier this session diagnosing the faceted-sphere
   "sparkle"/non-convergence problem. This is a validity-of-the-equation
   check, not an image-fidelity check, hence kept separate from 1+2.

Self-contained: includes _wall_L_theta (the exact 3-leg wall-ground
round-trip path length) directly, rather than importing it from the
edge-closed-form module, since that module's boundary-term formula has
a known unresolved singularity near broadside and isn't a dependency
this should drag in.
"""
import numpy as np


def _wall_L_theta(o, center_xy, normal, u_hat, u, z):
    """Exact 3-leg round-trip path length and ground incidence angle for a
    ray: platform o -> wall point (center_xy + u*u_hat, z) -> mirror off
    `normal` -> ground plane z=0 -> (retroreflection) -> platform.

    u, z: arrays of matching length -- position along the wall's own
    length axis (u_hat) and height, evaluated pointwise (u[i], z[i])."""
    u = np.atleast_1d(u).astype(float)
    z = np.atleast_1d(z).astype(float)
    n = max(len(u), len(z))
    u = np.broadcast_to(u, (n,))
    z = np.broadcast_to(z, (n,))
    C1 = np.tile(np.array([center_xy[0], center_xy[1], 0.0]), (n, 1))
    C1 = C1 + u[:, None] * u_hat[None, :]
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


def _wall_pos(center_xy, u_hat, u, z):
    """3-D position on a wall's plane at (u along u_hat, z height)."""
    p = np.array([center_xy[0], center_xy[1], 0.0]) + u * u_hat
    p[2] = z
    return p


def order1_range_fn(o, center_xy, u_hat, axis, fixed):
    """axis='z' varies height (fixed u); axis='u' varies length (fixed z)."""
    def f(t):
        u, z = (fixed, t) if axis == 'z' else (t, fixed)
        p = _wall_pos(center_xy, u_hat, u, z)
        return np.linalg.norm(o - p)
    return f


def order2_range_fn(o, center_xy, normal, u_hat, axis, fixed):
    def f(t):
        u, z = (fixed, t) if axis == 'z' else (t, fixed)
        return _wall_L_theta(o, center_xy, normal, u_hat, np.array([u]), np.array([z]))[0][0] / 2.0
    return f


def _pos_fn(center_xy, u_hat, axis, fixed):
    def f(t):
        u, z = (fixed, t) if axis == 'z' else (t, fixed)
        return _wall_pos(center_xy, u_hat, u, z)
    return f


def adaptive_joint_partition(range_fn, pos_fn, o, T, delta_R=1.0, delta_cr=2.0,
                              fresnel_limit=None, t_eps=1e-4):
    """t-boundaries such that consecutive sub-facet centers satisfy ALL of:
      - range change <= delta_R (criterion 1)
      - cross-range (perpendicular-to-LOS) displacement <= delta_cr (criterion 2)
      - physical step size <= fresnel_limit, if given (criterion 3)
    A joint check evaluated together at every candidate boundary, not
    three independently-triggered passes -- whichever is tightest at a
    given point determines where the next boundary actually lands.

    Cross-range displacement here is the FULL perpendicular-to-LOS
    component of the position delta (not decomposed further into a pure
    along-track vs. elevation-perpendicular split) -- a deliberately
    conservative approximation: it can only trigger MORE subdivision than
    a stricter along-track-only definition would, never less, which is
    the safe direction to be wrong in.
    """
    def cross_range_delta(p_cur, p_test):
        r_hat = o - p_cur
        r_hat = r_hat / np.linalg.norm(r_hat)
        delta = p_test - p_cur
        perp = delta - np.dot(delta, r_hat) * r_hat
        return np.linalg.norm(perp)

    boundaries = [t_eps]
    t_cur = t_eps
    R_cur = range_fn(t_cur)
    p_cur = pos_fn(t_cur)
    guard = 0
    while t_cur < T - t_eps and guard < 5000:
        guard += 1
        lo, hi = t_cur, T - t_eps

        def violates(t_test):
            if fresnel_limit is not None and (t_test - t_cur) > fresnel_limit:
                return True
            dR = abs(range_fn(t_test) - R_cur)
            if dR > delta_R:
                return True
            dcr = cross_range_delta(p_cur, pos_fn(t_test))
            return dcr > delta_cr

        if not violates(hi):
            boundaries.append(T - t_eps)
            break
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if violates(mid):
                hi = mid
            else:
                lo = mid
        t_next = lo
        boundaries.append(t_next)
        t_cur = t_next
        R_cur = range_fn(t_cur)
        p_cur = pos_fn(t_cur)
    return np.array(sorted(set(boundaries)))


def adaptive_range_partition(range_fn, H, dR_max=1.0, z_eps=1e-4):
    """Legacy range-only (criterion 1 alone) partition -- kept for
    backward compatibility / A-B comparison. New code should use
    adaptive_joint_partition, which also enforces cross-range and
    Fresnel bounds."""
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


def merged_wall_partition(o, center_xy, normal, u_hat, H, L,
                           dR_max=1.0, delta_cr=2.0, wavelength=None,
                           fresnel_margin=1.0):
    """Union of the order-1 and order-2 partitions, along BOTH the
    height (z, in [0,H]) and length (u, in [-L/2,L/2]) axes, each using
    the joint range+cross-range+Fresnel criterion (module docstring).
    Returns (boundaries_z, boundaries_u) -- a caller building a 2-D grid
    of sub-facets should take the Cartesian (tensor) product of the two,
    which is an approximation to a fully joint 2-D adaptive mesh (height
    and length partitioned independently, each holding the other axis at
    its center value) rather than a true 2-D adaptive solve -- a
    reasonable, much simpler compromise given the two axes are only
    loosely coupled except at strongly grazing incidence.

    fresnel_margin: safety factor (<1 tightens, >1 loosens) applied to
    sqrt(R*lambda) -- 1.0 uses the textbook far-field distance directly.
    """
    R_repr = np.linalg.norm(o - np.array([center_xy[0], center_xy[1], H / 2.0]))
    fresnel_limit = None
    if wavelength is not None:
        fresnel_limit = fresnel_margin * np.sqrt(R_repr * wavelength)

    # height axis (u held at wall center, u=0)
    r1z = order1_range_fn(o, center_xy, u_hat, axis='z', fixed=0.0)
    r2z = order2_range_fn(o, center_xy, normal, u_hat, axis='z', fixed=0.0)
    pz = _pos_fn(center_xy, u_hat, axis='z', fixed=0.0)
    b1z = adaptive_joint_partition(r1z, pz, o, H, dR_max, delta_cr, fresnel_limit)
    b2z = adaptive_joint_partition(r2z, pz, o, H, dR_max, delta_cr, fresnel_limit)
    boundaries_z = np.array(sorted(set(np.concatenate([b1z, b2z, [1e-4, H - 1e-4]]))))

    # length axis (z held at wall mid-height) -- parametrized u in
    # [-L/2, L/2]; adaptive_joint_partition works over [0,T], so shift.
    z_mid = H / 2.0
    r1u = order1_range_fn(o, center_xy, u_hat, axis='u', fixed=z_mid)
    r2u = order2_range_fn(o, center_xy, normal, u_hat, axis='u', fixed=z_mid)
    pu = _pos_fn(center_xy, u_hat, axis='u', fixed=z_mid)
    r1u_shift = lambda t: r1u(t - L / 2.0)
    r2u_shift = lambda t: r2u(t - L / 2.0)
    pu_shift = lambda t: pu(t - L / 2.0)
    b1u = adaptive_joint_partition(r1u_shift, pu_shift, o, L, dR_max, delta_cr, fresnel_limit)
    b2u = adaptive_joint_partition(r2u_shift, pu_shift, o, L, dR_max, delta_cr, fresnel_limit)
    boundaries_u = np.array(sorted(set(np.concatenate([b1u, b2u, [1e-4, L - 1e-4]])))) - L / 2.0

    return boundaries_z, boundaries_u


if __name__ == "__main__":
    standoff, altitude = 8000.0, 3000.0
    o = np.array([0.0, -standoff, altitude])
    center_xy = np.array([0.0, -10.0])
    normal = np.array([0.0, -1.0, 0.0])
    u_hat = np.array([1.0, 0.0, 0.0])
    wavelength = 0.03   # 10 GHz

    for H, L in [(40.0, 25.0), (400.0, 100.0)]:
        for dR_max, delta_cr in [(1.0, 2.0), (0.25, 2.0)]:
            bz, bu = merged_wall_partition(o, center_xy, normal, u_hat, H, L,
                                            dR_max=dR_max, delta_cr=delta_cr, wavelength=wavelength)
            print(f"H={H:6.1f}m L={L:6.1f}m dR_max={dR_max:.2f}m delta_cr={delta_cr:.1f}m  "
                  f"height->{len(bz)-1:3d} slices  length->{len(bu)-1:3d} slices  "
                  f"total 2D facets->{(len(bz)-1)*(len(bu)-1):4d}")
