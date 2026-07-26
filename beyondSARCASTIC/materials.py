"""
Material EM properties and Fresnel reflectivity.

Scalar (unpolarized) reflectivity only -- this is the Tier-1 MVP simplification
called out in the slide outline (Slide 10/14). Swapping this for the full 2x2
Sinclair matrix (separate R_s / R_p, rotated into the sensor H/V basis, chained
via matrix product across bounces) is the natural Phase-2 upgrade.
"""

import numpy as np

# Relative complex permittivity eps_r = eps' - j*eps'' at ~X-band (10 GHz).
# Values are representative, not measured -- swap in real numbers if you have them.
MATERIALS = {
    "dry_soil": 4.0 - 0.5j,
    "concrete": 6.0 - 1.0j,
    "metal": None,  # special-cased below as a near-perfect conductor
}

# Simple diffuse (rough-surface) coefficient per material, standing in for the
# empirical sigma-0 clutter model from the outline (Slide 11). Not physically
# rigorous -- no speckle statistics, just enough to keep flat terrain from
# vanishing to literal zero under pure specular treatment. See run_demo.py notes.
DIFFUSE_COEFF = {
    "dry_soil": 0.09,
    "concrete": 0.03,
    "metal": 0.01,
}


def fresnel_reflectivity(material: str, theta_i: np.ndarray) -> np.ndarray:
    """
    Scalar reflectivity magnitude vs incidence angle (radians), measured from
    the surface normal. theta_i can be an array.

    Standard Fresnel form for a lossy dielectric half-space (non-magnetic,
    mu_r=1), avoiding an explicit complex refraction angle:

        R_perp = (cos(ti) - sqrt(eps_r - sin(ti)^2)) / (cos(ti) + sqrt(eps_r - sin(ti)^2))
        R_par  = (eps_r*cos(ti) - sqrt(eps_r - sin(ti)^2)) / (eps_r*cos(ti) + sqrt(eps_r - sin(ti)^2))

    Returned value is the average of |R_perp| and |R_par| -- a scalar stand-in
    until Phase 2 tracks them separately for polarimetry.
    """
    if material == "metal":
        # Near-perfect conductor: high, roughly angle-independent reflectivity.
        return np.full_like(theta_i, 0.95, dtype=np.float64)

    eps_r = MATERIALS[material]
    cos_ti = np.cos(theta_i)
    sin_ti = np.sin(theta_i)
    root = np.sqrt(eps_r - sin_ti**2 + 0j)  # complex sqrt

    r_perp = (cos_ti - root) / (cos_ti + root)
    r_par = (eps_r * cos_ti - root) / (eps_r * cos_ti + root)

    return 0.5 * (np.abs(r_perp) + np.abs(r_par))


def diffuse_coefficient(material: str) -> float:
    return DIFFUSE_COEFF[material]
