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


# RMS surface height (m), per material -- representative, not measured.
# This is a DIFFERENT quantity from DIFFUSE_COEFF above: DIFFUSE_COEFF sets
# how bright a purely-incoherent, isotropic point-scatterer background is
# (used by make_ground_clutter for diffuse terrain); ROUGHNESS_SIGMA_M
# governs how much of a smooth Fresnel reflection off an actual traced
# FACET survives as a coherent, specular (forward-scattering) return
# rather than being scattered away -- only meaningful for something ray
# tracing can bounce off of (e.g. multibounce_demo's ground facet), not
# for the diffuse point-scatterer layer, which has no facet/normal to
# even define a specular direction against.
ROUGHNESS_SIGMA_M = {
    "dry_soil": 0.012,   # ~1.2cm rms height -- tilled/bare soil, not smooth
    "concrete": 0.0015,  # ~1.5mm rms height -- finished concrete/pavement
    "metal": 0.0002,     # optically smooth at radar wavelengths
}


def specular_scatter_factor(material: str, theta_i: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Modified-Rayleigh coherent-scattering factor: the fraction of the bare
    Fresnel-reflected energy that stays in the coherent, specular
    (forward-scatter / mirror) direction, versus being diffusely
    scattered away in random directions, given the surface's RMS height
    roughness relative to wavelength AND local incidence angle:

        rho = exp( -(4*pi*sigma*cos(theta_i) / wavelength)^2 )

    rho -> 1: this surface, at this incidence angle and wavelength,
    behaves like a coherent mirror -- the full Fresnel reflectivity is
    available to a specular ray-traced bounce path (e.g. the ground leg
    of a wall-ground dihedral).
    rho -> 0: the surface is rough relative to wavelength at this
    incidence angle -- most of the energy is incoherently scattered in
    directions a specular ray trace can't represent, so a coherent
    multi-bounce path through this facet should contribute close to
    nothing here, even though fresnel_reflectivity() for the bare
    material is unchanged.

    Deliberately angle-dependent (cos(theta_i) in the exponent): the SAME
    rough surface looks smoother at grazing incidence than near-normal
    incidence -- this is the real, well-known angle dependence of the
    Rayleigh roughness criterion (referenced conceptually on the deck's
    Step 1 slide), not a convenience simplification.
    """
    sigma = ROUGHNESS_SIGMA_M[material]
    return np.exp(-((4.0 * np.pi * sigma * np.cos(theta_i)) / wavelength) ** 2)


def effective_specular_reflectivity(material: str, theta_i: np.ndarray, wavelength: float) -> np.ndarray:
    """fresnel_reflectivity() alone answers 'how reflective is this
    material'; multiplying by specular_scatter_factor() answers the
    question that actually matters for a ray-traced specular bounce:
    'how much of that reflection survives as a coherent forward-scatter
    return, at this specific angle/wavelength/roughness, instead of
    being lost to diffuse scattering'."""
    return fresnel_reflectivity(material, theta_i) * specular_scatter_factor(material, theta_i, wavelength)
