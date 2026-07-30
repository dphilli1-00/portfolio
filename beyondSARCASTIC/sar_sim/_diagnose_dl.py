"""Sanity check: how big is dL (round-trip path-length spread between a
wall's top and bottom reflecting to the ground) actually, for real
facets in the 300m scene? If it's absurdly large (hundreds of meters)
that's a bug, not physics -- a real building's height-induced spread
should be on the order of the building height itself times a modest
geometric factor, not orders of magnitude more."""
import numpy as np
from dense_sbr_demo import make_building_scene, make_ground_facet, get_backend
import multibounce_demo as mb

xp, on_gpu = get_backend(False)
footprint, density = 300.0, 200.0
standoff, altitude = 8000.0, 3000.0

facets_b = make_building_scene(xp, footprint, density, seed=0)
facets_g = make_ground_facet(xp, footprint, material='dry_soil')
half_extent_g = float(facets_g['footprint_half_extent'])

Cb, Nb = facets_b['center'], facets_b['normal']
HVb = facets_b['half_v']
o = xp.asarray([0.0, -standoff, altitude])

look = Cb - o[None, :]
R_asc = xp.linalg.norm(look, axis=1)
d_in = look / R_asc[:, None]
cos_inc1 = xp.sum(-d_in * Nb, axis=1)
visible1 = cos_inc1 > 0

C1_bottom = Cb.copy(); C1_bottom[:, 2] = Cb[:, 2] - HVb
C1_top = Cb.copy(); C1_top[:, 2] = Cb[:, 2] + HVb
G_bot, valid_bot, _ = mb._reflect_and_intersect_ground_plane(xp, C1_bottom, Nb, d_in, half_extent_g)
G_top, valid_top, _ = mb._reflect_and_intersect_ground_plane(xp, C1_top, Nb, d_in, half_extent_g)

o_b = o[None, :]
L_bot = xp.linalg.norm(o_b - C1_bottom, axis=1) + xp.linalg.norm(C1_bottom - G_bot, axis=1) + xp.linalg.norm(o_b - G_bot, axis=1)
L_top = xp.linalg.norm(o_b - C1_top, axis=1) + xp.linalg.norm(C1_top - G_top, axis=1) + xp.linalg.norm(o_b - G_top, axis=1)
dL = L_top - L_bot

mask_strict = mb.to_numpy(visible1 & valid_bot & valid_top)
mask_visible_only = mb.to_numpy(visible1)
print(f"visible1: {mask_visible_only.sum()}  valid_bot: {int(mb.to_numpy(valid_bot).sum())}  "
      f"valid_top: {int(mb.to_numpy(valid_top).sum())}  both+visible: {mask_strict.sum()}")

# use visible-only (ignore in_bounds) so we can see the raw geometry even
# where the top reflection point lands outside the modeled ground extent
mask = mask_visible_only
dL_np = mb.to_numpy(dL)[mask]
H_np = mb.to_numpy(2 * HVb)[mask]
wavelength = 0.03  # ~X-band

print(f"\n{mask.sum()} facets (visible1 only, ignoring ground in_bounds)")
print(f"wall heights H (m): min={H_np.min():.1f} max={H_np.max():.1f} mean={H_np.mean():.1f}")
print(f"dL (m): min={dL_np.min():.3f} max={dL_np.max():.3f} mean={dL_np.mean():.3f}")
print(f"dL / H ratio: min={np.min(dL_np/H_np):.3f} max={np.max(dL_np/H_np):.3f} mean={np.mean(dL_np/H_np):.3f}")
print(f"dL in wavelengths (lambda=3cm): min={dL_np.min()/wavelength:.1f} max={dL_np.max()/wavelength:.1f} mean={dL_np.mean()/wavelength:.1f}")
arg_mean_wavelength = np.pi * dL_np / wavelength
print(f"scalar-taper arg (pi*dL/lambda): min={arg_mean_wavelength.min():.1f} max={arg_mean_wavelength.max():.1f}")
taper_vals = np.where(np.abs(arg_mean_wavelength) < 1e-9, 1.0, np.sin(arg_mean_wavelength)/arg_mean_wavelength)
print(f"resulting scalar taper values: min={taper_vals.min():.4f} max={taper_vals.max():.4f} mean={np.abs(taper_vals).mean():.4f}")
