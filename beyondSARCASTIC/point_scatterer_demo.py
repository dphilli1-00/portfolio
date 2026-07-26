"""
Minimal point-scatterer SAR simulator + time-domain backprojection.

Purpose: produce concrete, real (if toy) SAR imagery for the deck --
not a facet/SBR ray tracer, just the classic textbook forward model:

    scene = set of point scatterers (amplitude, position[, velocity])
    platform flies a linear aperture, wideband stepped-frequency waveform
    phase history:  s(f, u) = sum_i A_i * exp(-j 4 pi f R_i(u) / c)
    image formation: time-domain backprojection onto a ground grid

Two scenes:
  1. static target + clutter          -> well-focused image
  2. same scene + one moving scatterer -> smeared/displaced return,
     illustrating why GMTI needs special handling (ties to deck slide
     on SAR -> GMTI extension)

Also includes a timing sweep (image size x pulse count) to get real
wall-clock numbers for "what's achievable on a standard desktop."
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt

from materials import diffuse_coefficient

C = 299_792_458.0


# ----------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------
def make_target_scatterers(rng):
    """Sparse, ASC-like discrete scattering centers (not a dense periodic
    outline -- evenly-spaced collinear points alias into grating-lobe
    ridges under coherent SAR processing, which is a real phenomenon but
    a distracting artifact for an illustrative example). Roughly
    MSTAR-target-sized (~5m x 3m) boxy vehicle, all scatterers at z=0 to
    avoid layover complicating this particular figure."""
    L, W = 5.0, 3.0
    pts = [
        (-L / 2, -W / 2, 0.0, 3.2),   # corner reflectors
        (-L / 2,  W / 2, 0.0, 2.8),
        ( L / 2, -W / 2, 0.0, 3.0),
        ( L / 2,  W / 2, 0.0, 3.4),
        ( 0.0,    0.0,   0.0, 2.2),   # turret-like body scatterer
        (-L / 4,  0.0,   0.0, 0.9),   # weaker mid-body edges
        ( L / 4,  0.0,   0.0, 0.9),
        ( 0.0,   -W / 2, 0.0, 0.7),
        ( 0.0,    W / 2, 0.0, 0.7),
    ]
    # small jitter so nothing lands on an exact periodic lattice
    pts = np.array(pts)
    pts[:, :2] += rng.normal(0, 0.03, size=(len(pts), 2))
    return pts  # columns: x, y, z, amplitude


def make_clutter(rng, extent=40.0, n=400, material='concrete', mean_amp=None):
    """Sparse random ground clutter, Rayleigh-ish amplitudes.

    mean_amp now defaults to materials.diffuse_coefficient(material)
    (0.03 for concrete) instead of an arbitrary 0.05 -- reuses the
    Fresnel/diffuse material model already sitting in materials.py rather
    than a guessed constant. Pass mean_amp explicitly to override."""
    if mean_amp is None:
        mean_amp = diffuse_coefficient(material)
    xy = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    z = np.zeros(n)
    amp = rng.rayleigh(mean_amp, size=n)
    return np.column_stack([xy, z, amp])


# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------
def synth_aperture(n_pulses, standoff=8000.0, altitude=3000.0, squint_len=400.0):
    """Straight-line flight path, broadside-ish stripmap geometry."""
    u = np.linspace(-squint_len / 2, squint_len / 2, n_pulses)
    plat = np.column_stack([
        u,
        np.full(n_pulses, -standoff),
        np.full(n_pulses, altitude),
    ])
    return plat  # (n_pulses, 3)


def phase_history(scatterers, plat_pos, freqs, ref_pos=(0.0, 0.0, 0.0),
                   times=None, velocities=None):
    """
    scatterers: (N,4) x,y,z,amp   (static case)
    plat_pos:   (P,3) platform xyz per pulse
    freqs:      (K,)  stepped-frequency samples across the band
    ref_pos:    scene-center reference point for motion compensation
    times:      (P,)  slow time per pulse (needed only if velocities given)
    velocities: (N,3) per-scatterer velocity, optional (movers)

    returns: (P, K) complex phase history

    NOTE: this does motion compensation to ref_pos before forming the
    phase history -- i.e. phase is referenced to (R_i - R_ref), not the
    raw (thousands-of-meters) absolute range. This is standard practice
    in real stepped-frequency/spotlight SAR processors: without it, the
    absolute range vastly exceeds the unambiguous range window set by
    the frequency step size, and range compression aliases into noise.
    """
    P = plat_pos.shape[0]
    K = freqs.shape[0]
    pos = scatterers[:, :3]
    amp = scatterers[:, 3]
    ref_pos = np.asarray(ref_pos)

    R_ref = np.linalg.norm(plat_pos - ref_pos, axis=1)  # (P,)

    s = np.zeros((P, K), dtype=np.complex128)
    for p in range(P):
        if velocities is not None:
            t = times[p]
            cur_pos = pos + velocities * t
        else:
            cur_pos = pos
        R = np.linalg.norm(cur_pos - plat_pos[p], axis=1)  # (N,)
        dR = R - R_ref[p]
        # (K,N) phase, summed over N -> (K,)
        phase = np.exp(-1j * 4.0 * np.pi * np.outer(freqs, dR) / C)
        s[p, :] = (amp[None, :] * phase).sum(axis=1)
    return s, R_ref


# ----------------------------------------------------------------------
# Image formation: time-domain backprojection
# ----------------------------------------------------------------------
def backproject(s, plat_pos, freqs, grid_x, grid_y, fc, R_ref):
    """
    s: (P,K) mocomp'd phase history (referenced to R_ref per pulse)
    range-compress each pulse (IFFT over frequency -> range profile),
    then backproject onto (grid_x, grid_y) ground plane at z=0.

    Matched filter: interpolate the range-compressed sample at each
    pixel's *reference-relative* range, then reapply the carrier phase
    for that same relative range (conjugate of the forward model) to
    coherently combine across pulses. Using the pixel's *absolute*
    range here instead of the reference-relative range is the classic
    off-by-a-huge-number bug -- it reintroduces exactly the phase term
    mocomp was meant to remove and turns the image into noise.
    """
    P, K = s.shape
    B = freqs[-1] - freqs[0]
    range_res = C / (2 * B)

    # Hamming weighting in range (frequency) and cross-range (aperture) to
    # suppress sinc sidelobes from the uniformly-illuminated band/aperture --
    # standard practice in real SAR processors, not just cosmetic.
    range_win = np.hamming(K)
    az_win = np.hamming(P)
    range_profile = np.fft.fftshift(np.fft.ifft(s * range_win[None, :], axis=1), axes=1)
    range_axis = (np.arange(K) - K // 2) * range_res

    Xg, Yg = np.meshgrid(grid_x, grid_y)
    img = np.zeros(Xg.shape, dtype=np.complex128)

    for p in range(P):
        dx = Xg - plat_pos[p, 0]
        dy = Yg - plat_pos[p, 1]
        dz = 0.0 - plat_pos[p, 2]
        R_pixel_abs = np.sqrt(dx**2 + dy**2 + dz**2)
        dR_pixel = R_pixel_abs - R_ref[p]
        samp = np.interp(dR_pixel, range_axis, range_profile[p, :].real) \
            + 1j * np.interp(dR_pixel, range_axis, range_profile[p, :].imag)
        img += az_win[p] * samp * np.exp(1j * 4.0 * np.pi * fc * dR_pixel / C)

    return img


# ----------------------------------------------------------------------
# Scene runner
# ----------------------------------------------------------------------
def run_scene(mover=False, n_pulses=200, n_freq=128, img_size=256,
              fc=10e9, bandwidth=600e6, seed=0):
    rng = np.random.default_rng(seed)
    target = make_target_scatterers(rng)
    clutter = make_clutter(rng)
    scene = np.vstack([target, clutter])

    plat = synth_aperture(n_pulses)
    freqs = fc + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq)

    velocities = None
    times = None
    if mover:
        # dedicated moving scatterer, offset from the target so its smear
        # isn't masked by the strong body/turret return; 3 m/s cross-range
        mover_pt = np.array([[0.0, 6.0, 0.0, 3.5]])
        scene = np.vstack([scene, mover_pt])
        velocities = np.zeros((scene.shape[0], 3))
        velocities[-1] = [3.0, 0.0, 0.0]
        times = np.linspace(-1.5, 1.5, n_pulses)  # 3 s dwell

    t0 = time.perf_counter()
    s, R_ref = phase_history(scene, plat, freqs, times=times, velocities=velocities)
    t1 = time.perf_counter()

    extent = 20.0
    grid = np.linspace(-extent / 2, extent / 2, img_size)
    img = backproject(s, plat, freqs, grid, grid, fc, R_ref)
    t2 = time.perf_counter()

    return img, dict(phase_history_s=t1 - t0, backproject_s=t2 - t1,
                      n_pulses=n_pulses, n_freq=n_freq, img_size=img_size)


def save_image(img, path, title):
    mag = np.abs(img)
    db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.imshow(db, cmap='gray', vmin=-40, vmax=0, origin='lower',
              extent=[-10, 10, -10, 10])
    ax.set_title(title, fontsize=12.5, wrap=True)
    ax.set_xlabel('cross-range (m)')
    ax.set_ylabel('range (m)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    out_dir = '.'

    print('Rendering static scene...')
    img_static, timing_static = run_scene(mover=False, n_pulses=200, img_size=256)
    save_image(img_static, f'{out_dir}/sar_static_example.png',
               'Simulated SAR chip -- static target + ground clutter')
    print(timing_static)

    print('Rendering mover scene...')
    img_mover, timing_mover = run_scene(mover=True, n_pulses=200, img_size=256)
    save_image(img_mover, f'{out_dir}/sar_mover_example.png',
               'Same scene + one moving scatterer (3 m/s) -- smear/displacement')
    print(timing_mover)

    print('Timing sweep (image size x pulse count)...')
    sweep = []
    for img_size in (128, 256, 512):
        for n_pulses in (100, 200, 400):
            _, timing = run_scene(mover=False, n_pulses=n_pulses,
                                   img_size=img_size)
            total = timing['phase_history_s'] + timing['backproject_s']
            sweep.append(dict(total_s=total, **timing))
            print(f"  {img_size}x{img_size}, {n_pulses} pulses -> "
                  f"{total:.3f} s ({1/total:.2f} Hz)")

    with open(f'{out_dir}/timing_sweep.json', 'w') as f:
        json.dump(sweep, f, indent=2)

    print('Done.')
