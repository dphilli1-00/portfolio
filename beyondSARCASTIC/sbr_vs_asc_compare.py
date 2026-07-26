"""
Head-to-head, same-scene, same-hardware comparison: dense SBR (per-pulse
ray-facet intersection) vs. a pre-computed/cached ASC-style model, on the
identical building scene. This is the deck's central claim (n* break-even,
speedup-vs-scale, generation-speed slides) actually measured on real
hardware instead of modeled from cited benchmarks.

Both methods share the SAME facets from dense_sbr_demo.make_building_scene
-- that's what makes this a fair comparison, not two different scenes.

Dense SBR (per pulse):
    cast a ray grid, test every ray against every facet, keep nearest hit.
    Cost ~ O(rays x facets), repeated fresh every pulse.
    (see dense_sbr_demo.py for the ray-facet intersection itself)

ASC-cached (one-time build, then per pulse):
    BUILD (once): read each facet's center position, normal, and
    reflectivity directly from the known geometry -- alpha_k = 0 (flat
    plate, per the deck's canonical scattering-type table). In a real
    pipeline this would come from a ray-traced aspect sweep + CLEAN/
    matrix-pencil fit; here the geometry is already known, so consistent
    with the deck's own "position/length read from mesh -- effectively
    free" framing, the build step is just a data read.
    PER PULSE: direct closed-form phase evaluation at each facet's fixed
    center position, no ray casting, no per-pulse intersection test.
    A cheap backface cull (facet normal vs. look direction) replaces
    "does a ray actually reach this facet" -- O(facets), not O(rays x
    facets). Cost ~ O(facets), independent of ray density entirely.

Usage:
    python3 sbr_vs_asc_compare.py --gpu --footprint 1000 --density 200 \
        --rays 100 --pulses 200 --freq 64
"""

import argparse
import time
import json
import numpy as np

from dense_sbr_demo import make_building_scene, ray_facet_intersect, C, get_backend, make_aim_grid


def run_asc_cached(xp, on_gpu, facets, plat, freqs, ref_pos):
    """
    ASC-cached forward model on the SAME facets as the dense SBR run.
    Per-pulse cost: O(facets), no ray casting.
    """
    n_pulses = plat.shape[0]
    K = freqs.shape[0]
    F = facets['center'].shape[0]

    C_ = facets['center']    # (F,3)
    N_ = facets['normal']    # (F,3)
    AMP = facets['amp']      # (F,)

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    t_build_total = 0.0
    t_eval_total = 0.0

    # ---- BUILD (once): nothing to compute -- position/normal/amplitude
    # already known from geometry. Timed anyway, for honesty, even though
    # it's just an array read (mirrors the deck's "effectively free"
    # closed-form-assignment claim -- show the zero, don't just assert it).
    t0 = time.perf_counter()
    asc_pos = C_
    asc_normal = N_
    asc_amp = AMP
    if on_gpu:
        xp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    t_build_total = t1 - t0

    for p in range(n_pulses):
        o = plat[p]

        t0 = time.perf_counter()
        look = asc_pos - o[None, :]
        look = look / xp.linalg.norm(look, axis=1, keepdims=True)
        # backface cull: only facets facing the platform contribute --
        # O(facets), replaces dense SBR's O(rays x facets) visibility test
        cos_inc = xp.sum(-look * asc_normal, axis=1)
        visible = cos_inc > 0
        amp_eff = xp.where(visible, asc_amp * xp.abs(cos_inc), 0.0)

        R_ref = xp.linalg.norm(o - ref_pos)
        R_asc = xp.linalg.norm(asc_pos - o[None, :], axis=1)
        dR = R_asc - R_ref

        phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)   # (K,F)
        s[p, :] = (amp_eff[None, :] * phase).sum(axis=1)
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        t_eval_total += (t1 - t0)

    return s, dict(n_facets=F, t_build_s=t_build_total, t_eval_s=t_eval_total,
                    t_total_s=t_build_total + t_eval_total,
                    t_per_pulse_ms=t_eval_total / n_pulses * 1000.0)


def run_dense_sbr_timed(xp, on_gpu, facets, plat, aim_pts, freqs, ref_pos):
    """
    IMPORTANT correctness note (found while building the Tier 2 validation,
    not present in the original per-ray version of this function): a dense
    ray grid puts MANY rays on the same facet. Summing one phase term per
    RAY (as the earlier version did) coherently sums that facet ~N times
    for N incident rays -- amplitude scales with ray density, not with
    physical facet reflectivity, so the result doesn't converge as --rays
    increases and isn't comparable to ASC's one-term-per-facet convention
    at all (measured: near-zero correlation between the two phase
    histories, SSIM ~0.04, garbage).

    Fix: rays are cast ONLY to determine visibility/occlusion (a facet is
    "seen" this pulse iff at least one ray's nearest hit lands on it --
    this is real, expensive, O(rays x facets) work, still fully timed
    below). The scattering CONTRIBUTION per visible facet then uses the
    same one-point-per-facet closed-form convention as ASC (facet center,
    analytic cos_inc), not a per-ray sum. This isolates exactly the
    physical effect worth testing: does ray-traced occlusion (a facet
    blocked by nearer geometry -- its own building's other walls, or a
    different building) exclude facets that ASC's naive front-face-only
    cull can't detect and would over-count.
    """
    n_pulses = plat.shape[0]
    K = freqs.shape[0]
    R = aim_pts.shape[0]

    C_ = facets['center']
    N_ = facets['normal']
    AMP = facets['amp']

    s = xp.zeros((n_pulses, K), dtype=xp.complex128)
    t_intersect_total = 0.0
    t_phase_total = 0.0

    for p in range(n_pulses):
        o = plat[p]
        d = aim_pts - o[None, :]
        d = d / xp.linalg.norm(d, axis=1, keepdims=True)

        t0 = time.perf_counter()
        hit_pt, hit_amp, cos_inc, hit_mask, nearest_idx = ray_facet_intersect(xp, o, d, facets)
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        t_intersect_total += (t1 - t0)

        t0 = time.perf_counter()
        hit_facets = nearest_idx[hit_mask]
        visible_idx = xp.unique(hit_facets) if hit_facets.shape[0] > 0 else hit_facets

        R_ref = xp.linalg.norm(o - ref_pos)
        if visible_idx.shape[0] > 0:
            vpos = C_[visible_idx]
            vnorm = N_[visible_idx]
            vamp = AMP[visible_idx]
            look = vpos - o[None, :]
            look = look / xp.linalg.norm(look, axis=1, keepdims=True)
            cos_v = xp.abs(xp.sum(-look * vnorm, axis=1))
            R_v = xp.linalg.norm(vpos - o[None, :], axis=1)
            dR = R_v - R_ref
            amp_eff = vamp * cos_v
            phase = xp.exp(-1j * 4.0 * xp.pi * xp.outer(freqs, dR) / C)
            s[p, :] = (amp_eff[None, :] * phase).sum(axis=1)
        if on_gpu:
            xp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()
        t_phase_total += (t2 - t0)

    return s, dict(n_rays=R, t_intersect_s=t_intersect_total, t_phase_s=t_phase_total,
                    t_total_s=t_intersect_total + t_phase_total,
                    t_per_pulse_ms=(t_intersect_total + t_phase_total) / n_pulses * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--footprint', type=float, default=1000.0)
    ap.add_argument('--density', type=float, default=200.0)
    ap.add_argument('--rays', type=int, default=100, help='ray grid side for dense SBR (rays^2/pulse)')
    ap.add_argument('--pulses', type=int, default=200)
    ap.add_argument('--freq', type=int, default=64)
    ap.add_argument('--standoff', type=float, default=8000.0)
    ap.add_argument('--altitude', type=float, default=3000.0)
    ap.add_argument('--fc', type=float, default=10e9)
    ap.add_argument('--bandwidth', type=float, default=600e6)
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.gpu)
    print(f"Backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")
    print(f"Scene: {args.footprint}m x {args.footprint}m, {args.density}/km^2\n")

    facets = make_building_scene(xp, args.footprint, args.density, seed=0)
    print(f"{facets['n_buildings']} buildings, {facets['n_facets']} facets\n")

    squint_len = 400.0
    u = np.linspace(-squint_len / 2, squint_len / 2, args.pulses)
    plat = xp.asarray(np.column_stack([
        u, np.full(args.pulses, -args.standoff), np.full(args.pulses, args.altitude)
    ]))

    aim_pts, margin = make_aim_grid(xp, args.footprint, args.rays, args.standoff, args.altitude,
                                     max_height_m=40.0)
    if margin > 0.01 * args.footprint:
        print(f"aim grid padded +/-{margin:.1f}m beyond footprint for roof layover "
              f"(grid side {args.footprint + 2*margin:.0f}m)")

    freqs = xp.asarray(args.fc + np.linspace(-args.bandwidth / 2, args.bandwidth / 2, args.freq))
    ref_pos = xp.zeros(3)

    print("Running dense SBR (per-pulse ray-facet intersection)...")
    s_sbr, sbr_stats = run_dense_sbr_timed(xp, on_gpu, facets, plat, aim_pts, freqs, ref_pos)
    print(json.dumps(sbr_stats, indent=2))

    print("\nRunning ASC-cached (build once, evaluate closed-form per pulse)...")
    s_asc, asc_stats = run_asc_cached(xp, on_gpu, facets, plat, freqs, ref_pos)
    print(json.dumps(asc_stats, indent=2))

    speedup = sbr_stats['t_per_pulse_ms'] / max(asc_stats['t_per_pulse_ms'], 1e-9)
    print(f"\n=== Measured speedup (this run, this hardware) ===")
    print(f"Dense SBR:  {sbr_stats['t_per_pulse_ms']:.3f} ms/pulse")
    print(f"ASC-cached: {asc_stats['t_per_pulse_ms']:.3f} ms/pulse  "
          f"(+ {asc_stats['t_build_s']*1000:.4f} ms one-time build)")
    print(f"Speedup: {speedup:.1f}x")

    result = dict(backend='gpu' if on_gpu else 'cpu', footprint_m=args.footprint,
                  density_per_km2=args.density, n_buildings=facets['n_buildings'],
                  n_facets=facets['n_facets'], n_rays=sbr_stats['n_rays'],
                  n_pulses=args.pulses, n_freq=args.freq,
                  dense_sbr=sbr_stats, asc_cached=asc_stats, speedup=speedup)
    out_path = 'sbr_vs_asc_gpu.json' if on_gpu else 'sbr_vs_asc_cpu.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
