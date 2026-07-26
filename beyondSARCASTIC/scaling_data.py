"""
Fermi-estimate compute-scaling data: footprint scale (N canonical structures)
vs per-pulse phase-history compute time, for three approaches:

  - "Ansys"          : dense SBR + UTD, full ray trace every pulse
  - "Fidelity-tiered" : my pipeline's own dense-SBR mode, no ASC caching yet --
                    the "don't ray-bound on grass" tier: ray-trace structures
                    only, terrain handled by the empirical clutter model
                    instead of being ray-traced. Same brute-force per-pulse
                    re-trace as Ansys for the structures that remain, just
                    without the UTD edge-diffraction overhead and without
                    wasting rays on terrain that was never going to matter.
                    This isolates the "stop ray-tracing grass" lever from the
                    "cache the structures" lever -- they're independent wins.
  - "My pipeline"    : ASC-cached -- ray-trace once per structure (offline
                    build, NOT shown here), then per-pulse cost is just
                    evaluating the closed-form ASC/GTD sum

All three curves share the same functional form:

    t(N) = t_fixed + N * r

  t_fixed : per-pulse overhead common to both -- platform/aperture geometry
            update, motion-comp reference range, output buffer write.
            NOT a function of scene content, so it's identical for both
            curves. This is what makes them converge at small N: at N=1,
            neither approach's per-structure term matters yet.

  r       : incremental cost per structure, per pulse. This is the number
            that differs by ~150x between the two approaches, and is the
            entire reason the curves diverge at scale.

ASSUMPTIONS (Fermi-estimate quality -- stated explicitly, not measured):

  RT-core throughput ......... 10 Gigarays/sec (conservative, Turing-class;
                                Ampere/Ada are faster -- see Willis & Hossain,
                                arXiv:2005.09736, for RT-core SAR acceleration)
  r_ansys (dense SBR+UTD) .... 75 microsec/structure
      Derived: reuses this deck's own "fidelity-tiered dense SBR, structures
      only" anchor point (~1 sec/pulse @ N=20,000 structures -> 50 us/structure),
      then applies a +50% overhead multiplier for UTD edge-diffraction
      processing that plain PO-only SBR doesn't do. The 1.5x factor is an
      ASSUMPTION, not a measured Ansys number -- flagged as uncertain.
  r_tiered (fidelity-tiered, no ASC) .... 50 microsec/structure
      This IS the anchor point above, used directly, no UTD multiplier --
      structures-only dense SBR, terrain excluded from the ray budget. Same
      per-structure ray-tracing cost model as Ansys minus the UTD overhead
      assumption; isolates what "don't ray-trace grass" buys on its own,
      before ASC caching enters the picture at all.
  r_asc (ASC-cached eval) .... 0.5 microsec/structure
      Derived bottom-up, independent of the above: ~5 scattering centers per
      structure (sparse ASC representation) x ~100 FLOPs/center (closed-form
      GTD amplitude/phase/persistence-window terms + Sinclair-matrix 2x2
      polarimetric combine) = ~500 FLOPs/structure, at a deliberately
      conservative 1 GFLOP/s single-core scalar rate (no SIMD/GPU batching --
      real implementation would likely beat this).
  t_fixed ..................... 2 ms/pulse, both approaches (assumption --
      not independently grounded, order-of-magnitude guess for aperture/
      doppler bookkeeping + I/O).
  Terrain/diffuse clutter: excluded from both curves -- handled by the
      empirical clutter model regardless of structure representation choice,
      so it doesn't differentiate the two approaches. Consistent with the
      "don't ray-trace grass" position taken elsewhere in this deck.
  One-time ASC library BUILD cost is NOT included in this per-pulse curve --
      that's a separate, already-estimated fixed cost (see the "City-scale
      build cost" worked example) that amortizes across the whole collect.
      This plot is steady-state per-pulse cost only, i.e. it assumes the ASC
      library already exists for whatever's in the scene.

GOTCHA for reading the resulting plot: the "growing benefit" you'll see is
NOT because ASC scales sub-linearly with N -- both curves are linear in N.
It's because (a) a shared fixed floor keeps both curves pinned together at
small N, and (b) the per-structure slopes differ by ~150x, so the dense-SBR
line escapes that floor almost immediately while the ASC line stays near it
for far longer. Don't caption this as "ASC scales better" -- it scales the
same way, just ~150x flatter.
"""

import csv
import json

RT_GIGARAYS = 10.0          # documentation only, not used directly below
T_FIXED = 2e-3               # sec/pulse, shared
R_ANSYS = 75e-6              # sec/structure (dense SBR + UTD)
R_TIERED = 50e-6             # sec/structure (fidelity-tiered dense SBR, no ASC, no UTD)
R_ASC = 0.5e-6                # sec/structure (ASC-cached closed-form eval)
N_PULSES_COLLECT = 500        # matches earlier worked-example collect length

scenes = [
    ("Single trihedral (calibration target)", 1),
    ("Single vehicle / small target cluster", 5),
    ("City block (~25 structures)", 25),
    ("Small district (~125 structures)", 125),
    ("District (~625 structures)", 625),
    ("Large district (~3,125 structures)", 3125),
    ("Full 10km x 10km city (~20,000 structures)", 20000),
]

rows = []
for label, N in scenes:
    t_ansys = T_FIXED + N * R_ANSYS
    t_tiered = T_FIXED + N * R_TIERED
    t_asc = T_FIXED + N * R_ASC
    rows.append({
        "scene": label,
        "n_structures": N,
        "ansys_per_pulse_s": t_ansys,
        "tiered_per_pulse_s": t_tiered,
        "mypipeline_per_pulse_s": t_asc,
        "speedup_ansys_vs_mypipeline": t_ansys / t_asc,
        "speedup_ansys_vs_tiered": t_ansys / t_tiered,
        "speedup_tiered_vs_mypipeline": t_tiered / t_asc,
        "ansys_per_500pulse_s": t_ansys * N_PULSES_COLLECT,
        "tiered_per_500pulse_s": t_tiered * N_PULSES_COLLECT,
        "mypipeline_per_500pulse_s": t_asc * N_PULSES_COLLECT,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/scaling_data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/scaling_data.json", "w") as f:
    json.dump({
        "assumptions": {
            "rt_gigarays_per_sec": RT_GIGARAYS,
            "t_fixed_s_per_pulse": T_FIXED,
            "r_ansys_s_per_structure": R_ANSYS,
            "r_tiered_s_per_structure": R_TIERED,
            "r_asc_s_per_structure": R_ASC,
            "n_pulses_per_collect": N_PULSES_COLLECT,
        },
        "rows": rows,
    }, f, indent=2)

print(f"{'scene':42s} {'N':>7s} {'ansys':>10s} {'tiered':>10s} {'mine':>10s} {'ansys/tier':>11s} {'tier/mine':>10s}")
for r in rows:
    print(f"{r['scene']:42s} {r['n_structures']:7d} "
          f"{r['ansys_per_pulse_s']*1000:8.3f}ms {r['tiered_per_pulse_s']*1000:8.3f}ms "
          f"{r['mypipeline_per_pulse_s']*1000:8.4f}ms "
          f"{r['speedup_ansys_vs_tiered']:9.1f}x {r['speedup_tiered_vs_mypipeline']:8.1f}x")
