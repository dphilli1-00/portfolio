"""
Fix the flaw just identified: don't hold pulse count P fixed across every
footprint scale. Scope each scale's resolution requirement to what that
collect is actually FOR:

  N=1 (isolated calibration trihedral): just need to isolate the return
      from background/clutter -- no need to resolve it from anything else,
      since there's nothing else there. Coarse res is fine: ~20m.
  N=20,000 (full city): need building-level ATR-quality detail -- the
      2m res locked in earlier (TerraSAR-X-consistent), unchanged.

Intermediate scales interpolate log-linearly in (log N, log delta) between
these two grounded endpoints -- an ASSUMPTION, not a derived law, but a
reasonable Fermi-quality bridge: as the scene goes from "one target" to
"whole city," the mission goes from "characterize a point" to "image a
city," and resolution demand should climb somewhere in between. Flag this
interpolation explicitly -- it's the weakest-grounded piece of this
particular analysis.

P(N) = R * lambda / delta(N)^2   (same Nyquist pulse-count formula as before,
                                   now evaluated at each scale's own delta)
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

C = 299_792_458.0
T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0
fc = 10e9
wavelength = C / fc
R_STANDOFF = 600_000.0

DELTA_SMALL = 20.0   # m, isolated single-target collect (N=1)
DELTA_CITY = 2.0      # m, full-city ATR-quality imaging (N=20,000)
N_SMALL, N_CITY = 1, 20000

scenes = [
    ("4m trihedral", 1),
    ("Target cluster", 5),
    ("City block", 25),
    ("Small district", 125),
    ("District", 625),
    ("Large district", 3125),
    ("Full city", 20000),
]

def delta_for_N(N):
    # log-linear interpolation in (log N, log delta) between the two
    # grounded endpoints
    log_n_frac = math.log10(N) / math.log10(N_CITY)  # 0 at N=1, 1 at N=20000
    log_delta = math.log10(DELTA_SMALL) + log_n_frac * (math.log10(DELTA_CITY) - math.log10(DELTA_SMALL))
    return 10 ** log_delta

rows = []
for label, N in scenes:
    delta = delta_for_N(N)
    B_bandwidth = C / (2 * delta)
    P = R_STANDOFF * wavelength / (delta ** 2)
    ansys_per_collect = P * (T_FIXED + N * R_ANSYS)
    mine_floor = P * (T_FIXED + N * R_ASC)
    build_cost = N * B_WORST
    n_star = build_cost / (ansys_per_collect - mine_floor)
    rows.append({
        "scene": label, "n_structures": N, "resolution_m": delta,
        "bandwidth_MHz": B_bandwidth / 1e6, "pulses_per_collect": P,
        "ansys_per_collect_s": ansys_per_collect,
        "mine_runtime_floor_per_collect_s": mine_floor,
        "floor_speedup": ansys_per_collect / mine_floor,
        "crossover_collects_n_star": n_star,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/footprint_sweep_realistic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'scene':16s} {'N':>7s} {'res(m)':>7s} {'BW(MHz)':>8s} {'P(pulses)':>10s} "
      f"{'ansys/coll':>11s} {'floor':>9s} {'speedup':>8s} {'n*':>8s}")
for r in rows:
    print(f"{r['scene']:16s} {r['n_structures']:7d} {r['resolution_m']:6.2f}m "
          f"{r['bandwidth_MHz']:7.1f} {r['pulses_per_collect']:10.0f} "
          f"{r['ansys_per_collect_s']:10.2f}s {r['mine_runtime_floor_per_collect_s']:8.3f}s "
          f"{r['floor_speedup']:7.1f}x {r['crossover_collects_n_star']:7.1f}")

# --- multi-curve overlay, same style as before, now with N-specific P ---
n_plot = np.array([1, 1.5, 2, 3, 5, 6, 8, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000])
colors = cm.viridis(np.linspace(0.05, 0.9, len(scenes)))

fig, ax = plt.subplots(figsize=(11, 8))
for r, color in zip(rows, colors):
    ansys_pc, floor, build = r["ansys_per_collect_s"], r["mine_runtime_floor_per_collect_s"], r["n_structures"] * B_WORST
    mine_curve = build / n_plot + floor
    ax.plot(n_plot, [ansys_pc] * len(n_plot), color=color, linewidth=2, linestyle='-',
            label=f"{r['scene']} (N={r['n_structures']:,}, {r['resolution_m']:.1f}m res) -- Ansys")
    ax.plot(n_plot, mine_curve, color=color, linewidth=2, linestyle='--', marker='o', markersize=3,
            label=f"{r['scene']} -- mine")

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of collects run against the same built library (log)')
ax.set_ylabel('Amortized compute time PER COLLECT (sec, log)')
ax.set_title('Amortized cost per collect, resolution scoped to mission per scale\n'
             '(coarse res + few pulses for isolated targets, fine res + many pulses for the city)')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.grid(True, which='both', alpha=0.25)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/footprint_sweep_realistic_overlay.png", dpi=150)

# --- two-panel: cost/collect vs N, and n* vs N (now NOT flat) ---
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                  gridspec_kw={'height_ratios': [2.2, 1]})
N_vals = [r["n_structures"] for r in rows]
ax1.plot(N_vals, [r["ansys_per_collect_s"] for r in rows], marker='o', color='#B85C00',
         linewidth=2, label='Ansys dense rays (per collect)')
ax1.plot(N_vals, [r["mine_runtime_floor_per_collect_s"] for r in rows], marker='o', color='#0B7285',
         linewidth=2, label='My approach -- runtime floor')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_ylabel('Cost per collect (sec, log)')
ax1.set_title('Footprint sweep with mission-scoped resolution (not held fixed)')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', alpha=0.3)

ax2.plot(N_vals, [r["crossover_collects_n_star"] for r in rows], marker='o', color='#5B6672', linewidth=2)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Footprint scale (# structures, N)')
ax2.set_ylabel('Break-even\ncollects (n*, log)')
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()
fig2.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/footprint_sweep_realistic_nstar.png", dpi=150)
print("\nSaved footprint_sweep_realistic.csv, _overlay.png, _nstar.png")
