"""
Locked-in scenario: 2m resolution, X-band (fc=10GHz), satellite spotlight
platform, R=600km slant range, full 10km x 10km city (N=20,000 structures).

Real pulse count per collect (from radar_params.py):
    P = R * lambda / delta_az^2 = 600,000 * 0.029979 / 4 = 4,497 pulses

This replaces the earlier round-number "500 pulses" placeholder with the
actual Nyquist-sampled pulse count this system would fly.

Sweep footprint scale from a single 4m trihedral (N=1, the canonical
calibration target) up to the full city (N=20,000), and compute TOTAL
wall-clock time for one real collect at each scale:

    Ansys dense rays:  P * (t_fixed + N * r_ansys)      -- no caching, ever
    My approach:       N * B_worst + P * (t_fixed + N * r_asc)  -- pays the
                        worst-case build cost (2 sec/structure) up front,
                        then the cheap per-pulse ASC eval

Per-structure rates (r_ansys, r_asc, t_fixed) are unchanged from the
earlier ray-tracing/FLOP-count derivations -- they're about geometric
compute cost, not waveform bandwidth, so they don't need to be re-derived
for the 2m-resolution case.
"""

import csv
import matplotlib.pyplot as plt

C = 299_792_458.0
T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0

fc = 10e9
wavelength = C / fc
delta = 2.0          # m, locked-in resolution
R_STANDOFF = 600_000.0  # m, satellite slant range
P = round(R_STANDOFF * wavelength / (delta ** 2))

print(f"Locked-in collect: delta={delta}m, R={R_STANDOFF/1000:.0f}km, P={P:,} pulses/collect")

scenes = [
    ("N=1 (4m trihedral)", 1),
    ("N=5 (target cluster)", 5),
    ("N=25 (city block)", 25),
    ("N=125 (small district)", 125),
    ("N=625 (district)", 625),
    ("N=3,125 (large district)", 3125),
    ("N=20,000 (full 10km x 10km city)", 20000),
]

def fmt_time(s):
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s/60:.2f}min"
    return f"{s/3600:.2f}hr"

rows = []
for label, N in scenes:
    ansys_total = P * (T_FIXED + N * R_ANSYS)
    mine_total = N * B_WORST + P * (T_FIXED + N * R_ASC)
    rows.append({
        "scene": label,
        "n_structures": N,
        "ansys_total_s": ansys_total,
        "mine_total_s": mine_total,
        "ansys_human": fmt_time(ansys_total),
        "mine_human": fmt_time(mine_total),
        "ratio_mine_over_ansys": mine_total / ansys_total,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/final_scaling_real_params.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'scene':34s} {'ansys':>10s} {'mine':>10s} {'mine/ansys':>10s}")
for r in rows:
    print(f"{r['scene']:34s} {r['ansys_human']:>10s} {r['mine_human']:>10s} {r['ratio_mine_over_ansys']:9.1f}x")

fig, ax = plt.subplots(figsize=(10, 6.5))
N_vals = [r["n_structures"] for r in rows]
ax.plot(N_vals, [r["ansys_total_s"] for r in rows], marker='o', color='#B85C00',
        linewidth=2, label='Ansys dense rays (SBR+UTD, no caching)')
ax.plot(N_vals, [r["mine_total_s"] for r in rows], marker='o', color='#0B7285',
        linewidth=2, label='My approach (worst-case build, B=2s/structure)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Footprint scale (# structures, N) -- single 4m trihedral to full city')
ax.set_ylabel('Total time for ONE real collect (sec, log)')
ax.set_title(f'Single collect, real system params: 2m res, X-band, satellite @ {R_STANDOFF/1000:.0f}km\n'
             f'({P:,} pulses/collect)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/final_scaling_real_params.png", dpi=150)
print("\nSaved final_scaling_real_params.csv/.png")
