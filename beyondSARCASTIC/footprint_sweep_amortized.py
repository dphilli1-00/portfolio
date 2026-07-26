"""
Same footprint sweep as before (single 4m trihedral -> full city, N=1 to
20,000), now using the AMORTIZED per-collect metric instead of a single
n=1 collect -- i.e. combining the footprint-scale sweep with the
runtime-floor / crossover-collects framing from the last plot.

Locked-in real params: 2m res, X-band, satellite @ 600km -> P=4,497
pulses/collect. Worst-case ASC build: B=2 sec/structure.

Two things per N:
  1. Ansys per-collect cost:            P*(t_fixed + N*r_ansys)
  2. My runtime FLOOR per-collect cost: P*(t_fixed + N*r_asc)   (n->infinity,
     i.e. the build tax fully amortized away -- this is the real "ASC is
     faster" number)

Plus the crossover collect count n*(N) = N*B / (ansys_per_collect(N) -
mine_floor(N)) at each scale -- included to check whether it actually
depends on footprint scale or not.
"""

import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0
P = 4497  # pulses/collect, locked-in real params (2m res, X-band, sat @ 600km)

scenes = [
    ("N=1 (4m trihedral)", 1),
    ("N=5 (target cluster)", 5),
    ("N=25 (city block)", 25),
    ("N=125 (small district)", 125),
    ("N=625 (district)", 625),
    ("N=3,125 (large district)", 3125),
    ("N=20,000 (full city)", 20000),
]

rows = []
for label, N in scenes:
    ansys_per_collect = P * (T_FIXED + N * R_ANSYS)
    mine_floor = P * (T_FIXED + N * R_ASC)
    n_star = (N * B_WORST) / (ansys_per_collect - mine_floor)
    rows.append({
        "scene": label,
        "n_structures": N,
        "ansys_per_collect_s": ansys_per_collect,
        "mine_runtime_floor_per_collect_s": mine_floor,
        "floor_speedup": ansys_per_collect / mine_floor,
        "crossover_collects_n_star": n_star,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/footprint_sweep_amortized.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'scene':22s} {'ansys/collect':>14s} {'mine floor':>12s} {'speedup':>9s} {'n* collects':>12s}")
for r in rows:
    print(f"{r['scene']:22s} {r['ansys_per_collect_s']:13.2f}s {r['mine_runtime_floor_per_collect_s']:11.3f}s "
          f"{r['floor_speedup']:8.1f}x {r['crossover_collects_n_star']:11.2f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [2.2, 1]})

N_vals = [r["n_structures"] for r in rows]
ax1.plot(N_vals, [r["ansys_per_collect_s"] for r in rows], marker='o', color='#B85C00',
         linewidth=2, label='Ansys dense rays (per collect, flat cost basis)')
ax1.plot(N_vals, [r["mine_runtime_floor_per_collect_s"] for r in rows], marker='o', color='#0B7285',
         linewidth=2, label='My approach -- runtime floor (build fully amortized)')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_ylabel('Cost per collect (sec, log)')
ax1.set_title('Footprint sweep: single trihedral to full city\n(amortized / steady-state comparison, real params: 2m res, X-band, sat@600km)')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', alpha=0.3)

ax2.plot(N_vals, [r["crossover_collects_n_star"] for r in rows], marker='o', color='#5B6672', linewidth=2)
ax2.set_xscale('log')
ax2.set_ylim(0, 8)
ax2.set_xlabel('Footprint scale (# structures, N)')
ax2.set_ylabel('Break-even\ncollects (n*)')
ax2.grid(True, which='both', alpha=0.3)
ax2.axhline(rows[0]["crossover_collects_n_star"], color='#B85C00', linestyle=':', linewidth=1,
            label=f'n* = {rows[0]["crossover_collects_n_star"]:.2f} (constant across all scales)')
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/footprint_sweep_amortized.png", dpi=150)
print("\nSaved footprint_sweep_amortized.csv/.png")
