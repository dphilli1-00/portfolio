"""
Total runtime normalized by number of experiments (collects) -- the plot
that actually isolates "is ASC worth it" from "is ASC fast."

    Ansys amortized cost/collect = P*(t_fixed + N*r_ansys)         -- FLAT.
        No build step, so cost per collect never changes with reuse.

    Mine amortized cost/collect  = N*B_worst/n + P*(t_fixed + N*r_asc)
        n = number of collects run against the SAME built library.
        As n -> 1, dominated by the build tax (N*B_worst/n is huge).
        As n -> infinity, the build tax -> 0 and this approaches the pure
        runtime floor: P*(t_fixed + N*r_asc) -- the ~125x-faster number
        that's been true in every per-pulse comparison all along.

Locked-in scenario: 2m res, X-band, satellite @ 600km, full city (N=20,000),
P=4,497 pulses/collect, worst-case build B=2 sec/structure.
"""

import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0
N = 20000
P = 4497  # pulses/collect, locked-in real params

ansys_per_collect = P * (T_FIXED + N * R_ANSYS)      # constant
mine_runtime_floor = P * (T_FIXED + N * R_ASC)         # asymptote as n->inf
build_cost = N * B_WORST

n_star = build_cost / (ansys_per_collect - mine_runtime_floor)
print(f"Ansys: {ansys_per_collect:.2f} sec/collect, FLAT, forever")
print(f"Mine runtime floor (n->inf): {mine_runtime_floor:.2f} sec/collect "
      f"({ansys_per_collect/mine_runtime_floor:.1f}x faster than Ansys)")
print(f"Build tax: {build_cost:,.0f} sec, one-time")
print(f"Crossover: n* = {n_star:.2f} collects")

n_values = [1, 2, 3, 5, n_star, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
rows = []
for n in n_values:
    mine_amortized = build_cost / n + mine_runtime_floor
    rows.append({
        "n_collects": n,
        "ansys_per_collect_s": ansys_per_collect,
        "mine_amortized_per_collect_s": mine_amortized,
        "mine_cheaper": mine_amortized < ansys_per_collect,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/amortized_per_collect.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'n_collects':>12s} {'ansys/collect':>14s} {'mine/collect':>14s} {'mine cheaper?':>13s}")
for r in rows:
    print(f"{r['n_collects']:12.2f} {r['ansys_per_collect_s']:14.2f} "
          f"{r['mine_amortized_per_collect_s']:14.2f} {'YES' if r['mine_cheaper'] else 'no':>13s}")

fig, ax = plt.subplots(figsize=(9, 6.5))
n_plot = [1, 2, 3, 5, 6, 8, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
mine_plot = [build_cost / n + mine_runtime_floor for n in n_plot]
ax.plot(n_plot, [ansys_per_collect] * len(n_plot), color='#B85C00', linewidth=2,
        label=f'Ansys dense rays: {ansys_per_collect:,.0f} sec/collect (flat, no caching)')
ax.plot(n_plot, mine_plot, color='#0B7285', linewidth=2, marker='o', markersize=4,
        label='My approach: amortized cost/collect (build tax / n + runtime)')
ax.axhline(mine_runtime_floor, color='#0B7285', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Runtime floor (n->inf): {mine_runtime_floor:.1f} sec/collect '
                 f'({ansys_per_collect/mine_runtime_floor:.0f}x faster)')
ax.axvline(n_star, color='#5B6672', linestyle='--', linewidth=1.5,
           label=f'Crossover: n*={n_star:.1f} collects')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of collects run against the same built library (log)')
ax.set_ylabel('Amortized compute time PER COLLECT (sec, log)')
ax.set_title('Total runtime normalized by number of experiments\n(city scale, real params: 2m res, X-band, satellite @ 600km)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/amortized_per_collect.png", dpi=150)
print("\nSaved amortized_per_collect.csv/.png")
