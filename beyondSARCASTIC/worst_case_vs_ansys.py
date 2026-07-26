"""
Stress-test comparison: WORST CASE for my pipeline vs Ansys's actual
baseline (dense SBR + UTD, full ray tracing every pulse) -- not my own
tiered mode. Two pessimistic-for-me choices stacked together:

  1. ASC build cost uses the pessimistic B = 2 sec/structure (the blended
     ray-trace + classify + assign + ML-fallback-tail estimate), not the
     optimistic 0.5 sec/structure classification-dominated number.
  2. The opponent is Ansys dense rays (r_ansys = 75 us/structure, includes
     UTD overhead), not my own fidelity-tiered dense SBR (r_tiered = 50
     us/structure, which is a friendlier, lower baseline since it already
     borrows my own "skip terrain" trick).

This is the most conservative version of the argument: if it still holds up
here, it holds up everywhere in between.
"""

import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_ANSYS = 75e-6      # dense SBR + UTD, full ray trace, every pulse
R_ASC = 0.5e-6        # ASC-cached closed-form eval
B_WORST = 2.0          # sec/structure, pessimistic build cost
P_COLLECT = 500
N_CITY = 20000

# --- break-even vs pulses, city scale ---
P_star = B_WORST / (R_ANSYS - R_ASC)
print(f"Break-even (worst-case B={B_WORST}s/structure, vs Ansys dense rays r={R_ANSYS*1e6:.0f}us/structure):")
print(f"  P* = {P_star:,.0f} pulses = {P_star/P_COLLECT:,.1f} collects of {P_COLLECT} pulses")

pulse_counts = [1, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 26846, 40000, 60000, 80000, 100000]
rows_pulses = []
for P in pulse_counts:
    c_ansys = P * (T_FIXED + N_CITY * R_ANSYS)
    c_asc = N_CITY * B_WORST + P * (T_FIXED + N_CITY * R_ASC)
    rows_pulses.append({"n_pulses": P, "n_collects": P / P_COLLECT,
                         "ansys_dense_total_s": c_ansys, "asc_worstcase_total_s": c_asc,
                         "asc_cheaper": c_asc < c_ansys})

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/worst_case_breakeven.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_pulses[0].keys()))
    w.writeheader()
    w.writerows(rows_pulses)

print(f"\n{'pulses':>8s} {'collects':>9s} {'ansys_s':>12s} {'asc_s':>12s} {'asc cheaper?':>13s}")
for r in rows_pulses:
    print(f"{r['n_pulses']:8d} {r['n_collects']:9.1f} {r['ansys_dense_total_s']:12.1f} "
          f"{r['asc_worstcase_total_s']:12.1f} {'YES' if r['asc_cheaper'] else 'no':>13s}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([r["n_pulses"] for r in rows_pulses], [r["ansys_dense_total_s"] for r in rows_pulses],
        marker='o', color='#B85C00', linewidth=2, label='Ansys dense rays (SBR+UTD, no caching)')
ax.plot([r["n_pulses"] for r in rows_pulses], [r["asc_worstcase_total_s"] for r in rows_pulses],
        marker='o', color='#0B7285', linewidth=2, label='My pipeline, WORST CASE (B=2s/structure build)')
ax.axvline(P_star, color='#5B6672', linestyle='--', linewidth=1.5,
           label=f'Break-even: {P_star:,.0f} pulses (~{P_star/500:.0f} collects)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Total pulses run against the library (log)')
ax.set_ylabel('Total wall-clock compute time (sec, log)')
ax.set_title(f'Worst case vs Ansys dense rays (city scale, N={N_CITY:,})')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/worst_case_breakeven.png", dpi=150)

# --- cost vs footprint scale, single 500-pulse collect ---
scenes = [("N=1", 1), ("N=5", 5), ("N=25", 25), ("N=125", 125),
          ("N=625", 625), ("N=3,125", 3125), ("N=20,000", 20000)]
rows_scale = []
for label, N in scenes:
    build = N * B_WORST
    runtime = P_COLLECT * (T_FIXED + N * R_ASC)
    asc_total = build + runtime
    ansys_total = P_COLLECT * (T_FIXED + N * R_ANSYS)
    rows_scale.append({"scene": label, "n_structures": N,
                        "asc_worstcase_total_s": asc_total,
                        "ansys_dense_total_s": ansys_total,
                        "ratio_asc_over_ansys": asc_total / ansys_total})

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/worst_case_by_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_scale[0].keys()))
    w.writeheader()
    w.writerows(rows_scale)

print(f"\nSingle {P_COLLECT}-pulse collect, worst case ASC vs Ansys dense rays, by scale:")
print(f"{'scene':10s} {'asc_worst':>11s} {'ansys':>10s} {'asc/ansys':>10s}")
for r in rows_scale:
    print(f"{r['scene']:10s} {r['asc_worstcase_total_s']:10.2f}s {r['ansys_dense_total_s']:9.2f}s "
          f"{r['ratio_asc_over_ansys']:9.1f}x")

fig2, ax2 = plt.subplots(figsize=(9, 6))
labels = [r["scene"] for r in rows_scale]
x = range(len(labels))
w = 0.3
ax2.bar([i - w/2 for i in x], [r["asc_worstcase_total_s"] for r in rows_scale], width=w,
        color="#0B7285", label="My pipeline, WORST CASE (B=2s/structure)")
ax2.bar([i + w/2 for i in x], [r["ansys_dense_total_s"] for r in rows_scale], width=w,
        color="#B85C00", label="Ansys dense rays (SBR+UTD, no caching)")
ax2.set_yscale("log")
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=20, ha="right")
ax2.set_ylabel("Compute time, single 500-pulse collect (sec, log)")
ax2.set_title("Worst case vs Ansys dense rays, by footprint scale")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/worst_case_by_scale.png", dpi=150)
print("\nSaved worst_case_breakeven.csv/.png, worst_case_by_scale.csv/.png")
