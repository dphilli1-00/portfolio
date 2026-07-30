"""
Two follow-on analyses to scaling_data.py:

1. DECOMPOSITION -- of the total per-pulse speedup (Ansys -> mine), how much
   comes from each of the two independent levers:
     lever A: fidelity tiering ("don't ray-trace grass")   Ansys -> tiered
     lever B: ASC caching (closed-form eval, no re-trace)  tiered -> mine
   These multiply: (Ansys/tiered) x (tiered/mine) = Ansys/mine.

2. BREAK-EVEN -- ASC isn't free. Building the library costs B sec/structure,
   PAID ONCE, before the cheap per-pulse eval exists at all (ray-trace the
   structure, classify against canonical types, do the target decomposition,
   assign closed-form parameters -- see the "City-scale build cost" worked
   example). Tiered dense SBR has no such upfront cost -- it just re-traces
   every pulse, forever, at a higher per-pulse price.

   So the real comparison isn't per-pulse cost, it's TOTAL cost as a function
   of how many pulses you ultimately run against that library:

     C_tiered(N, P) = P * (t_fixed + N * r_tiered)
     C_asc(N, P)    = N * B  +  P * (t_fixed + N * r_asc)

   Solve C_tiered = C_asc for the break-even pulse count P*:

     N*B = P* * N * (r_tiered - r_asc)
     P* = B / (r_tiered - r_asc)

   Notice N cancels -- build cost and per-pulse savings both scale with N,
   so the break-even pulse count is the SAME regardless of footprint scale.
   What matters is how many pulses/looks you ultimately spend against a
   built library, not how big the scene is.

Build-cost assumption B: blends the "City-scale build cost" worked example's
per-structure stages (ray trace ~1s + classify ~0.5s + closed-form assign
~free, with an ML-fallback tail at ~1.8s for irregular geometry) into a
single round number, ~2 sec/structure. At N=20,000 that's ~11 hours on one
workstation -- consistent with the "hours, not days" claim made there.
"""

import json
import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_ANSYS = 75e-6
R_TIERED = 50e-6
R_ASC = 0.5e-6
B_BUILD = 2.0          # sec/structure, one-time ASC library build cost
N_PULSES_COLLECT = 500

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/scaling_data.json") as f:
    base = json.load(f)
rows = base["rows"]

# ---------------------------------------------------------------------
# 1. Decomposition: where does the speedup come from, at each scale?
# ---------------------------------------------------------------------
decomp_rows = []
for r in rows:
    lever_a = r["speedup_ansys_vs_tiered"]       # tiering only
    lever_b = r["speedup_tiered_vs_mypipeline"]  # caching only
    total = lever_a * lever_b
    decomp_rows.append({
        "scene": r["scene"],
        "n_structures": r["n_structures"],
        "lever_a_tiering": lever_a,
        "lever_b_caching": lever_b,
        "total_speedup": total,
        "caching_share_of_log_speedup": (
            0.0 if total <= 1 else
            (lever_b - 1) / ((lever_a - 1) + (lever_b - 1)) if (lever_a - 1) + (lever_b - 1) > 0 else 0.0
        ),
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/speedup_decomposition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(decomp_rows[0].keys()))
    w.writeheader()
    w.writerows(decomp_rows)

print("DECOMPOSITION -- where the speedup comes from")
print(f"{'scene':42s} {'tiering(A)':>11s} {'caching(B)':>11s} {'total A*B':>10s}")
for r in decomp_rows:
    print(f"{r['scene']:42s} {r['lever_a_tiering']:10.2f}x {r['lever_b_caching']:10.1f}x {r['total_speedup']:9.1f}x")

# bar chart: grouped bars per scene, lever A vs lever B
fig, ax = plt.subplots(figsize=(9, 6))
labels = [f"N={r['n_structures']:,}" for r in decomp_rows]
x = range(len(labels))
w = 0.38
ax.bar([i - w/2 for i in x], [r["lever_a_tiering"] for r in decomp_rows], width=w,
       color="#5B6672", label="Lever A: fidelity tiering (skip terrain)")
ax.bar([i + w/2 for i in x], [r["lever_b_caching"] for r in decomp_rows], width=w,
       color="#0B7285", label="Lever B: ASC caching (no per-pulse re-trace)")
ax.set_yscale("log")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Speedup factor (x)")
ax.set_title("Where the speedup comes from: tiering vs caching, by footprint scale")
ax.legend()
ax.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/speedup_decomposition.png", dpi=150)

# ---------------------------------------------------------------------
# 2. Break-even: pulses needed before the ASC build cost pays for itself
# ---------------------------------------------------------------------
P_star = B_BUILD / (R_TIERED - R_ASC)
print(f"\nBREAK-EVEN pulse count (independent of N): {P_star:,.0f} pulses")
print(f"  = {P_star / N_PULSES_COLLECT:,.1f} collects of {N_PULSES_COLLECT} pulses each")

# verify numerically at two different N to confirm independence, and
# tabulate total cost vs pulse count for the city-scale case specifically
breakeven_rows = []
pulse_counts = [1, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000]
N_CITY = 20000
for P in pulse_counts:
    c_tiered = P * (T_FIXED + N_CITY * R_TIERED)
    c_asc = N_CITY * B_BUILD + P * (T_FIXED + N_CITY * R_ASC)
    breakeven_rows.append({
        "n_pulses": P,
        "n_collects_of_500": P / N_PULSES_COLLECT,
        "tiered_total_cost_s": c_tiered,
        "asc_total_cost_s": c_asc,
        "asc_cheaper": c_asc < c_tiered,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/breakeven_data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(breakeven_rows[0].keys()))
    w.writeheader()
    w.writerows(breakeven_rows)

print(f"\nBREAK-EVEN table (city scale, N={N_CITY:,}):")
print(f"{'pulses':>8s} {'collects':>9s} {'tiered_s':>12s} {'asc_s':>12s} {'asc cheaper?':>13s}")
for r in breakeven_rows:
    print(f"{r['n_pulses']:8d} {r['n_collects_of_500']:9.1f} "
          f"{r['tiered_total_cost_s']:12.1f} {r['asc_total_cost_s']:12.1f} "
          f"{'YES' if r['asc_cheaper'] else 'no':>13s}")

# sanity check: confirm P* is ~independent of N by checking N=3125 too
N_check = 3125
c_tiered_check = P_star * (T_FIXED + N_check * R_TIERED)
c_asc_check = N_check * B_BUILD + P_star * (T_FIXED + N_check * R_ASC)
print(f"\nSanity check at N={N_check}: tiered={c_tiered_check:.1f}s, asc={c_asc_check:.1f}s "
      f"(should be ~equal at P*={P_star:.0f})")

# plot: total cost vs pulses, log-log, city scale
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot([r["n_pulses"] for r in breakeven_rows],
         [r["tiered_total_cost_s"] for r in breakeven_rows],
         marker='o', color='#5B6672', linewidth=2, label='Fidelity-tiered dense SBR (no upfront cost)')
ax2.plot([r["n_pulses"] for r in breakeven_rows],
         [r["asc_total_cost_s"] for r in breakeven_rows],
         marker='o', color='#0B7285', linewidth=2, label='ASC-cached (pays N*B build cost upfront)')
ax2.axvline(P_star, color='#B85C00', linestyle='--', linewidth=1.5,
            label=f'Break-even: {P_star:,.0f} pulses (~{P_star/500:.0f} collects)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Total pulses run against the library (log)')
ax2.set_ylabel('Total wall-clock compute time (sec, log)')
ax2.set_title(f'Break-even: when does building the ASC library pay off?\n(city scale, N={N_CITY:,} structures)')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()
fig2.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/breakeven_plot.png", dpi=150)
print("\nSaved: speedup_decomposition.csv/.png, breakeven_data.csv, breakeven_plot.png")
