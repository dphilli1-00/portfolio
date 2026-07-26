"""
Cost contributions vs footprint scale (N structures) -- same x-axis as
scaling_plot.py / speedup_decomposition.py, instead of the pulse-count axis
that made the ASC break-even plot confusing (ASC's line was flat there
because the build cost dominated across the ENTIRE plotted pulse range --
that's a real result, but it hid what's actually driving cost as the scene
gets bigger).

For a single standard collect (P = 500 pulses, matching the rest of this
deck's worked examples), break total cost into its real components and plot
them against N:

  ASC build cost      = N * B                       (one-time, no P dependence)
  ASC runtime cost     = P * (t_fixed + N * r_asc)    (grows with N, but slowly)
  ASC total            = build + runtime
  Tiered total          = P * (t_fixed + N * r_tiered) (no build cost at all)

This should make it obvious why ASC "looked flat" before: build cost is
O(N) with a MUCH steeper slope (2 sec/structure) than either runtime term
(tens of microsec/structure) -- so for any single collect, build cost isn't
just large, it's the thing actually growing as the scene scales up. Tiered's
cost also grows with N, just ~80x slower per structure than the ASC build
step. That's the real reason a single collect never favors ASC: the thing
you're trading away (avoid a slow per-pulse re-trace) is being paid for with
something that scales with N even MORE steeply (the one-time build).
"""

import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_ANSYS = 75e-6
R_TIERED = 50e-6
R_ASC = 0.5e-6
B_BUILD = 2.0
P_COLLECT = 500  # one standard collect, matches rest of deck

scenes = [
    ("N=1", 1),
    ("N=5", 5),
    ("N=25", 25),
    ("N=125", 125),
    ("N=625", 625),
    ("N=3,125", 3125),
    ("N=20,000", 20000),
]

rows = []
for label, N in scenes:
    build = N * B_BUILD
    asc_runtime = P_COLLECT * (T_FIXED + N * R_ASC)
    asc_total = build + asc_runtime
    tiered_total = P_COLLECT * (T_FIXED + N * R_TIERED)
    rows.append({
        "scene": label,
        "n_structures": N,
        "asc_build_cost_s": build,
        "asc_runtime_cost_s": asc_runtime,
        "asc_total_cost_s": asc_total,
        "tiered_total_cost_s": tiered_total,
        "build_share_of_asc_total": build / asc_total,
    })

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/cost_contributions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Cost contributions for a single {P_COLLECT}-pulse collect")
print(f"{'scene':10s} {'build':>10s} {'asc_runtime':>12s} {'asc_total':>10s} {'tiered':>10s} {'build_share':>11s}")
for r in rows:
    print(f"{r['scene']:10s} {r['asc_build_cost_s']:9.2f}s {r['asc_runtime_cost_s']:11.3f}s "
          f"{r['asc_total_cost_s']:9.2f}s {r['tiered_total_cost_s']:9.2f}s "
          f"{r['build_share_of_asc_total']*100:10.1f}%")

# grouped bar chart, x = N (footprint scale), y = cost components (log)
fig, ax = plt.subplots(figsize=(10, 6))
labels = [r["scene"] for r in rows]
x = range(len(labels))
w = 0.2
ax.bar([i - 1.5*w for i in x], [r["asc_build_cost_s"] for r in rows], width=w,
       color="#B85C00", label="ASC build cost (one-time)")
ax.bar([i - 0.5*w for i in x], [r["asc_runtime_cost_s"] for r in rows], width=w,
       color="#FF9E2C", label="ASC runtime cost (500-pulse collect)")
ax.bar([i + 0.5*w for i in x], [r["asc_total_cost_s"] for r in rows], width=w,
       color="#0B7285", label="ASC total (build + runtime)")
ax.bar([i + 1.5*w for i in x], [r["tiered_total_cost_s"] for r in rows], width=w,
       color="#5B6672", label="Tiered total (no build cost)")
ax.set_yscale("log")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Compute time (sec, log)")
ax.set_title(f"Cost contributions vs footprint scale (single {P_COLLECT}-pulse collect)")
ax.legend()
ax.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/cost_contributions.png", dpi=150)
print("\nSaved cost_contributions.csv/.png")
