"""
Same footprint-scale sweep as cost_contributions.py, but now assuming the
ASC library is built INCREMENTALLY as the scene grows: structures already
built at a smaller scale are reused, not rebuilt, when you scale up to the
next tier. Only the NEWLY added structures at each step pay the one-time
build cost B.

This changes the "ASC build cost" bar at each step from N_k * B (cold start,
as if you built everything from scratch every time -- what the previous
script assumed) to (N_k - N_[k-1]) * B (marginal, new structures only).

Tiered dense SBR has no such concept -- it never caches anything, so it
always re-traces every structure in the scene, every pulse, regardless of
whether the scene grew incrementally or not. Reuse only helps ASC.

scenes are treated as NESTED supersets (city contains district contains
block, etc.) -- i.e. going from N=3,125 to N=20,000 means 16,875 genuinely
new structures, and the 3,125 from the previous step are already in the
library for free.
"""

import csv
import matplotlib.pyplot as plt

T_FIXED = 2e-3
R_TIERED = 50e-6
R_ASC = 0.5e-6
B_BUILD = 2.0
P_COLLECT = 500

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
n_prev = 0
cumulative_asc_reuse = 0.0
for label, N in scenes:
    n_new = N - n_prev
    build_incremental = n_new * B_BUILD
    build_coldstart = N * B_BUILD
    runtime = P_COLLECT * (T_FIXED + N * R_ASC)
    asc_step_reuse_total = build_incremental + runtime
    asc_coldstart_total = build_coldstart + runtime
    tiered_total = P_COLLECT * (T_FIXED + N * R_TIERED)
    cumulative_asc_reuse += asc_step_reuse_total
    savings_vs_coldstart = asc_coldstart_total - asc_step_reuse_total
    rows.append({
        "scene": label,
        "n_structures": N,
        "n_new_structures": n_new,
        "asc_build_incremental_s": build_incremental,
        "asc_runtime_s": runtime,
        "asc_step_total_with_reuse_s": asc_step_reuse_total,
        "asc_total_cold_start_s": asc_coldstart_total,
        "savings_from_reuse_s": savings_vs_coldstart,
        "tiered_total_s": tiered_total,
        "cumulative_asc_campaign_cost_s": cumulative_asc_reuse,
    })
    n_prev = N

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/cost_contributions_reuse.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'scene':10s} {'new':>7s} {'reuse_total':>12s} {'coldstart':>11s} {'savings':>9s} {'tiered':>9s}")
for r in rows:
    print(f"{r['scene']:10s} {r['n_new_structures']:7d} "
          f"{r['asc_step_total_with_reuse_s']:11.2f}s {r['asc_total_cold_start_s']:10.2f}s "
          f"{r['savings_from_reuse_s']:8.2f}s {r['tiered_total_s']:8.2f}s")

print(f"\nCumulative ASC campaign cost through city scale (7 collects, one per step): "
      f"{rows[-1]['cumulative_asc_campaign_cost_s']:.1f}s")

fig, ax = plt.subplots(figsize=(10, 6))
labels = [r["scene"] for r in rows]
x = range(len(labels))
w = 0.25
ax.bar([i - w for i in x], [r["asc_step_total_with_reuse_s"] for r in rows], width=w,
       color="#0B7285", label="ASC step total (WITH reuse -- only new structures rebuilt)")
ax.bar([i for i in x], [r["asc_total_cold_start_s"] for r in rows], width=w,
       color="#4FD8EB", label="ASC total (cold start -- rebuild everything, for comparison)")
ax.bar([i + w for i in x], [r["tiered_total_s"] for r in rows], width=w,
       color="#5B6672", label="Tiered total (no caching, no reuse benefit possible)")
ax.set_yscale("log")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Compute time for this step's 500-pulse collect (sec, log)")
ax.set_title("Effect of structure reuse: incremental ASC build cost vs cold start, by footprint scale")
ax.legend(fontsize=9)
ax.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/cost_contributions_reuse.png", dpi=150)
print("\nSaved cost_contributions_reuse.csv/.png")
