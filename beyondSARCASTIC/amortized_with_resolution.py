"""
Combine the resolution-scoping assumption (delta and P vs N -- the
log-linear interpolation flagged as the weakest-grounded piece) directly
onto the amortized cost-per-collect plot, so the assumption driving each
curve is visible in the same figure instead of buried in a legend/table.

Top panel: resolution (m) and pulse count (pulses/collect) vs footprint
           scale N -- this IS the assumption, shown explicitly.
Bottom panel: amortized cost per collect vs number of collects, one curve
              pair per scale, colored to match the top panel.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter, NullFormatter

plt.close('all')
def format_time(x, pos=None):
    if x <= 0:
        return ""
    if x < 60:
        return f"{x:.3g}s"
    if x < 3600:
        return f"{x/60:.3g}min"
    if x < 86400:
        return f"{x/3600:.3g}hr"
    return f"{x/86400:.3g}day"

C = 299_792_458.0
T_FIXED = 2e-3
R_ANSYS = 75e-6
R_ASC = 0.5e-6
B_WORST = 2.0
fc = 10e9
wavelength = C / fc
R_STANDOFF = 600_000.0
# Same sensor mode / same resolution for every scale (2m GSD, matching real
# TerraSAR-X Spotlight) -- NOT re-tuned per scene. What changes with scene
# size is the imaged FOOTPRINT, and a smaller footprint needs fewer azimuth
# pulses/lines to cover it at that fixed resolution -- same physical logic
# as cropping fewer lines out of the same collect, not a different waveform.
#
# Assume roughly uniform structure density (same avg building spacing
# everywhere in the city), so imaged AREA scales with N and imaged LINEAR
# extent L scales with sqrt(N): L(N) = L_CITY * sqrt(N / N_CITY).
# Pulse count is grounded at the real anchor (TerraSAR-X: ~4,497 pulses to
# cover the full 10km x 10km city at 2m res, from P = R*lambda/delta^2) and
# scaled down proportionally to footprint for smaller scenes:
#   P(N) = P_CITY_REAL * sqrt(N / N_CITY)
DELTA_FIXED = 2.0          # m, constant across all scales
N_CITY = 20000
L_CITY = 10_000.0          # m, TerraSAR-X spotlight footprint side
P_CITY_REAL = R_STANDOFF * wavelength / (DELTA_FIXED ** 2)   # ~4,497 pulses, grounded anchor

scenes = [
    ("4m trihedral", 1),
    ("Target cluster", 5),
    ("City block", 25),
    ("Small district", 125),
    ("District", 625),
    ("Large district", 3125),
    ("Full city", 20000),
]

def footprint_for_N(N):
    return L_CITY * math.sqrt(N / N_CITY)

def P_for_N(N):
    return P_CITY_REAL * math.sqrt(N / N_CITY)

rows = []
for label, N in scenes:
    delta = DELTA_FIXED
    P = P_for_N(N)
    ansys_pc = P * (T_FIXED + N * R_ANSYS)
    floor = P * (T_FIXED + N * R_ASC)
    build = N * B_WORST
    n_star = build / (ansys_pc - floor)
    rows.append(dict(scene=label, N=N, delta=delta, P=P, ansys_pc=ansys_pc, floor=floor,
                      build=build, n_star=n_star))

colors = cm.viridis(np.linspace(0.05, 0.9, len(scenes)))
N_vals = [r["N"] for r in rows]

fig = plt.figure(figsize=(11, 8))
fig2= plt.figure(figsize=(11, 8))
# --- TOTAL cost vs n (not divided by n) ---
ax1 = fig.add_subplot(111)
ax2 =  fig2.add_subplot(111)
n_plot = np.array([1, 1.5, 2, 3, 5, 6, 8, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000])
for r, c in zip(rows, colors):
    ansys_total = r["ansys_pc"] * n_plot          # linear in n -- 2 passes cost 2X, etc.
    mine_total = r["build"] + r["floor"] * n_plot  # fixed build cost + linear runtime
    ax1.plot(n_plot, ansys_total, color=c, linewidth=2, linestyle='-',
              label=f"{r['scene']} (N={r['N']:,}, δ={r['delta']:.1f}m) -- Ansys")
    ax1.plot(n_plot, mine_total, color=c, linewidth=2, linestyle='--', marker='o', markersize=3,
              label=f"{r['scene']} -- mine")
    ax1.scatter([r["n_star"]], [r["ansys_pc"] * r["n_star"]], color=c, marker='X', s=110, zorder=6,
                 edgecolor='black', linewidth=0.8)
    ax2.plot(ansys_total/mine_total)

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Number of collects run against the same built library (log)')
ax1.set_ylabel('TOTAL compute time (log)')
ax1.yaxis.set_major_formatter(FuncFormatter(format_time))
ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.set_ylim([0,86400])  # cap at 24hr
ax1.set_title('Total cost vs number of collects (X marks each scale\'s break-even n*)')
ax1.legend(fontsize=6.5, ncol=2, loc='lower right')
ax1.grid(True, which='both', alpha=0.25)
ax1.set_xlim([0,1000])

fig.tight_layout()
#fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/amortized_with_resolution.png", dpi=150)
print("Saved amortized_with_resolution.png")
for r in rows:
    print(f"{r['scene']:16s} N={r['N']:6d} delta={r['delta']:6.2f}m P={r['P']:7.0f} n*={r['n_star']:7.1f}")

# ---------------------------------------------------------------------
# The X markers on the main plot trend UP-AND-LEFT as N grows: bigger
# scenes cost more per collect (up), but need FEWER repeat collects to
# break even (n* shrinks, so left). That's the real finding, but it reads
# backwards against the usual "up-and-right = better/bigger" chart
# convention, because the x-axis there is "number of collects," and n*
# is inversely related to N.
#
# This second view swaps the x-axis to N (footprint scale) directly and
# plots the TOTAL COST AT THE CROSSOVER POINT. Both increase together
# with N -- bigger scenes need a bigger absolute investment to reach
# break-even, even though that break-even arrives faster in relative
# terms (fewer collects). Same underlying numbers, reoriented so the
# trend reads up-and-right.
# ---------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 6.5))
N_sorted = [r["N"] for r in rows]
cost_at_crossover = [r["ansys_pc"] * r["n_star"] for r in rows]
ax3.plot(N_sorted, cost_at_crossover, color='#5B6672', linewidth=1.5, zorder=1)
for r, c, cost in zip(rows, colors, cost_at_crossover):
    ax3.scatter([r["N"]], [cost], color=c, marker='X', s=160, zorder=5,
                edgecolor='black', linewidth=0.9)
    ax3.annotate(f"n*={r['n_star']:.0f}", (r["N"], cost), textcoords="offset points",
                 xytext=(8, 6), fontsize=8, color='#333333')
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlabel('Footprint scale (# structures, N) -- log')
ax3.set_ylabel('Total cost AT break-even (log)')
ax3.yaxis.set_major_formatter(FuncFormatter(format_time))
ax3.yaxis.set_minor_formatter(NullFormatter())
ax3.set_title('Crossover point vs footprint scale -- reoriented to trend up-and-right\n'
              '(bigger scene = bigger break-even investment, reached in fewer relative collects)')
ax3.grid(True, which='both', alpha=0.3)
fig3.tight_layout()
#fig3.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/crossover_vs_scale.png", dpi=150)
print("Saved crossover_vs_scale.png")

# ---------------------------------------------------------------------
# Normalize mine's total cost by Ansys's total cost for the SAME number
# of passes n. ratio(n) = mine_total(n) / ansys_total(n)
#                        = build/(ansys_pc*n) + floor/ansys_pc
# At n=1: ratio = (build+floor)/ansys_pc -- how much worse mine is for a
#   single one-off collect (can be >>1).
# As n->infinity: ratio -> floor/ansys_pc = 1/speedup -- the pure runtime
#   advantage, unreachable in practice but the asymptotic floor.
# ratio=1 exactly at n=n* by construction (that's the break-even
# definition) -- so this is the same crossover, just replotted as a
# single normalized curve per scale with a flat "parity" line at y=1,
# instead of two absolute-cost curves per scale.
# ---------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(9.5, 7))
n_plot2 = np.array([1, 1.2, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000])
for r, c in zip(rows, colors):
    ratio = (r["build"] / (r["ansys_pc"] * n_plot2) + r["floor"] / r["ansys_pc"])
    ax4.plot(n_plot2, ratio, color=c, linewidth=2, marker='o', markersize=3,
              label=f"{r['scene']} (N={r['N']:,})")
    ax4.scatter([r["n_star"]], [1.0], color=c, marker='X', s=130, zorder=6,
                edgecolor='black', linewidth=0.9)

ax4.axhline(1.0, color='black', linewidth=1.2, linestyle='-', alpha=0.7, zorder=2)
ax4.text(1.05, 1.15, 'parity (mine == Ansys for the same n passes)', fontsize=8, color='#333')
ax4.set_xscale('log'); ax4.set_yscale('log')
ax4.set_xlabel('Number of passes, n (log)')
ax4.set_ylabel('mine_total(n) / ansys_total(n)  --  log\n(below 1.0 = mine cheaper)')
ax4.set_title('My approach normalized by Ansys, for the same number of passes\n'
              '(X marks each scale\'s break-even -- where the curve crosses parity)')
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, which='both', alpha=0.3)
fig4.tight_layout()
#fig4.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/normalized_by_ansys.png", dpi=150)
print("Saved normalized_by_ansys.png")

# ---------------------------------------------------------------------
# Number of passes needed to see improvement (n*) directly on the x-axis,
# paired with footprint scale N on the y-axis.
# ---------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(9, 6.5))
n_star_vals = [r["n_star"] for r in rows]
N_vals2 = [r["N"] for r in rows]
ax5.plot(n_star_vals, N_vals2, color='#5B6672', linewidth=1.5, zorder=1)
for r, c in zip(rows, colors):
    ax5.scatter([r["n_star"]], [r["N"]], color=c, marker='X', s=160, zorder=5,
                edgecolor='black', linewidth=0.9)
    ax5.annotate(r["scene"], (r["n_star"], r["N"]), textcoords="offset points",
                 xytext=(8, 6), fontsize=8, color='#333333')
ax5.set_xscale('log'); ax5.set_yscale('log')
ax5.set_xlabel('Number of passes needed to see improvement, n* (log)')
ax5.set_ylabel('Footprint scale (# structures, N) -- log')
ax5.set_title('Break-even pass count vs footprint scale')
ax5.grid(True, which='both', alpha=0.3)
fig5.tight_layout()
#fig5.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/n_star_vs_scale.png", dpi=150)
print("Saved n_star_vs_scale.png")
for r in rows:
    print(f"{r['scene']:16s} n*={r['n_star']:7.1f} N={r['N']:6d}")

# ---------------------------------------------------------------------
# The combination that actually trends up-and-right for "benefit":
# x = footprint scale N, y = asymptotic speedup factor (ansys_pc/floor).
# Unlike n* (cost of REACHING the benefit -- decreases with N) or
# cost-at-crossover (entangles absolute cost growth with n*), speedup
# is the SIZE of the benefit itself once past break-even, and it
# increases monotonically with N. Both axes climb together.
# ---------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(9, 6.5))
speedups = [r["ansys_pc"] / r["floor"] for r in rows]
ax6.plot(N_vals, speedups, color='#5B6672', linewidth=1.5, zorder=1)
for r, c, sp in zip(rows, colors, speedups):
    ax6.scatter([r["N"]], [sp], color=c, marker='o', s=140, zorder=5,
                edgecolor='black', linewidth=0.9)
    ax6.annotate(f"{r['scene']}\n{sp:.1f}x", (r["N"], sp), textcoords="offset points",
                 xytext=(8, 4), fontsize=8, color='#333333')
ax6.set_xscale('log'); ax6.set_yscale('log')
ax6.set_xlabel('Footprint scale (# structures, N) -- log')
ax6.set_ylabel('Asymptotic speedup, ansys/mine (log)')
ax6.set_title('Benefit increases with scale: speedup vs footprint scale\n(up-and-right = bigger, better)')
ax6.grid(True, which='both', alpha=0.3)
fig6.tight_layout()
#fig6.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/speedup_vs_scale.png", dpi=150)
print("Saved speedup_vs_scale.png")
for r, sp in zip(rows, speedups):
    print(f"{r['scene']:16s} N={r['N']:6d} speedup={sp:7.1f}x")

# ---------------------------------------------------------------------
# Combine the speedup-vs-scale story (up-and-right, the "benefit") with
# the raw underlying Ansys and ASC per-collect numbers it's derived from,
# so nobody has to take the ratio on faith.
# Top: speedup = ansys_pc / floor, vs N.
# Bottom: the two raw curves themselves (ansys_pc, floor) vs N.
# ---------------------------------------------------------------------
fig7 = plt.figure(figsize=(10, 13.5))
gs7 = fig7.add_gridspec(3, 1, height_ratios=[1.1, 0.9, 1.1], hspace=0.32)

ax7a = fig7.add_subplot(gs7[0])
ax7a.plot(N_vals, speedups, color='#5B6672', linewidth=1.5, zorder=1)
for r, c, sp in zip(rows, colors, speedups):
    ax7a.scatter([r["N"]], [sp], color=c, marker='o', s=140, zorder=5,
                 edgecolor='black', linewidth=0.9)
    ax7a.annotate(f"{sp:.1f}x", (r["N"], sp), textcoords="offset points",
                  xytext=(6, 4), fontsize=8, color='#333333')
ax7a.set_xscale('log'); ax7a.set_yscale('log')
ax7a.set_ylabel('Asymptotic speedup (log)')
ax7a.set_title('Benefit vs footprint scale (top), underlying Ansys/ASC data it comes from (bottom)')
ax7a.grid(True, which='both', alpha=0.3)

ax7b = fig7.add_subplot(gs7[1], sharex=ax7a)
ansys_vals = [r["ansys_pc"] for r in rows]
floor_vals = [r["floor"] for r in rows]
ax7b.plot(N_vals, ansys_vals, color='#B85C00', linewidth=2, marker='o', markersize=6,
          label='Ansys dense rays (per collect)')
ax7b.plot(N_vals, floor_vals, color='#0B7285', linewidth=2, marker='o', markersize=6,
          label='My approach -- runtime floor (per collect)')

# third curve: the one-time fixed build cost (N*B_WORST) -- not a
# per-collect quantity, it's paid once before the runtime floor even
# applies. Plotted on the same time axis, distinct linestyle, to show
# what "getting there" actually costs upfront at each scale.
build_vals = [r["build"] for r in rows]
ax7b.plot(N_vals, build_vals, color='#333333', linewidth=2, linestyle=':', marker='s', markersize=6,
          label='My approach -- ONE-TIME build cost (fixed, paid once)')

ax7b.set_xscale('log'); ax7b.set_yscale('log')
ax7b.set_xlabel('Footprint scale (# structures, N) -- log')
ax7b.set_ylabel('Time (log)\nper-collect (solid) vs one-time (dotted)')
ax7b.yaxis.set_major_formatter(FuncFormatter(format_time))
ax7b.yaxis.set_minor_formatter(NullFormatter())
ax7b.legend(fontsize=8.5)
ax7b.grid(True, which='both', alpha=0.3)

# --- third panel: concretely show the payoff at the flagship scale ---
# For the full city (N=20,000), plot TOTAL cost vs number of passes n,
# and mark n*=6: at or beyond 6 passes against the same built library,
# mine is cheaper than Ansys for the same work. Below 6, Ansys wins.
city_row = rows[-1]
ax7c = fig7.add_subplot(gs7[2])
n_plot3 = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000])
ansys_city = city_row["ansys_pc"] * n_plot3
mine_city = city_row["build"] + city_row["floor"] * n_plot3
ax7c.plot(n_plot3, ansys_city, color='#B85C00', linewidth=2.5, label='Ansys dense rays (total)')
ax7c.plot(n_plot3, mine_city, color='#0B7285', linewidth=2.5, label='My approach (total = build + n*floor)')
ax7c.axvline(city_row["n_star"], color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax7c.scatter([city_row["n_star"]], [city_row["ansys_pc"] * city_row["n_star"]], color='#5B6672',
             marker='X', s=180, zorder=6, edgecolor='black', linewidth=1.0)
ax7c.annotate(f"n* = {city_row['n_star']:.0f} passes\nbeyond this, mine wins",
              (city_row["n_star"], city_row["ansys_pc"] * city_row["n_star"]),
              textcoords="offset points", xytext=(12, -28), fontsize=10, color='#222222',
              arrowprops=dict(arrowstyle='->', color='#333333', lw=1))
ax7c.fill_betweenx([min(mine_city.min(), ansys_city.min()) * 0.7, max(mine_city.max(), ansys_city.max()) * 1.3],
                    city_row["n_star"], n_plot3.max(), color='#0B7285', alpha=0.06, zorder=0)
ax7c.text(n_plot3.max() * 0.55, mine_city.max() * 0.5, '6+ PASSES\n= NET BENEFIT',
          fontsize=10, color='#0B7285', ha='center', fontweight='bold')
ax7c.set_xscale('log'); ax7c.set_yscale('log')
ax7c.set_xlabel('Number of passes, n (log)')
ax7c.set_ylabel('Total compute time (log)')
ax7c.yaxis.set_major_formatter(FuncFormatter(format_time))
ax7c.yaxis.set_minor_formatter(NullFormatter())
ax7c.set_title(f'Full city (N={city_row["N"]:,}): 6 or more passes over the same library -> net benefit')
ax7c.legend(fontsize=9, loc='upper left')
ax7c.grid(True, which='both', alpha=0.3)

#fig7.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/speedup_with_underlying_data.png", dpi=150)
print("Saved speedup_with_underlying_data.png")
for r in rows:
    print(f"{r['scene']:16s} N={r['N']:6d} build={r['build']:9.1f}s ansys_pc={r['ansys_pc']:9.3f}s floor={r['floor']:8.3f}s")

# ---------------------------------------------------------------------
# Same "total cost vs number of passes, n* marked" panel as the city
# case, now repeated for EVERY footprint scale as small multiples.
# ---------------------------------------------------------------------
fig8, axes8 = plt.subplots(3, 3, figsize=(15, 13))
axes8_flat = axes8.flatten()

for i, (r, c) in enumerate(zip(rows, colors)):
    ax = axes8_flat[i]
    n_plot4 = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000])
    ansys_r = r["ansys_pc"] * n_plot4
    mine_r = r["build"] + r["floor"] * n_plot4
    ax.plot(n_plot4, ansys_r, color='#B85C00', linewidth=2, label='Ansys (total)')
    ax.plot(n_plot4, mine_r, color='#0B7285', linewidth=2, label='Mine (total)')
    ax.axvline(r["n_star"], color='black', linestyle='--', linewidth=1.3, alpha=0.8)
    ax.scatter([r["n_star"]], [r["ansys_pc"] * r["n_star"]], color=c, marker='X', s=140, zorder=6,
               edgecolor='black', linewidth=0.9)
    ymin = min(mine_r.min(), ansys_r.min()) * 0.7
    ymax = max(mine_r.max(), ansys_r.max()) * 1.3
    ax.fill_betweenx([ymin, ymax], r["n_star"], n_plot4.max(), color='#0B7285', alpha=0.07, zorder=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"{r['scene']} (N={r['N']:,})\nn* = {r['n_star']:.0f} passes -> net benefit", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, which='both', alpha=0.25)
    if i == 0:
        ax.legend(fontsize=7, loc='upper left')

# hide the two unused grid slots (7 scenes in a 3x3 grid)
for j in range(len(rows), len(axes8_flat)):
    axes8_flat[j].axis('off')

fig8.suptitle('Break-even (n*) for every footprint scale, single-target trihedral to full city', fontsize=14)
fig8.tight_layout(rect=[0, 0, 1, 0.97])
#fig8.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/breakeven_all_scales_grid.png", dpi=150)
print("Saved breakeven_all_scales_grid.png")

# ---------------------------------------------------------------------
# Trimmed to three representative anchors: trihedral, district, city.
# Each panel's title now also labels the resolution (delta) used to set
# up that scale's collect (the assumption feeding pulse count P).
# ---------------------------------------------------------------------
anchor_indices = [0, 4, 6]  # 4m trihedral, District, Full city
fig9, axes9 = plt.subplots(1, 3, figsize=(16, 5.5))

for ax, idx in zip(axes9, anchor_indices):
    r = rows[idx]
    c = colors[idx]
    n_plot5 = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000])
    ansys_r = r["ansys_pc"] * n_plot5
    mine_r = r["build"] + r["floor"] * n_plot5
    ax.plot(n_plot5, ansys_r, color='#B85C00', linewidth=2.2, label='Ansys (total)')
    ax.plot(n_plot5, mine_r, color='#0B7285', linewidth=2.2, label='Mine (total)')
    ax.axvline(r["n_star"], color='black', linestyle='--', linewidth=1.3, alpha=0.8)
    ax.scatter([r["n_star"]], [r["ansys_pc"] * r["n_star"]], color=c, marker='X', s=160, zorder=6,
               edgecolor='black', linewidth=1.0)
    ymin = min(mine_r.min(), ansys_r.min()) * 0.7
    ymax = max(mine_r.max(), ansys_r.max()) * 1.3
    ax.fill_betweenx([ymin, ymax], r["n_star"], n_plot5.max(), color='#0B7285', alpha=0.07, zorder=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Number of passes, n (log)')
    ax.set_title(f"{r['scene']} (N={r['N']:,}, setup δ={r['delta']:.1f}m)\n"
                 f"n* = {r['n_star']:.0f} passes -> net benefit", fontsize=10.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, which='both', alpha=0.25)
    if idx == anchor_indices[0]:
        ax.legend(fontsize=8, loc='upper left')

axes9[0].set_ylabel('Total compute time (log)')
fig9.suptitle('Break-even (n*): trihedral vs district vs full city', fontsize=14)
fig9.tight_layout(rect=[0, 0, 1, 0.94])
#fig9.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/breakeven_three_anchors.png", dpi=150)
print("Saved breakeven_three_anchors.png")

# ---------------------------------------------------------------------
# Same three anchors (trihedral, district, city), now overlaid on ONE
# set of axes instead of separate subplots.
# ---------------------------------------------------------------------
fig10, ax10 = plt.subplots(figsize=(10, 7.5))
n_plot6 = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000])

for idx in anchor_indices:
    r = rows[idx]
    c = colors[idx]
    ansys_r = r["ansys_pc"] * n_plot6
    mine_r = r["build"] + r["floor"] * n_plot6
    ax10.plot(n_plot6, ansys_r, color=c, linewidth=2.3, linestyle='-',
              label=f"{r['scene']} (N={r['N']:,}, δ={r['delta']:.1f}m) -- Ansys")
    ax10.plot(n_plot6, mine_r, color=c, linewidth=2.3, linestyle='--', marker='o', markersize=3,
              label=f"{r['scene']} -- ASC Cached")
    # n* < 1 means ASC is already cheaper on the very first collect -- the
    # shared per-pulse fixed overhead (large P at fine resolution) swamps the
    # build tax before n=1 even happens. Don't plot the X off the left edge
    # of the data (n_plot6 starts at 1); mark n=1 itself as "already ahead".
    if r["n_star"] >= 1:
        ax10.scatter([r["n_star"]], [r["ansys_pc"] * r["n_star"]], color=c, marker='X', s=170, zorder=6,
                     edgecolor='black', linewidth=1.0)
        ax10.annotate(f"n*={r['n_star']:.0f}", (r["n_star"], r["ansys_pc"] * r["n_star"]),
                      textcoords="offset points", xytext=(8, -14), fontsize=9, color='#222222', fontweight='bold')
    else:
        # No text label here -- the setup-delta box below already says
        # "ASC already faster at n=1"; a second label just collides with it.
        ax10.scatter([1], [r["ansys_pc"]], color=c, marker='X', s=170, zorder=6,
                     edgecolor='black', linewidth=1.0)

# Annotate the gap at n=1 for ALL THREE anchors -- this vertical gap IS
# the one-time build cost for that scale. Paid once, at n=1, before the
# runtime floor ever kicks in -- everything to the right is the payoff.
def fmt_delta(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}hr"

x_ann = 1.0
text_offsets = [(16, 8), (16, 8), (16, 8)]  # trihedral, district, city -- all above the "mine" line
for idx, offset in zip(anchor_indices, text_offsets):
    r = rows[idx]
    c = colors[idx]
    y_ansys_1 = r["ansys_pc"] * x_ann
    y_mine_1 = r["build"] + r["floor"] * x_ann
    delta = y_mine_1 - y_ansys_1
    if delta >= 0:
        ax10.annotate('', xy=(x_ann, y_ansys_1), xytext=(x_ann, y_mine_1),
                      arrowprops=dict(arrowstyle='<->', color=c, lw=1.8))
        label_txt = f"{r['scene']} setup: Δ={fmt_delta(delta)} (paid ONCE)"
        label_xy = (x_ann, y_mine_1)
    else:
        # ASC is already cheaper at n=1 -- the fixed per-pulse overhead (huge P
        # at fine resolution) dwarfs the build tax, so there's no amortization
        # gap to pay down. Arrow points the other way; label off the Ansys line.
        ax10.annotate('', xy=(x_ann, y_mine_1), xytext=(x_ann, y_ansys_1),
                      arrowprops=dict(arrowstyle='<->', color=c, lw=1.8))
        label_txt = f"{r['scene']}: ASC already {fmt_delta(-delta)} faster at n=1"
        label_xy = (x_ann, y_ansys_1)
    ax10.annotate(label_txt,
                  xy=label_xy, xytext=offset, textcoords="offset points",
                  fontsize=8.5, color='#222222', va='bottom', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=c, alpha=0.92, linewidth=1.5))
    print(f"{r['scene']} n=1 gap: ansys={y_ansys_1:.2f}s ASC Cached={y_mine_1:.2f}s delta={delta:.2f}s")

ax10.set_xscale('log'); ax10.set_yscale('log')
ax10.set_xlabel('Number of passes, n (log)')
ax10.set_ylabel('Total compute time (log)')
ax10.yaxis.set_major_formatter(FuncFormatter(format_time))
ax10.yaxis.set_minor_formatter(NullFormatter())
ax10.set_title('Break-even (n*): trihedral vs district vs full city, same axes')
ax10.legend(fontsize=8.5, loc='upper left')
ax10.grid(True, which='both', alpha=0.3)
fig10.tight_layout()
fig10.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/breakeven_three_anchors_combined.png", dpi=150)
print("Saved breakeven_three_anchors_combined.png")
print(f"City n=1 gap: ansys={y_ansys_1:.1f}s mine={y_mine_1:.1f}s delta={y_mine_1-y_ansys_1:.1f}s")
