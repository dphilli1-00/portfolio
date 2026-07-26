import json
import matplotlib.pyplot as plt

with open("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/scaling_data.json") as f:
    data = json.load(f)

rows = data["rows"]
N = [r["n_structures"] for r in rows]
ansys = [r["ansys_per_pulse_s"] * 1000 for r in rows]   # ms
tiered = [r["tiered_per_pulse_s"] * 1000 for r in rows]  # ms
mine = [r["mypipeline_per_pulse_s"] * 1000 for r in rows]  # ms

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(N, ansys, marker='o', color='#B85C00', linewidth=2, label='Ansys (dense SBR + UTD)')
ax.plot(N, tiered, marker='o', color='#5B6672', linewidth=2, linestyle='--',
        label='Fidelity-tiered dense SBR (mine, no ASC -- skip terrain)')
ax.plot(N, mine, marker='o', color='#0B7285', linewidth=2, label='My pipeline (ASC-cached)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Footprint scale (# canonical structures, N)')
ax.set_ylabel('Per-pulse phase-history compute time (ms)')
ax.set_title('Per-pulse compute vs footprint scale: dense SBR vs ASC-cached')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("/sessions/sweet-friendly-mayer/mnt/outputs/sar_sim/scaling_plot.png", dpi=150)
print("saved scaling_plot.png")
