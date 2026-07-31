# sar_sim project state (as of this session)

Read this first in a new session before touching code — it's the fastest way back to where things stood. Written so a fresh Claude session with just folder access (no conversation history) can pick up the thread.

## What this project is

Investigating whether wall-ground and wall-wall multi-bounce SAR scattering can be modeled in **closed form** (point-ASC / attributed scattering centers) instead of expensive **SBR** (Shooting and Bouncing Rays) ray tracing, and validating the closed form against SBR as ground truth.

## Current status, one line each

- **Order-1 (direct wall return):** validated, matches SBR closely. Not in question anymore.
- **Order-2 (double bounce, wall-ground OR wall-wall):** validated on sparse scenes (SSIM ~0.90-0.93, coherence ~0.93-0.98). On a **denser** 15-building scene it degraded (SSIM 0.19) — flagged, not yet root-caused. Worth investigating before trusting order2 on dense/city-block scenes.
- **Order-3 (triple bounce):** implemented, mechanically verified correct (reflection math, roof-bounce dead-end, self-exclusion logic all checked by hand). SBR-vs-ASC agreement is bad (SSIM ~-0.02) but almost certainly because SBR under-samples order3 paths at any practical ray count (77→210 paths going from 4900→22500 rays, nowhere near converged) — this is very likely a sampling artifact, not a model bug, though not proven. **Confirmed not to matter**: order3's contribution to the combined image is negligible (~0.002 SSIM shift), so `run_tier2_multibounce.py` defaults to order1+2 only (`--order3` to include it).
- **Tier 2 (200-building, 1km×1km, real collection geometry) validation:** script written (`run_tier2_multibounce.py`) and smoke-tested for correctness (small scene, ran clean, figures render correctly), but **never run at full scale** — that's meant to run on the user's own GPU (RTX 4050 was the reference machine for the *original* single-bounce version of this validation). This is the natural next step when picking this back up.

## Key files (project root unless noted)

- `sar_sim/adaptive_facet_partition.py` — core validated fix: height-partitions wall facets by actual range sensitivity per bounce order, instead of one undivided facet per wall (which was wrong for tall walls — up to 14m of range migration collapsed into one point).
- `scene_multi_building_adaptive.py` — multi-building scene generator using the above. Carries `facet_building_id` per facet (needed because building facet counts vary, unlike the old uniform generator). **No collision avoidance** — buildings can spatially overlap at some seeds (confirmed at seed=1, density=350: buildings 1 and 3 overlap). Matters more as density goes up.
- `box_projected_multibounce.py` — the current closed-form model. Key pieces:
  - `_reflect_and_intersect_scene`: generalizes "reflect off a facet's normal, intersect the ground plane" to "intersect whichever is nearer: ground plane or another building's box" (closed-form ray/AABB slab test, not ray tracing).
  - `run_asc_box_projected_multibounce(..., include_order3=False)`: leg1+leg2 always, leg3 optional. Has per-pulse timing instrumentation (`stats['t_total_s']`, `stats['t_per_pulse_ms']`) matching the convention used elsewhere in this codebase.
- `run_multi_building_validation.py`, `run_tier2_multibounce.py` — the two "real" validation entry points (small toy scene, and the 200-building/1km real-geometry scene respectively). Both are meant to be run locally, not in a sandboxed environment — they're too slow for a ~45s tool-call budget at realistic settings.
- `run_order3_levels.py` / `plot_order3_levels.py` — displays order1/order2/order3 backprojected images side by side, SBR vs ASC. Useful for visually spot-checking any bounce-level fidelity question.
- `compare_ground_points.py`, `compare_occlusion_all.py`, `compare_sbr_vs_analytic_paths.py` — diagnostic scripts from earlier in the investigation, kept for reference if the SBR-vs-ASC divergence needs re-diagnosing.

## Real collection geometry (for Tier 2 / "real" validation runs)

Matches the original Tier 1/2 validation slides:
- standoff = 60000 m, altitude = 22500 m (→ ~20.6° depression angle)
- fc = 10 GHz, bandwidth = 600 MHz
- 2m cross-range resolution → pulses auto-bump to **450** via Nyquist (`--az-res 2`)
- footprint = 1000m × 1000m, density = 200 buildings/km² → seed=0 for the original slide's building layout (adaptive generator will produce MORE facets than the original "1000 facets" at the same density — that's the intended fix, not a bug, see `adaptive_facet_partition.py`)
- Original single-bounce reference timing (RTX 4050): 818x/pulse speedup, 1.7s ASC vs 1398s SBR total. The multi-bounce version will be slower on both sides (more physics being modeled) — no fresh timing at this scale yet.

## Gotchas discovered this session (don't re-derive these)

1. **A taper bug, not a physics bug**: an early version of the box-projected leg2/leg3 code reimplemented `_azimuth_sinc_taper` locally and got the sinc argument wrong (missing factor of 2). Caused a false "wall-to-wall bounces make things worse" result. Fix: always import `_azimuth_sinc_taper` from `multibounce_demo`, never reimplement it.
2. **Apparent cross-range "smearing" in backprojected images is NOT a bug** — it's a real SAR effect. A flat wall facet whose normal points near-exactly at the sensor's flight line has its own directivity peak fall mid-aperture (not at platform broadside), so only a narrow slice of the synthetic aperture actually illuminates it well — giving that facet's own return genuinely coarser cross-range resolution than the system nominal. Range resolution is unaffected (bandwidth-limited). Ruled out aliasing directly: identical smear at 300, 900, and 2700 pulses.
3. **Backprojection-tool filesystem sync**: editing files via the Windows-side Edit/Write tools can lag behind what the bash-mounted sandbox sees by more than one tool call — confirmed twice this session (edits reported "success" but the bash-mounted copy was stale). If a script mysteriously doesn't pick up a recent edit, don't trust — `cat` the file directly in bash to check, and if it's stale, write the file directly via a bash heredoc instead of relying on the sync.
4. **SBR order-N ground truth needs exponentially more rays as N grows** — a ray has to survive N sequential "landed in the right discrete cell" tests. Order2 needed ~120x120 rays to stabilize on a small scene; order3 was still climbing (77→210 paths) at 22500 rays without converging. Don't trust an SBR-vs-ASC mismatch at high bounce order as a model bug until ray density is pushed hard.

## Natural next steps (pick up here)

1. Run `run_tier2_multibounce.py` at full scale (or a middle-ground size first) on the user's GPU, to get real timing + fidelity numbers for the new two-figure Tier 2 deliverable.
2. Investigate the order2 SSIM regression on dense (15-building) scenes (0.19 vs ~0.90 on sparse scenes) — not yet root-caused, could matter more for the 200-building Tier 2 run.
3. Decide whether the building-overlap issue in the scene generator needs a real fix (collision avoidance) before trusting dense-scene validation numbers.
