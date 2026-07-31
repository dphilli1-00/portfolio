# Closed-form SAR multi-bounce scattering — project handoff

Saved as a standalone record so this can be picked up as its own project,
independent of the chat session it grew out of. Written for a reader (human
or a fresh Claude session) who wasn't there for any of it.

## What this project is

Two linked threads:

1. **A game-engine-style SAR simulation pipeline**, positioned explicitly
   against commercial SBR tools (Ansys HFSS SBR+, etc.): author a 3D scene as
   plain arrays, ray-trace multi-bounce reflectivity, synthesize phase
   history, backproject to an image. The pitch (see `presentation/`) is
   speed — an ASC-cached closed-form path amortizes the cost of re-tracing a
   static scene across many looks, at a measured 818x speedup over dense
   ray tracing for single-bounce, validated to SSIM >= 0.95 per building on
   a 200-building city scene.

2. **A deep, still-open investigation into whether wall-ground (and
   wall-ground-wall) multi-bounce returns can be modeled in closed form —
   with no ray tracing at all — well enough to match a ray-traced
   reference.** This is the harder, more interesting half, and where most
   of the real intellectual content lives. It has NOT converged to a clean
   answer. That's reported honestly below and in the deck's correction/
   follow-up slides, not papered over.

## Directory layout

```
outputs/
  sar_sim/            all simulation code, validation harnesses, diagnostics
  presentation/        the pptx deck (build_deck.js generates sar_sim_pipeline.pptx)
  PROJECT_SUMMARY.md   this file
```

### `sar_sim/` — core modules (stable, validated)

- `materials.py` — Fresnel reflectivity + diffuse coefficient per material
  (concrete, dry soil, etc.), `effective_specular_reflectivity(material,
  theta, wavelength)` used everywhere a ground-bounce needs an angle- and
  wavelength-dependent reflection coefficient.
- `dense_sbr_demo.py` — the reference ray tracer. `make_building_scene`,
  `make_ground_facet`, `ray_facet_intersect` (chunked, vectorized
  ray-facet intersection — the expensive O(rays x facets) kernel),
  `make_aim_grid`, `get_backend` (numpy/cupy switch), `run_dense_sbr` /
  `run_dense_sbr_timed` (the single-bounce numbers behind the deck's Tier 2
  and speedup slides — **never modified by later multi-bounce work**, kept
  isolated on purpose).
- `sbr_vs_asc_compare.py` — single-bounce ASC-cached closed-form path
  (`run_asc_cached`) and the harness that produced the 818x / SSIM>=0.95
  Tier 2 numbers.
- `multibounce_demo.py` — everything multi-bounce. The dense-SBR recursive
  ray tracer extended to order 2/3 (`run_multibounce_sbr`), the closed-form
  ASC-cached double-bounce path (`run_asc_cached_multibounce`), the
  general closed-form triple-bounce chain
  (`run_asc_cached_order3_wall_ground_wall`), and every taper/decomposition
  variant tried along the way — **all left in place with full docstrings
  documenting what was measured**, not deleted when they didn't work. This
  file's docstrings are themselves close to a lab notebook; read them before
  re-deriving anything.
- `validation_tier2_compare.py` — single-bounce SSIM/coherence validation
  harness (`backproject`, `required_freq_samples`, `compute_ref_ranges`,
  per-building scoring).
- `validation_multibounce_compare.py` — `score_whole_image` (SSIM, amp
  RMS, phase RMS, complex coherence) used by every multi-bounce A/B test.
- `radar_params.py`, `scene.py`, `point_scatterer_demo.py` — earlier/
  simpler single-bounce point-scatterer demo (Tier 1 MVP), superseded by
  `dense_sbr_demo.py` for anything multi-bounce but still the simplest
  worked example if you want to understand backprojection from scratch.
- `amortized_*.py`, `cost_contributions*.py`, `footprint_sweep_*.py`,
  `scaling_*.py`, `speedup_breakdown.py`, `worst_case_vs_ansys.py`,
  `final_scaling_real_params.py` — the cost-model / speedup-estimate
  scripts behind the deck's city-scale compute slides.
- `docs/pipeline_outline.md` — original slide-outline design doc (Tier 1
  MVP scope). Superseded in structure by the actual deck but still useful
  background on the original motivation.

### `sar_sim/` — diagnostics and A/B tests (scratch, but kept for the record)

Every `_diagnose_*.py` and `_taper_ab_test*.py` / `_test_order3_*.py`
script is a one-off measurement, not production code, kept because the
numbers they produced are cited in docstrings and in this document and
should be reproducible. Notable ones:

- `_diagnose_dl.py` — measured actual wall-to-ground round-trip
  path-length spread (ΔL) across real facets: min -16.8m, max 201.6m, mean
  43.2m — thousands of wavelengths at X-band, the number that falsified
  the "single point + amplitude taper" approach to the range-direction
  problem.
- `_diagnose_hit_density.py` — measured the ray tracer's own hit-density
  histogram vs. wall height: bimodal, peaks at z-fraction 0.4-0.6, mean
  0.513, NOT uniform.
- `_diagnose_sbr_convergence.py` — order-2 ray-density convergence sweep:
  self-consistent to coherence >=0.9995 across a 9x ray-count increase.
  Order-2 is a trustworthy reference.
- `_diagnose_order3_convergence.py` — the same sweep for order-3: **0 → 7
  → 20 unique paths as ray density goes 80 → 100 → 130 rays/side, with no
  sign of leveling off.** Order-3 is NOT a trustworthy reference at any
  ray density tested. This is the single most important negative result
  in the whole investigation — see below.
- `_taper_ab_test.py` / `_taper_ab_test2.py` / `_taper_ab_test3.py` — the
  range-taper and regime-adaptive A/B tests (numbers summarized below).
- `_test_order3_closed_form.py` — the closed-form order-3 validation
  harness.

### `presentation/`

- `build_deck.js` — generates `sar_sim_pipeline.pptx` (currently 38
  slides). Run with `node build_deck.js` from inside `presentation/`.
  Structure: motivation/positioning → landscape (Ansys comparison) →
  pipeline steps (materials, multi-bounce, polarimetry, weather) → Tier 2
  validation (SSIM, coherence) → **correction slide** (Tier 2 is
  single/double bounce only, triple bounce exposed real gaps) →
  **follow-up slide** (the ground-bounce fidelity investigation, azimuth
  taper shipped, range-direction decomposition didn't beat baseline,
  regime-adaptive threshold measured the wrong physical quantity) →
  summary → backup slides (case studies, extensions).
- `render_city_3d.py`, `gen_speed_figure.py`, `render_icons.js` — figure
  generation feeding the deck.

## What's actually validated (Tier 2, single/double bounce)

Full city scene: 200 buildings, 1000 facets, 1km x 1km, real collection
geometry (60km standoff / 22.5km altitude, ~20.6° depression, 2m
resolution, 450-pulse aperture). Per-building SSIM >= 0.95 pass rate:
98.5%. 818x measured speedup (dense SBR vs. ASC-cached), on a single
consumer laptop GPU (RTX 4050) — a measured number, not a Fermi estimate.
Coherence (a stricter, complex-valued check) catches 16/200 buildings
(8%) that SSIM alone passes but coherence doesn't — worth a coherence
threshold alongside SSIM in the Tier 2 criteria, not a replacement for it.

**This validated result is single-bounce and wall-ground double-bounce
only.** Triple bounce is where things got interesting and did not
resolve cleanly. That's the rest of this document.

## The multi-bounce fidelity investigation — full chronology

Starting point: a closed-form ASC/GTD parametric model
(`E_s(f,θ,φ) = Σ A_i (f/fc)^αi · sinc[2π(f/c)·L_i·(sinθ_i - sinθ)] · exp(...)`,
eq. 1 in Wang, Li, Zhang, Ye, Ding, "3-D Attributed Scattering Center Model
of Dynamic Group Targets by Forward Parametric Modeling Method," IEEE TGRS
2024, DOI 10.1109/TGRS.2024.3401241) disagreeing with ray-traced ground
truth for wall-ground dihedral double-bounce returns. The question:
can the disagreement be closed WITHOUT ray tracing — the entire point being
to eliminate the O(rays x facets) cost the closed-form ASC path exists to
avoid.

### 1. Azimuth beamwidth taper — real fix, shipped

A dihedral corner has a diffraction-limited azimuthal acceptance angle
(first null at sin(Δφ) = λ/L), not infinite acceptance. Neither branch of
the code modeled this before — both used a binary geometric include/
exclude test. Implemented `_azimuth_sinc_taper` (multibounce_demo.py) per
the paper's eq. 1. **Measured fix**: isolated double-bounce coherence
0.0024 → 0.178 against ray-traced ground truth. Shipped as the default
behavior in `run_asc_cached_multibounce`.

### 2. Range-direction (wall-height) spread — three attempts, all measured worse

A 20-40m wall face is thousands of wavelengths tall at X-band. Its true
ground-bounce return spreads across many range cells; representing it as
one coherent point is the wrong model once ΔL (measured: up to 201.6m,
thousands of wavelengths) gets that large.

- **Scalar range taper** (`_range_distributed_taper`, mean-wavelength
  sinc): SSIM 0.484 → 0.413, coherence 0.178 → 0.017. Worse.
- **Corrected per-frequency (K,F) taper** (`_range_distributed_taper_kf`):
  SSIM 0.423, coherence 0.017 — nearly identical to the broken scalar
  version. This **falsified the diagnosis**, not just the fix: the
  problem was never "collapsing to a mean wavelength," it's that no
  single-point-plus-amplitude-taper can put energy at multiple distinct
  range pixels, full stop.
- **Multi-point decomposition** (`_ground_bounce_subpoint_contrib`,
  n_sub=1..64 height samples per wall, each independently scored): phase
  RMS improved monotonically (127° → 76°, converged by n_sub~16-32,
  confirming the physics is more correct) but SSIM/coherence got WORSE
  than the single-point baseline (0.484 → 0.44, 0.178 → 0.06).
  Density-weighted sub-point placement (matching the ray tracer's own
  measured hit-density histogram) partially recovered SSIM, not
  coherence.

Both scripts and full numbers: `_taper_ab_test.py`, `_taper_ab_test2.py`.

### 3. Regime-adaptive escalation — built, measured flat, root-caused

Idea: only pay for multi-point decomposition on facets that actually need
it (ΔL > N range-resolution cells, c/2B ≈ 0.25m), leave the rest on the
cheap single-point path. Swept threshold 1-30 cells: **flat, near-zero
change from baseline every time** (escalated 56-63% of facets, SSIM/
coherence barely moved). Root cause, found by inverting the classifier to
decompose ONLY the small-ΔL group: that reproduced the "decompose
everything" bad result almost exactly, proving the damage came from the
SMALL-ΔL group, not the large one — backwards from the model's founding
assumption.

**Why**: the threshold used range-resolution cells (~0.25m, ~8λ), the
wrong physical yardstick. The real criterion for "is single-point-plus-
phase safe to sum coherently" is wavelength-scale (λ/2 ≈ 1.5cm), and even
the smallest measured ΔL (0.019m ≈ 0.64λ) is already past that. By the
rigorous criterion almost nothing in this scene is "safe," yet single-
point still wins empirically — meaning its apparent accuracy likely
reflects agreement with the ray tracer's own per-ray discretization
choices, not more-correct physics. Full numbers: `_taper_ab_test3.py`.

### 4. Is the ray-traced reference itself trustworthy? (order-2: yes; order-3: no)

Pushback worth recording verbatim because it changed the whole framing:
*"dense sbr might not be correct in itself?"* Ran a ray-density
convergence sweep on order-2 (`_diagnose_sbr_convergence.py`): coherence
stayed >=0.9995 across a 9x ray-count increase (60→180 rays/side).
Order-2 is a stable, trustworthy reference at the density used throughout.
But it's not literal continuous-physics ground truth either — its ground-
plane-uniform aim grid maps to a non-uniform (bimodal) wall-height hit
density under oblique viewing, and that pattern is the CONVERGED answer
of that specific discretization, not noise. Two independently-converged
numerical schemes (ground-plane-uniform ray grid vs. hand-designed
sub-point placement) disagreeing on where along a wall's height energy
concentrates is a real, characterized, unresolved discretization mismatch
— not a bug in either one.

### 5. Occluded acceptance angle — literature search, then implementation

Question: the azimuth taper (#1) assumes an unobstructed corner. Real
scenes obscure parts of that acceptance angle. How to handle this without
ray bouncing? Literature search surfaced the Angular Division Algorithm
(ADA/MADA — precomputed aspect-angle visibility tables, no runtime ray
bouncing) and exact-from-polygon-visibility techniques (analytic angular
occlusion via silhouette projection). Before building a new angular-
interval mechanism, found the codebase already has one:
`_segment_occluded_by_facets` in `run_asc_cached_multibounce` — an EXACT
per-leg segment-vs-facet intersection test (same math the ray tracer
uses, applied to 3 analytically-known segments per candidate instead of a
dense exploratory grid), default-on (`occlusion_method='exact'`) and
already baked into every double-bounce number reported above.

### 6. Closed-form triple bounce — built, found real bugs, ended on a genuine reversal

The existing double-bounce code's docstring claimed a ray-tracing-free
triple bounce was structurally unavailable ("which wall a reflected ray
hits next" looked like an unavoidable search problem). **This turned out
to be wrong.** Built `run_asc_cached_order3_wall_ground_wall` +
`_next_surface_hit` (multibounce_demo.py): at each bounce, find whichever
surface — another wall, or the ground — is nearest along the exactly-known
reflected ray direction. One deterministic `ray_facet_intersect` call
plus one closed-form ground-plane test per step, compared by distance.
Covers wall-wall-wall, wall-wall-ground, wall-ground-wall, all of it, with
zero ray bouncing anywhere.

First test (`_test_order3_closed_form.py`, 300m/18-building scene): 7 real
ray-traced order-3 paths vs. 71 closed-form candidates. Root cause found:
the double-bounce code can assume its final leg points straight back at
the sensor because a vertical wall and horizontal ground are ALWAYS
exactly perpendicular — physically exact, not an approximation. A
wall-wall pair has no such guarantee (two buildings can face any relative
direction), and the chain was silently reusing that assumption anyway.
Added an explicit check that the final bounce's true reflected direction
(already known from the law of reflection, not assumed) actually points
back near the sensor. Candidates dropped 71 → 28. Tightening the pointing
tolerance from 5.7° to 1.4° changed nothing — the 28 survivors are
already near-exact geometric configurations, not loose false positives.

Then: **is the "7 real paths" reference itself trustworthy?** Ran the
same convergence sweep used for order-2 in #4, this time on order-3
(`_diagnose_order3_convergence.py`):

```
rays=80  (6,400 aim pts):  order3 paths = 0
rays=100 (10,000 aim pts): order3 paths = 7
rays=130 (16,900 aim pts): order3 paths = 20
```

**Not converged, climbing hard, no sign of leveling off.** Order-3
requires two consecutive near-exact geometric alignments, an extremely
narrow target for a finite ray grid to discover statistically — order-1
and order-2 converged easily because a single alignment is easy to sample
into; two in sequence is not. The "7 real paths" being chased was itself
a severe undercount from under-resolved sampling, not ground truth. This
reverses the framing that anchored the whole session: the closed-form
chain, once the retroreflection-pointing bug was fixed, is plausibly
CLOSER to the true population than the ray-traced reference, not
overcounting relative to it.

## Open problems (honest, unresolved, in priority order)

1. **Order-3 validation needs a reference that isn't ray tracing.** Ray
   tracing cannot practically converge for narrow, multi-stage corner
   returns (see #6 above). The closed-form literature found and NOT yet
   implemented — Griesser & Balanis, "Dihedral corner reflector backscatter
   using higher order reflections and diffractions," IEEE TAP, Nov. 1987
   (validated against MEASURED RCS, not simulation) — is the right
   independent check, extended to trihedral geometry, before trusting
   either the 28-candidate closed-form count or any ray-traced count at
   any practical density.
2. **Range-direction (wall-height) spread has no working closed-form fix.**
   Multi-point decomposition is physically more correct (phase RMS
   improves) but scores worse on SSIM/coherence against the specific
   ray-traced reference used — likely a reference-discretization-matching
   artifact, not a physics failure, but not proven. The literature answer
   not yet implemented: Anastassiu & Kaklamani's closed-form
   polygon-plate-over-lossy-half-space PO solution (Radio Science 2003) —
   an exact boundary-contour integral, no sub-point sampling needed at
   all, which would settle this differently than another taper attempt.
3. **Occluded angular acceptance** (azimuth taper truncated by nearby
   geometry, not just full-path blocking) is handled today via exact
   per-leg segment tests (#5), which work but are evaluated fresh per
   pulse. A precomputed ADA-style angular-interval table (per corner, once
   per scene, independent of pulse) would be cheaper and closer to what
   the literature actually recommends for this — not yet built.
4. Deck task #11 (spell out acronyms on first use) — unrelated, still
   pending from early in the session, listed here so it isn't lost.

## How to reproduce the key numbers

All commands run from `sar_sim/` (Python 3, numpy; add `--gpu` for cupy if
available):

```bash
# Tier 2 single-bounce validation (818x speedup, SSIM>=0.95 numbers)
python3 sbr_vs_asc_compare.py --help   # see validation_tier2_compare.py for the harness

# order-2 azimuth taper A/B
python3 _taper_ab_test.py

# order-2 range-taper / decomposition A/B
python3 _taper_ab_test2.py
python3 _taper_ab_test3.py

# order-2 SBR ray-density convergence (confirms order-2 reference is trustworthy)
python3 _diagnose_sbr_convergence.py

# order-3 closed-form chain vs. ray-traced order-3
python3 _test_order3_closed_form.py

# order-3 SBR ray-density convergence (shows order-3 reference is NOT trustworthy)
python3 _diagnose_order3_convergence.py
```

Rebuild the deck: `cd presentation && node build_deck.js` (writes
`sar_sim_pipeline.pptx`, 38 slides as of this writeup).

## Suggested next session's first task

Implement the Griesser & Balanis closed-form dihedral (then trihedral)
backscatter model as a standalone canonical check — no scene, no ray
tracer, just an isolated corner at varying angle/frequency — and compare
both the azimuth taper (#1) and the order-3 chain (#6) against it
directly. That breaks the "comparing two approximations to each other"
trap this whole investigation kept running into, and is the one piece of
literature surfaced during the session that was never actually built.
