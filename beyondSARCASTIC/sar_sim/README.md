# sar_sim

Small-scale, GPU-friendly SAR simulation: hand-authored scenes (ground +
buildings) -> ray-traced multi-bounce reflectivity (dense SBR) and a
closed-form ASC-cached alternative -> phase history -> backprojection image
formation. No Blender/Unity/Unreal -- geometry is authored directly as
numpy/cupy arrays.

**This has grown well past the original Tier 1 MVP** described in
`docs/pipeline_outline.md` (still useful background, now historical) --
single-bounce is validated (818x speedup, SSIM>=0.95/building on a 200-
building city scene), and there's a substantial, still-open investigation
into closed-form multi-bounce (wall-ground double bounce, wall-ground-wall
triple bounce) fidelity against ray-traced ground truth.

**Start at `../PROJECT_SUMMARY.md`** for the full picture: what's
validated, what's open, the full chronology of what was tried on the
multi-bounce problem and what was actually measured, and how to reproduce
every number. This file only orients you inside `sar_sim/` itself.

## Status

| Module | Status |
|---|---|
| `materials.py` | done -- Fresnel reflectivity + diffuse coefficient per material |
| `scene.py` / `dense_sbr_demo.py` | done -- scene authoring + ray-traced multi-bounce SBR |
| `sbr_vs_asc_compare.py` | done -- single-bounce closed-form ASC path + Tier 2 validation |
| `multibounce_demo.py` | done, still open -- multi-bounce ray tracer + closed-form double/triple bounce, see PROJECT_SUMMARY.md |
| `validation_tier2_compare.py`, `validation_multibounce_compare.py` | done -- SSIM/coherence scoring harnesses |

## Backend note

Everything is written in plain NumPy so it runs anywhere, including this
sandbox (which can't reach `download.pytorch.org` or pull the full CUDA torch
wheel in a reasonable time). The array ops are kept close to what `torch`
supports 1:1 -- porting a module to GPU is generally:

```python
# import numpy as np
import torch as np  # ...plus moving arrays to .to('cuda') and swapping
                     # np.asarray -> torch.as_tensor, etc.
```

Do this per-module once you're running on your own machine with `torch`
installed and a real GPU available; `raytrace.py` (the O(N_pulses x N_rays x
N_triangles) ray-triangle intersection) is the module that actually benefits
from it.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

(once `run_demo.py` exists -- see Status above)
