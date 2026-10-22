# D3FT: electronic response and structural design

This branch, **response-design**, adds self-consistent electronic adjoints,
forward controls and many-input structural design to direct plane-wave DFT.
The existing **v1.0** tag preserves the earlier ground-state benchmark release.
The new source and records are separate; no earlier tag is replaced.

## Install the scientific runtime

Recorded solver environment: Python3.10, JAX/jaxlib0.6.2, double precision.

```bash
git clone --branch response-design https://github.com/pincher-chen/d3ft.git
cd d3ft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e third_party/jrystal_dependency -e .
python scripts/verify_response_release.py
python scripts/check_response_cpu.py
python scripts/run_response.py --stage inspect
```

Install a compatible CUDA-enabled JAX build separately for real GPU runs;
`jax[cuda12]==0.6.2` is the recorded version. Inspection and CPU checks do not
request an allocation. Optional FD/particle-hole trial modules remain optional;
they are not the solver used by the fixed-occupation response examples.

## Run the real saved-state example

```bash
# Read-only catalogue and settings; no GPU initialization:
python scripts/run_response.py --list
python scripts/run_response.py --stage reverse

# Explicit opt-in on a GPU you already have; NEW output folder required:
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage reverse --execute --output runs/reverse54
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage forward --execute --output runs/forward54
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design54 --execute --output runs/joint54
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design6 --execute --output runs/strain6
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design48 --execute --output runs/coordinates48
```

The bundled MgSiN2 Gamma checkpoint, fixed Miller mask, density and independent
orbital eigenspace are real accepted calculation outputs. These arrays are
hash-verified and copied into each new experiment directory. Full calculations
are substantial; metadata inspection is the default. See
[workflow documentation](docs/response_design.md) for units, numerical scope,
acceptance, checkpoint handling and limitations.

## What is distributed

| Location | Contents |
|---|---|
| `d3ft/` | Fixed-occupation response-compatible solver, forces/stress, radial kernels and checkpoint support |
| `validation/workflows/response_design/` | Gamma/k-mesh response, design, ionic-relaxation, fitting, spectra and verification source |
| `examples/response_design/` | Geometries, pseudopotentials, case catalogue and bounded-design settings |
| `reproducibility/response_design/` | Scientific protocols, accepted/unsuccessful outcomes, CPU checks and the representative Gamma restart seed |
| `Figures/response_design/` | Latest figures; numeric source data and plotting code for Figures2–5 |
| `provenance/RESPONSE_DESIGN_MANIFEST.json` | Distributed-source/data SHA256 inventory |

The extended workers and campaign records are included, but not every large
historical k-mesh checkpoint. Only the Gamma example is a complete bundled
restart. Extended campaigns require regenerated states and user-configured
paths/hardware; old campaign controllers are archival sources, not a ready-made
queue for somebody else's allocation.

## Figures

**Figure1 is a manually edited architecture diagram. Only its final PDF and a
README are supplied. No obsolete script, draft image or data is claimed to
regenerate it.** Figures2–5 have their own data, scripts and source manifests.
The supplemental structure rendering includes the exact three cells and an
optional OVITO CPU renderer.

Standard panels use Python3.12/Matplotlib3.10.8. Band plotting uses a separate
Python3.13 environment with `requirements-response-bands.txt`.

```bash
python scripts/response_figures.py --band-python /path/to/band-venv/bin/python
```

Original outcome values, failed criteria and infrastructure interruptions remain
separate. In particular, all held-out calibration losses are worse than PBE;
limited external improvements do not imply universal generalization, and the
sampled-gap transfer criterion was not met. See the workflow documentation and
stored acceptance records.

## Licence and attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The direct-minimization
infrastructure follows JRystal; the exact production dependency is vendored under
`third_party/jrystal_dependency/` with its attribution. UPF files retain their
original headers. Earlier solver benchmarks and their frozen versions remain
available through **v1.0**.
