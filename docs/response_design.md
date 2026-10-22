# Response and design workflows

## Source layout

- `d3ft/`: the fixed-occupation response-compatible ground-state solver, radial
  integration, forces/stress, opt-in charge checking and checkpoint support.
- `validation/workflows/response_design/reverse_response/`: GammaResponse,
  KMeshResponse, projected real-inner-product PCG, geometry/phase derivatives,
  compact/streamed k blocks and independent numerical checks.
- `validation/workflows/response_design/joint_design*.py`: the bounded joint
  controller and true electronic reoptimization/force/spectrum backend.
- `validation/workflows/response_design/e9_e10/`: ionic relaxation, force-constrained
  design and postprocessing bands.
- `validation/workflows/response_design/b_external/`: frozen-parameter external
  evaluations, point budgets, immutable outputs and restart policy.
- `validation/workflows/response_design/validation_extensions_20261021/`:
  strain-only/coordinate-only controls, random-direction finite differences,
  isolated-node timing, and distributed response-layout validation.
- `validation/workflows/phase3/`: shared Hamiltonian, residual, spectrum and
  equilibrium-Jacobian helpers imported by those workers.

The original campaign controllers are supplied as source for audit and adaptation.
They retain experiment-stage names and historical allocation information; they
are not a portable automatic queue. Use the explicit runner below, or configure
the archived worker protocol for your own paths/allocation. No command in the
quick start submits, cancels or obtains an allocation.

## Recorded examples

`python scripts/run_response.py --list` lists all campaign scopes.
`reproducibility/response_design/records/` contains the small protocols, summaries,
point readouts, trajectories and acceptance outcomes, including unsuccessful
attempts. `examples/response_design/` supplies actual crystal geometries,
pseudopotentials and settings. `SOURCE_TRANSFORMATIONS.json` distinguishes original
source digests from copies with neutral namespaces and relocatable metadata.
Numerical values and binary scientific arrays were not fitted or changed during
packaging.

### Ready-to-run saved-state Gamma example

A real, accepted MgSiN2 parent and its independent eigenspace are included in
`reproducibility/response_design/seeds/MgSiN2_gamma/`. It has16 atoms,96 electrons,
a144^3 grid and100-Ha cutoff, with SG15 Mg/Si/N pseudopotentials. Its48 occupied
orbitals and a fixed ordered Miller mask are retained. The gap is4.410987370941647eV;
the stored stationarity residual is4.598864265099009e-9Ha.

The runner makes owned copies and rebinds only machine paths and metadata hashes.
It checks the UPF/state/eigenspace digests before executing. All input arrays remain
unchanged. No pickle from an unrelated source is needed.

```bash
# Metadata inspection only; no JAX/device initialization:
python scripts/run_response.py --stage inspect

# Run on your existing GPU allocation, with your GPU selection:
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage reverse --execute --output runs/reverse54
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage forward --axes 0 1 2 3 4 5 --execute --output runs/forward6
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage spectrum --execute --output runs/independent_spectrum

# All54 forward tangents: omit --axes. A fresh run directory is mandatory.
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage forward --execute --output runs/forward54

# Joint, strain-only and coordinate-only controls use the same target and bounds:
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design54 --execute --output runs/design54
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design6 --execute --output runs/design6
CUDA_VISIBLE_DEVICES=0 python scripts/run_response.py --stage design48 --execute --output runs/design48
```

The target is4.510987370941646eV. The strain-step L2 limit is0.003 and cumulative
strain limit0.03; coordinate steps and cumulative bounds are0.02 and0.15Bohr.
The combined controller uses a scaled trust-region norm; separate component
maxima are not an alternative definition of that constraint. Joint and
coordinate-only demonstrations stop after3 updates; strain-only allows5 and
stops once the unchanged2-meV target criterion is met. Finite residual forces
must not be described as ionic equilibrium.

For an existing Slurm allocation, supply your allocation and device list:

```bash
D3FT_PYTHON=python bash scripts/slurm_response.sh YOUR_JOBID 0 reverse --execute --output runs/reverse54
```

This real example can need hours and approximately tens of GiB of GPU memory.
The54-tangent forward run takes substantially longer than the adjoint. CPU tests
are deliberately small mathematical/interface checks and are not substituted for
these production calculations.

### Extended campaigns

The k-mesh, calibration, external-material and band-path workers are included
with recorded protocols and outcomes. They require their own regenerated
ground states or checkpoints; the Gamma example is the only complete restart
seed shipped here. The large k4 orbital checkpoints and every historical
checkpoint are not distributed. Reconstruct them from the archived physical
settings and geometry before using a response worker; do not replace a fixed
basis, occupation convention, tolerance or band-edge branch silently.

For sampled response, public kernel classes are `KMeshResponse` and
`sharded_kmesh_response.py`; `mesh_target.py` explicitly rejects unresolved hard
band-edge branches. Shared density and the2/Nk occupation weight are part of the
model. Geometry vectors always put6 engineering strains before Cartesian
coordinates. The distributed layout implementation and its validation are
included separately from the single-device streamed route.

## Interpretation of the archived outcomes

- Three strain-only updates also reach the joint-design target; coordinate-only
  updates do not reach it in three steps. This does not establish the necessity
  of internal coordinates for that particular target.
- The sampled k2 design meets its design criterion but misses the k4 transfer
  criterion:1.125466meV absolute error against a1-meV criterion. The target shift
  is20meV; a failed transfer criterion remains a failed transfer criterion.
- The hydrostatic deformation potential uses k2 sampling, without demonstrated
  k convergence. Literature definitions are not numerical acceptance tests.
- All three held-out calibration losses are worse than zero-parameter PBE.
  Frozen-parameter external lattice errors improve for the four reported
  compounds; this is limited external transfer, not a universal generalization
  claim. The held-out Si fine bulk-modulus criterion failed and remains recorded.
  GaP uses a local finite-temperature reference proxy and its B is not certified.
- Numerical acceptance, target acceptance, timeout/infrastructure interruption
  and budget pauses are separate outcomes. No historical failed record is
  overwritten with a later accepted one.

## Figures

`Figures/response_design/Figure_1/` contains only the final hand-edited PDF and a
README. No obsolete drawing source is presented as its generator. Figures2–5
retain their actual numeric inputs, scripts, published source hashes and PDFs.
The supplemental structure panel has the true three archived cells and an OVITO
CPU renderer. Geometric neighbour lines are not electronic bond orders.

Use Python3.12/Matplotlib3.10.8 for the standard panels. The Sumo/pymatgen bands
use the separate recorded Python3.13 plotting environment; see
`requirements-response-bands.txt`. For OVITO install the optional module in its
own native venv and point `OVITO_PYTHON` to it. No plotting command launches DFT.
