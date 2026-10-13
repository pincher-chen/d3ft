# D3FT

Direct free-energy minimization in plane-wave density-functional theory, with charge-conserving Fermi–Dirac coordinates and gap-scaled orbital rotation preconditioning. Built on JRystal's plane-wave infrastructure. D3FT is unrelated to Grimme's DFT-D3 dispersion correction.

This release brings together the executable solver, small verification tests, six physical benchmark cases, exact reference data, and self-contained paper figures. Existing experiment directories and manuscript sources were not modified during packaging.

## Quick start

Python 3.10 is the recorded research environment. CPU verification and figure rendering need no allocation.

```bash
git clone git@github.com:pincher-chen/d3ft.git
cd d3ft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-figures.txt
pip install --no-deps -e third_party/jrystal_dependency
pip install --no-deps -e .
make check
python scripts/run_tests.py --suite cpu
python make_figures.py
```

GPU calculations additionally require a CUDA-compatible JAX installation; the recorded runs used JAX/jaxlib 0.6.2 with the matching CUDA12 plugin on NVIDIA H100. For pip-managed CUDA12, install `jax[cuda12]==0.6.2`. Preserve the JAX version when comparing trajectories. Original environment and hardware records are under `provenance/environment/`.

## Which code version is used?

| Location | Role |
|---|---|
| `d3ft/` | Latest validated corrected-phase solver, including the Na NLCC physical-kernel repair; copied without scientific edits |
| `reproducibility/versions/paper_frozen/` | Original corrected-phase solver used for Al, Si, SiC, displaced SiC, and GaAs validation |
| `reproducibility/versions/ph_prefix/` | Original PH annealing prefix; retained exactly |
| `reproducibility/versions/ph_prefix_nlcc/` | Same prefix with the validated Na NLCC kernel |
| `archives/development_20261013/` | Original current development package from `chenp/d3ft2/d3ft-main`, including legacy Sinkhorn and response features |
| `reproducibility/implementation/` | Optional trial-only gather optimization and equivalence evidence; separate from the validation solver |
| `third_party/jrystal_dependency/` | Exact JRystal runtime dependency used by the paper |

**Use `scripts/run_case.py` for paper calculations.** The generic CLI and `config.yaml` retain legacy defaults; they do not select the paper's two-stage method automatically. The paper method uses an analytic metric direction and Armijo; its experiments should not be described as Adam-moment optimization. Recycling remains an implementation option, not a demonstrated contribution under the final schedule. No superiority of the PH factor over the occupation-difference factor is asserted here.

## Benchmark cases

```bash
python scripts/run_case.py GaAs --dry-run
# Requires one GPU; creates a NEW output directory, never overwrites:
python scripts/run_case.py GaAs --seed 301 --output runs/GaAs_s301
# Existing allocation, supplied by the user; never cancels it:
D3FT_PYTHON=python bash scripts/slurm_case.sh YOUR_JOBID GaAs 301 runs/GaAs_s301
```

The runner preserves the 1000-step anneal-then-switch prefix, occupation scales 1/10, corrected estimator, δ=0.001 Ha, stable internal algebra, Armijo and recycling; independent spectral and QE audits occur at 4000/8000/12000/16000/20000 steps. It stops at the first strict 8+3 pass. Failure at the budget remains censored. QE density, mesh and energy checks retain their original conventions and tolerances.

See [examples/README.md](examples/README.md) for all six cases, units and settings; [reproducibility/README.md](reproducibility/README.md) for protocol, reference generation and limitations.

## Figures and data

[Figures/README.md](Figures/README.md) indexes Figure1–6 and FigureS1. Every folder contains `data/`, `script/`, README, PDF and PNG. Scripts read only their local data and require no DFT software. Figure4 contains the five-system independent high-symmetry-path comparison against QE. Sources and SHA256 are recorded in each data directory.

```bash
python Figures/Figure_4/script/plot_figure_4.py
# Or all figures:
make figures
```

## Tests and provenance

[tests/README.md](tests/README.md) distinguishes CPU mathematics, legacy tests, GPU integration, and external-reference checks. Original packaging failures and final verification results are retained under `provenance/checks/`. Passing unit tests is not a substitute for the 11 physical acceptance tests.

`make check` verifies immutable code/data against `provenance/RELEASE_MANIFEST.json`, exact solver-version hashes and reference hashes. Rendered figure files are derived outputs and are excluded from the immutable hash set so rerendering PDFs does not invalidate source checks. `provenance/COPY_SOURCES.json` records original local source paths and original digests; these paths are provenance, not runtime dependencies.

## License and attribution

D3FT source: Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Redistributed JRystal source retains its headers and license. Pseudopotentials retain their original UPF metadata and provenance; see `reproducibility/inputs/pseudopotentials/README.md`. Historical baseline code and the shared NC control are distinct from the current solver. The original AE JRystal implementation is not a Hamiltonian-matched accuracy baseline.
