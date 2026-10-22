# Figures by scientific function

Folders are named by scientific purpose, independently of manuscript numbering.
Response-design and earlier ground-state figures coexist here. The collection column
identifies the source; ground-state figures are not reused as response-design evidence.
The unmodified earlier release remains at [`v1.0`](https://github.com/pincher-chen/d3ft/tree/v1.0).

| Folder | Collection | Contents |
|---|---|---|
| [Figure_response_architecture](Figure_response_architecture/) | response-design | Electronic response and design architecture |
| [Figure_response_scaling](Figure_response_scaling/) | response-design | Derivative agreement and input-count cost scaling |
| [Figure_joint_design](Figure_joint_design/) | response-design | Joint strain and internal-coordinate gap design |
| [Figure_bands_and_transfer](Figure_bands_and_transfer/) | response-design | Electronic bands, sampled-gap transfer and deformation response |
| [Figure_frozen_calibration](Figure_frozen_calibration/) | response-design | Frozen functional calibration and external evaluation |
| [Figure_mgsin2_structures](Figure_mgsin2_structures/) | response-design | Archived MgSiN2 start and endpoint structures |
| [Figure_self_diagonalization](Figure_self_diagonalization/) | ground-state | Hamiltonian self-diagonalization and commutator |
| [Figure_occupations](Figure_occupations/) | ground-state | Occupations versus Fermi–Dirac |
| [Figure_stopping_criteria](Figure_stopping_criteria/) | ground-state | Variance stopping criteria and independent acceptance checks |
| [Figure_ground_state_bands](Figure_ground_state_bands/) | ground-state | Five-system posterior band comparison with Quantum ESPRESSO |
| [Figure_metric_components](Figure_metric_components/) | ground-state | GaAs metric components and independent seeds |
| [Figure_solver_cost](Figure_solver_cost/) | ground-state | Cost per update and occupation-layer scaling |
| [Figure_sodium_bands](Figure_sodium_bands/) | ground-state | Sodium band comparison |

Each package has a README. Numerical packages retain `data/`, source hashes and
`script/`; their final PDF and plotting entry use the same functional name.
Raw data filenames are preserved so archived numerical provenance remains traceable.

**Figure_response_architecture is manually edited.** It contains only the authoritative
PDF and README; no obsolete generator, SVG or source data is supplied.

Replot response panels into a new output directory (separate band environment):

```bash
python scripts/response_figures.py --output-dir runs/response_figures --band-python /path/to/band-venv/bin/python
```

Earlier ground-state figures can be rendered with `python make_figures.py --collection ground-state`
in an environment satisfying `requirements-figures.txt`. This writes into a separate
output folder. OVITO structures have their own renderer instructions and are optional.
`catalogue.json` is the machine-readable index. Historical release/provenance files
may cite the former numbered paths; those are snapshot references, not current entrypoints.
