# Tests

Run `python scripts/run_tests.py --suite smoke` or `--suite cpu` from an active environment. The runner explicitly uses CPU JAX and the bundled JRystal package; it does not access Slurm or GPUs. Each executable test has a separate log and a JSON status record under `runs/`.

- Paper coordinates: `test_paper_coordinates.py` checks FD electron number, Pauli bound, gauge invariance, implicit derivatives, total-capacity boundary rejection, tiny-positive-occupation Rayleigh reconstruction, and preservation of builtin `float` as a NumPy dtype.
- Geometry/optimization: Stiefel tangent/retraction, descent direction and kinetic preconditioning.
- Operators/spectrum: Lanczos, derivatives, channel/projector layout, k-pool sharding, XC, density-grid guards.
- Response/workflow arithmetic: elastic reductions, implicit chains, uncertainty and phase3 guards. These are legacy feature tests, not claims in the JCTC paper.
- GPU integration and legacy AlP/QE parity scripts are retained for opt-in execution and need appropriate data/hardware; they are not run during packaging.

Version separation is explicit: `test_12_occupation_charge.py` checks the archived **development Sinkhorn** implementation, not the FD paper solver. The FD test runs the current validated solver. The old frozen Sinkhorn has extreme-scale gradient failures and is retained for exact provenance; it is not used by the FD paper protocol. The runner records the source scope per test rather than claiming interchangeable versions.

Initial packaging checks exposed four failures. Their logs are preserved in `provenance/checks/initial_cpu/`: legacy Sinkhorn test/source mismatch, an outdated synthetic fixture lacking cell/config fields, a missing `materials.json`, and a two-device sharding test run with one CPU device. Packaging restores the missing JSON, updates only that test fixture (original in archives), sets two virtual CPU devices for sharding, and pairs Sinkhorn with its actual development source. No frozen solver file or scientific threshold was changed.

A new physics run is tested separately with `scripts/run_case.py`: cold initialization → anneal → corrected phase → independent streamed eigsh → all 8 internal + 3 QE acceptance tests. Unit tests passing alone does not certify physical convergence.
