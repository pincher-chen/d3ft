# Reproduction workflow

## Solver identity

Scientific source files are byte-for-byte copies of verified experiment versions. `scripts/runtime.py` checks the selected manifest before running any driver and sets a local `PYTHONPATH` to the selected version plus the exact bundled JRystal dependency. It does not overwrite installed packages or edit snapshots.

| Version | Original tree SHA256 |
|---|---|
| corrected, non-NLCC | `b0839aedc0cf4cd4d6ea8cbad713dea46c6313f68515e4edb394930a96685fa4` |
| corrected, NLCC repaired | `b211079939f7c30a974558815de063c986e75b6ae200f3778352afd39b3371d8` |
| PH prefix | `a45b508a69d39a9f02c248ec3511e1c9e4c381e8385a2d2cc7fc380d9ddd656a` |
| PH prefix, NLCC repaired | `8698472f95d42c4e040b3319a203bdc4c2a0821e750d82dc5c63c0d5b76a1ad5` |

The driver wrappers only relocate inputs, outputs, references and module search paths. `provenance/COPY_SOURCES.json` retains original driver hashes; `provenance/RELEASE_MANIFEST.json` hashes portable copies. The original experimental directories are untouched.

## Cold-start protocol

`inputs/final_method.json` records all algorithm parameters and numerical edge cases. The portable runner uses the original random QR orbitals and occupations (keys seed and seed+10000), 1000 updates annealing 0.1→0.01 Ha under the PH prefix, then corrected updates at target temperature; FD coordinate temperature remains 0.1 Ha. Optimizer/trial state resets at the same stage boundaries. No convergence gate, parameter selection, density fitting or post-hoc tolerance relaxation is added.

Internal checks: charge ≤1e-10 e, orbital orthogonality ≤1e-10, external orbital residual ≤1e-6 Ha, commutator ≤1e-6 Ha, FD occupation error ≤1e-6, Ritz residual ≤1e-8 Ha, one-step spectral density change/Ne ≤1e-6, spectral free-energy change ≤1e-8 Ha. QE checks: |ΔF|≤1e-4 Ha/cell, globally aligned mesh-band MAE≤1e-3 eV, electron-measure density L1/Ne≤2e-4. Read the original scripts and protocol JSON for exact definitions.

Al/Si/SiC/displaced-SiC/GaAs use the original non-NLCC frozen solver. Na uses the same optimizer with the independently validated NLCC correction in both phases. Only Na among these six UPF combinations contains NLCC; the scope table and fixed-density XC check are in `analysis/NLCC_scope_20261010/`. New Na seeds 13101–13105 passed; original mismatch and infrastructure failures remain in the source archives.

## QE reference

`references/<case>/` contains unchanged `summary.json`, `spectrum.npz`, `rho.cube`, and available original SCF/NSCF/XML inputs. The first three are hash-checked at comparison time. These compact reference products suffice for the 3 external checks; large QE wavefunction/save directories are not in Git.

To regenerate the original SCF with QE7.5 (same physical input, relocated filesystem paths only):

```bash
python scripts/run_qe.py Si --pw /path/to/pw.x --output runs/QE_Si --dry-run
python scripts/run_qe.py Si --pw /path/to/pw.x --output runs/QE_Si --mpi mpirun -np 16
```

`--mpi` consumes remaining arguments, so put it last. Cutoffs in QE inputs are Ry, twice the solver's Ha. Density cutoffs, smearing, k offsets, FFT dimensions and band counts stay exactly as archived. For full reference regeneration, execute the archived NSCF and pp.x input stages after the SCF using the same relocated save directory. The baseline Na NSCF required CG recovery for degeneracies; its accepted reference spectrum is included, and the 2026-10-13 path data document that eigensolver change separately.

## Figures and implementation evidence

CPU figure reproduction uses only `Figures/Figure_N/data/`. Numerical comparison tables, NLCC scope, Si8/16/32 supercell summaries and implementation equivalence evidence are under `analysis/`.

Optional `implementation/production_candidate.py` retains the original PH-prefix executable and substitutes gather only for corrected-phase Armijo trial coefficient expansion. It is an implementation optimization of the same algorithm, not a new metric. The operator/objective/physical-endpoint evidence and original stepwise failures are retained in `analysis/implementation_chain_completed_20261010/`; do not infer bitwise identical full trajectories at numerical stationary points. Validation results remain attributed to the frozen implementation.

Raw checkpoints, full Slurm logs, failures, reference save files and historical development campaigns are intentionally not duplicated in this compact Git repository. Their original paths/hashes are recorded; the cold-start pipeline regenerates scientific states. No Slurm job is started by install, tests, checks or figure rendering. Only explicit `run_case.py` / `slurm_case.sh` invocations perform new calculations.

## Independent high-symmetry path spectra

After a cold reproduction passes, its final state can be used with the same 40-band independent eigsh/Ritz procedure as Figure4:

```bash
python scripts/runtime.py paper_frozen reproducibility/scripts/band_path.py --case Si --state runs/Si_s301/results/YOUR_ACCEPTED_STATE --output runs/Si_path
# For Na use version current (validated NLCC correction).
```

The script keeps the saved density fixed, reads the local Figure4 path coordinates, uses tol=1e-11/ncv=120/maxiter=3000, and rejects unresolved Ritz states or an insufficient high-band buffer. It records all Ritz residuals and physical/source metadata per point. Optional --lane/--lanes partitions points across independent visible GPUs; no Slurm submission occurs. The exact existing QE path inputs and spectra are included below references; to recompute those NSCFs, start with the same regenerated SCF save, use its unchanged density and archived path input.
