# Physical test cases

These are executable reproduction cases, not newly tuned examples. Exact cells and atomic coordinates are in `reproducibility/inputs/structures/`; controls and audit counts are in `cases.json`. All cases are spin-restricted, use equal weights on full Γ-centered k meshes, and use a physical temperature of 0.01 Ha. The FD coordinate temperature is 0.1 Ha.

| Case | Atoms | Electrons | k mesh | Cutoff (Ha) | FFT | Compared bands | XC | Recorded seed |
|---|---:|---:|---|---:|---|---:|---|---:|
| Al | 4 | 12 | 2³ | 30 | 64³ | 14 | LDA_X + PW | 301 |
| Si | 2 | 8 | 2³ | 40 | 64³ | 12 | PBE | 301 |
| SiC | 2 | 8 | 3³ | 50 | 64³ | 12 | PBE | 301 |
| SiC_distorted | 2 | 8 | 3³ | 50 | 64³ | 12 | PBE | 301 |
| GaAs | 2 | 18 | 3³ | 100 | 90³ | 17 | SG15/PBE, Ga3d included | 301 |
| Na | 2 | 2 | 2³ | 30 | 64³ | 9 | LDA_X + PW, NLCC repaired | 13101 |

Al and Na use conventional cubic fcc/bcc cells at a=4.05/4.23 Å. Si, SiC and GaAs use two-atom fcc primitive cells with a=5.43095784, 4.38133, 5.749232952198 Å. Displaced SiC has the same cell and C shifted by 0.02 Å along x. Exact structure bytes supersede rounded descriptions.

Inspect without computation: `python scripts/run_case.py SiC_distorted --dry-run`. Run with one visible GPU: `python scripts/run_case.py Si --seed 301 --output runs/Si_s301`. Output must not exist. Selecting another seed does not change any physical or algorithmic parameter.

Saved small endpoint records are under `reproducibility/records/<case>/`: config, original source provenance, summary, 8 internal checks, independent spectral audit and original task metadata (including all original stages). Complete wavefunction checkpoints and Ritz vectors remain in the original experimental archive and are not included in Git; this runner regenerates them from cold start. Compact eigenvalues/occupations are included. Original absolute paths in records document provenance and need not exist on a user's machine.

Independent audits compare more bands than the Table1 band count. Gap values are finite-temperature, no-SOC, discrete-mesh KS gaps, not experimental or continuum-BZ gaps. High-symmetry path data and path definitions are included with Figure4.
