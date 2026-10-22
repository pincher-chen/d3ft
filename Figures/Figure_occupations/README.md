# Occupations versus Fermi–Dirac

Main text, Section 4.2. Al, 3×3×3 mesh (27 k points × 26 bands = 702 states), seed 1101.

| Panel | Content |
|---|---|
| (a) | Shared random initial state (step 0, T = 0.1 Ha): f/u versus (h_ii − μ)/k_BT. |
| (b) | This work at steps 1000 and 4000 against the FD curve. |
| (c) | Both methods at step 4000 against h_ii − μ* (eV), μ* = chemical potential of the accepted state of this work. |

## data/
- `fig2_Al_FD.csv` — one row per state and snapshot: `arm`, `step`, `diagonal_Ha` (h_ii), `mu_Ha`, `scaled_energy`, `normalized_occupation` (f/u), `FD_target`.
- `SOURCES.json` — original path and SHA-256.

## script/
`python script/plot_figure_occupations.py` → `Figure_occupations.pdf/png`; also prints the numbers quoted in the text (QR gap edges, number of fractional states, max FD error).

Collection: **ground-state**. The folder and entry filenames describe the function, independently of manuscript figure numbering.
These are retained ground-state benchmark results; they are not response-design experiment data. The exact earlier tree remains available at tag `v1.0`.
