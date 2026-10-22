# Sodium band comparison

Na, 2-atom bcc conventional cell, corrected nonlinear-core-correction quadrature.

| Panel | Content |
|---|---|
| (a) | Bands of this work (lines) and QE (circles), energies relative to the lowest QE state at Γ. |
| (b) | Difference after global alignment (µeV). |

## data/
- `band_comparison.csv` — `point`, `band`, `distance_inverse_A`, `candidate_Ha`, `QE_Ha`, `aligned_delta_eV`; path points 0/12/24/36 = Γ/X/R/Γ.
- `SOURCES.json`.

## script/
`python script/plot_figure_sodium_bands.py` → `Figure_sodium_bands.pdf/png`.

Collection: **ground-state**. The folder and entry filenames describe the function, independently of manuscript figure numbering.
These are retained ground-state benchmark results; they are not response-design experiment data. The exact earlier tree remains available at tag `v1.0`.
