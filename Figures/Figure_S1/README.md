# Figure S1 — Na band structure along Γ–X–R–Γ (Supporting Information, Section S7)

Na, 2-atom bcc conventional cell, corrected nonlinear-core-correction quadrature.

| Panel | Content |
|---|---|
| (a) | Bands of this work (lines) and QE (circles), energies relative to the lowest QE state at Γ. |
| (b) | Difference after global alignment (µeV). |

## data/
- `band_comparison.csv` — `point`, `band`, `distance_inverse_A`, `candidate_Ha`, `QE_Ha`, `aligned_delta_eV`; path points 0/12/24/36 = Γ/X/R/Γ.
- `SOURCES.json`.

## script/
`python script/plot_figure_s1.py` → `Figure_S1.pdf/png`.
