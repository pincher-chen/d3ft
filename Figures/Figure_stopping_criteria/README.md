# Variance stopping criteria and independent acceptance checks

Main text, Section 4.3. Same runs as Figure 1 (Γ, seed 1101).

| Panel | Content |
|---|---|
| (a) | Variance of the free energy over the last 50 steps; dotted line = JRystal threshold 1e-8 Ha²; dashed verticals = first crossings. Values < 1e-25 drawn at 1e-25. |
| (b) | The eleven acceptance tests, each divided by its tolerance, at the first variance crossing (open) and at the end of each run (filled). Below 1 = pass. |

## data/
- `fig3_Al_stopping_criteria.csv` — rows of `kind` = `variance` (`step`, `variance_F_Ha2`) and test profiles (`kind` ∈ {variance hit, 4k pass, 6k censored}; `test`, `ratio`). Only `mesh == Gamma` rows are plotted.
- `SOURCES.json`.

## script/
`python script/plot_figure_stopping_criteria.py` → `Figure_stopping_criteria.pdf/png`.

Collection: **ground-state**. The folder and entry filenames describe the function, independently of manuscript figure numbering.
These are retained ground-state benchmark results; they are not response-design experiment data. The exact earlier tree remains available at tag `v1.0`.
