# Figures of the main text and Supporting Information

One folder per figure:

    Figure_N/
      data/       input data (copies; originals are not modified) + SOURCES.json (original path, SHA-256)
      script/     plot_figure_N.py — standalone, reads only ../data, writes ../Figure_N.pdf and ../Figure_N.png
      Figure_N.pdf, Figure_N.png
      README.md   panels, data columns, how to run

| Folder | Paper | Content |
|---|---|---|
| Figure_1 | Fig. 1 | Self-diagonalization of h, Al Γ |
| Figure_2 | Fig. 2 | Occupations vs Fermi–Dirac, Al 3³ |
| Figure_3 | Fig. 3 | Variance stopping criterion vs acceptance tests |
| Figure_4 | Fig. 4 | Five-system posterior bands vs QE and aligned error distributions |
| Figure_5 | Fig. 5 | GaAs metric components and independent seeds |
| Figure_6 | Fig. 6 | Cost per update and occupation-layer scaling |
| Figure_S1 | Fig. S1 | Na band structure |

Regenerate all: `python make_figures.py` (from the repository root) or run each script.
Python: use an environment with numpy, pandas, matplotlib, scipy (see requirements-figures.txt).
Style shared by all scripts: Liberation Sans, 8 pt labels/titles, 7 pt ticks, axis line width 0.8, inward ticks.

main.tex and supporting_information.tex include `Figures/Figure_N/Figure_N.pdf` directly.
The old single-file `make_figures.py` is kept in `history/make_figures_monolithic_20261012.py`; `figs/` and `figs_data/` are kept as earlier outputs and are no longer used.
