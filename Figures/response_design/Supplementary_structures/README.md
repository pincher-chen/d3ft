# Supplemental structures: exact three archived MgSiN2 cells

The original state names are start, reverse endpoint and independent FD candidate.
The16 atoms, row-vector cells and fractional coordinates are from the archived
scalar-path protocol. These are not substitutes for the54-input design endpoint.
Mg is green, Si vermilion and N blue; cell edges are black. Periodic Mg-N2.4A and
Si-N2.05A lines denote geometric neighbours, not electronic bond orders. All cells
use a common orthographic viewing direction and field of view; no strain is
exaggerated. All three strained cells are P1. There are no displacement arrows.

Render in a separate native venv with OVITO3.16.1, NumPy, Matplotlib and Pillow:

```bash
export OVITO_PYTHON=/path/to/ovito-venv/bin/python
python script/plot_si_figure_s2.py --output-dir trial_01
```

Actual input cells/sites, style and archived render reference are under data/;
individual render sources and output hashes are generated for each run. The
local plotting helper preserves aspect ratio and common scaling. No Slurm or
calculation GPU is used; Tachyon uses four CPU threads. The renderer's Linux
software-Vulkan fallback is documented inside script/render_structure.py.
