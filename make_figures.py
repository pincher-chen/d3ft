"""Regenerate all figures. Each figure is self-contained in Figures/Figure_N/ (data/, script/)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in sorted(ROOT.glob("Figures/Figure_*/script/plot_figure_*.py")):
    print("running", script.relative_to(ROOT))
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
