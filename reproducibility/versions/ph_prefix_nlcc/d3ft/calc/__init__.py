"""Top-level calculation orchestrators.

These mirror the structure of :mod:`jrystal.calc` but wire D3FT's three new modules
into the optimisation loop in place of Jrystal's defaults.

* :mod:`d3ft.calc.ground_state` — Ground-state energy minimisation (Module 1 + 2).
* :mod:`d3ft.calc.band_structure` — Band structure along a k-path (Module 3).
"""
from . import band_structure, bandgap, design, ground_state, stress

__all__ = ["band_structure", "ground_state"]
