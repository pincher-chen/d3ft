"""D3FT — Direct Differentiable Density Functional Theory.

D3FT extends ``jrystal`` with three new algorithmic modules:

* :mod:`d3ft.optim`        — Kinetic-preconditioned Riemannian Adam on Stiefel manifolds.
* :mod:`d3ft.occupation`   — Capacity-bounded Sinkhorn-OT occupations.
* :mod:`d3ft.spectrum`     — Differentiable stochastic Lanczos for band structures.

The orchestration layer (:mod:`d3ft.calc`) wires these modules together with Jrystal's
plane-wave primitives to produce ground-state and band-structure calculations.
"""

__version__ = "0.1.0"

from . import calc, config, occupation, optim, spectrum, utils

__all__ = [
    "calc",
    "config",
    "occupation",
    "optim",
    "spectrum",
    "utils",
]
