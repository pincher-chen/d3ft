"""D3FT-Riem: Riemannian optimization on the Stiefel manifold.

This subpackage implements Module 1 of D3FT v0.1.

Public API:
    - :func:`stiefel.project_tangent`, :func:`stiefel.retract_qr`, :func:`stiefel.retract_cayley`
    - :func:`preconditioner.kinetic_preconditioner`
    - :func:`riemannian_adam.riemannian_adam` (returns an ``optax.GradientTransformation``)
"""
from . import preconditioner, riemannian_adam, stiefel

__all__ = ["preconditioner", "riemannian_adam", "stiefel"]
