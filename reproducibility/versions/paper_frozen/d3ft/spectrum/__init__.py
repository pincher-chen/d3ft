"""D3FT-Lanczos: differentiable stochastic Lanczos for band structures.

This subpackage implements Module 3 of D3FT v0.1.

Public API:
    - :func:`lanczos.lanczos_tridiag`         — build a tridiagonal from HVPs
    - :func:`lanczos.lanczos_eigvals`         — extract lowest-k eigenvalues
    - :func:`implicit.eigvals_with_grad`      — eigenvalues + implicit gradients
"""
from . import implicit, lanczos

__all__ = ["implicit", "lanczos"]
