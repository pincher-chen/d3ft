"""D3FT-OT: Sinkhorn-smooth occupations.

This subpackage implements Module 2 of D3FT v0.1.

Public API:
    - :func:`sinkhorn.sinkhorn_occupation`
    - :func:`sinkhorn.epsilon_schedule`
"""
from . import sinkhorn

__all__ = ["sinkhorn"]
