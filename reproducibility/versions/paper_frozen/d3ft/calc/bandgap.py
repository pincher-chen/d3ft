"""Differentiable band edges.

`E_gap = min(conduction) - max(valence)` is not differentiable. `min` and `max`
are non-smooth wherever two states cross, and their autodiff gradient at a
crossing is whichever branch the tie-break happened to pick -- computed silently,
with no indication that it is meaningless. AlP switches its CBM between X and
Gamma under exactly the strain an inverse-design loop applies, so this is not a
corner case for that workflow but its operating point.

The replacement here is an occupation-gated soft extremum,

.. math::
    \\mathrm{VBM}_\\beta = \\frac{1}{\\beta}
        \\log \\sum_{nk} f_{nk}\\, e^{\\beta \\varepsilon_{nk}},
    \\qquad
    \\mathrm{CBM}_\\beta = -\\frac{1}{\\beta}
        \\log \\sum_{nk} (1 - f_{nk})\\, e^{-\\beta \\varepsilon_{nk}}

which is smooth in both the eigenvalues and the occupations, and tends to the
hard edges as ``beta -> inf``. Gating on the occupations matters: taking a soft
max over "bands below index n_occ" would just move the discontinuity from the
eigenvalues to the band index, which also jumps at a crossing. D3FT's occupations
are already smooth functions of the parameters (Sinkhorn / Fermi-Dirac), so
weighting by them keeps the whole expression differentiable.

Sign of the bias is fixed and known: log-sum-exp over-estimates a max and
under-estimates a min, so ``gap_beta <= gap_hard``, by at most
``log(N_occ)/beta + log(N_empty)/beta``. Pick ``beta`` from that bound rather
than by eye, and report both numbers.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

__all__ = [
  "soft_band_edges", "soft_band_gap", "hard_band_edges", "hard_band_gap",
  "softening_bias_bound",
]

# Log-weight standing in for log(0). A finite value rather than -inf so the
# backward pass never forms 0 * inf; large enough that exp() of it underflows to
# zero against any physical eigenvalue scale.
_LOG_ZERO = -1e30


def _log_weight(w: Float[Array, "*n"]) -> Float[Array, "*n"]:
  """``log(w)`` with zero weights mapped to a finite sentinel.

  Guarding the *argument* of the log rather than its result: ``log(0) = -inf``
  has derivative ``inf``, and ``0 * inf = NaN`` would then poison the gradient
  even for the states that carry no weight.
  """
  positive = w > 0.
  return jnp.where(positive, jnp.log(jnp.where(positive, w, 1.)), _LOG_ZERO)


def soft_band_edges(
  eigenvalues: Float[Array, "*n"],
  occupations: Float[Array, "*n"],
  beta: float,
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
  """Smooth ``(VBM, CBM)`` from eigenvalues and occupations.

  Args:
    eigenvalues: Kohn-Sham eigenvalues, any shape; reduced over every axis.
    occupations: occupancy in ``[0, 1]``, same shape. For a spin-restricted run
      pass ``f/2`` or any consistent normalisation -- only the ratio matters.
    beta: inverse sharpness, in inverse units of ``eigenvalues``. Larger is
      closer to the hard edges and stiffer to optimise through.

  Returns:
    ``(vbm, cbm)`` as scalars.
  """
  eps = jnp.asarray(eigenvalues)
  occ = jnp.clip(jnp.asarray(occupations), 0., 1.)

  # Weights enter as log-weights added to the exponent, NOT as a small floor on
  # the occupations: a floor leaks. With b = occ + 1e-30, an empty state at high
  # energy still contributes exp(beta * eps) * 1e-30, which for beta = 100 and
  # eps = 4 Ha overwhelms 1e-30 by hundreds of orders of magnitude and drags the
  # soft VBM up (and the soft CBM down) until the gap goes negative.
  vbm = logsumexp(beta * eps + _log_weight(occ)) / beta
  cbm = -logsumexp(-beta * eps + _log_weight(1. - occ)) / beta
  return vbm, cbm


def soft_band_gap(
  eigenvalues: Float[Array, "*n"],
  occupations: Float[Array, "*n"],
  beta: float,
) -> Float[Array, ""]:
  """``CBM_beta - VBM_beta``. Differentiable everywhere, including at crossings."""
  vbm, cbm = soft_band_edges(eigenvalues, occupations, beta)
  return cbm - vbm


def hard_band_edges(
  eigenvalues: Float[Array, "*n"],
  occupations: Float[Array, "*n"],
  threshold: float = 0.5,
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
  """The true ``(VBM, CBM)``. For reporting, never inside a gradient."""
  eps = jnp.asarray(eigenvalues)
  occ = jnp.asarray(occupations)
  filled = occ >= threshold
  vbm = jnp.max(jnp.where(filled, eps, -jnp.inf))
  cbm = jnp.min(jnp.where(filled, jnp.inf, eps))
  return vbm, cbm


def hard_band_gap(eigenvalues, occupations, threshold: float = 0.5):
  """The true gap. For reporting, never inside a gradient."""
  vbm, cbm = hard_band_edges(eigenvalues, occupations, threshold)
  return cbm - vbm


def softening_bias_bound(occupations, beta: float, threshold: float = 0.5) -> float:
  """Upper bound on ``gap_hard - gap_beta``.

  Log-sum-exp over ``N`` terms exceeds their max by at most ``log(N)/beta``, so
  the soft gap is low by at most ``(log N_occ + log N_empty) / beta``. Use this
  to choose ``beta`` for a target tolerance instead of guessing.
  """
  occ = jnp.asarray(occupations)
  n_occ = float(jnp.sum(occ >= threshold))
  n_empty = float(occ.size - n_occ)
  return float((jnp.log(jnp.maximum(n_occ, 1.)) +
                jnp.log(jnp.maximum(n_empty, 1.))) / beta)
