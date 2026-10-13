"""Non-linear core correction (NLCC).

Adapted from ``jrystal.pseudopotential.nlcc`` (Copyright 2025 Garena Online
Private Limited, Apache-2.0); see the project NOTICE.

The pseudo-core charge is placed on the FFT grid exactly like the local
potential: each atom's radial core charge is spherical-Bessel transformed at
``l = 0``, multiplied by the structure factor, summed over atoms and inverse-FFT'd.
It is added to the valence density *only* inside the exchange-correlation
functional (``rho_xc = rho_val + rho_core``); Hartree, kinetic, local, nonlocal
and Ewald all keep using the valence density.
"""
from typing import List

import jax.numpy as jnp
import numpy as np
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

from jrystal.sbt import sbt_numerical

from .geometry import safe_norm


def core_charge_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  core_charge_grid: List[Float[Array, "r"]],
  vol: float,
  kmax: float = None,
) -> Complex[Array, "x y z"]:
  """``rho_core(G)`` summed over atoms, in the FFT-array convention.

  ``kmax`` sizes the NumPy tabulation grid and must be given explicitly when
  ``g_vector_grid`` is traced (e.g. under a strain derivative).
  """
  g_radius = safe_norm(g_vector_grid, axis=-1)
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))
  if kmax is None:
    kmax = float(np.max(np.asarray(g_radius)))

  rho_g = []
  for r, rho_c in zip(r_grid, core_charge_grid):
    kk, f_k = sbt_numerical(r, jnp.expand_dims(rho_c, axis=0), l=0, kmax=kmax)
    rho_g.append(CubicSpline(kk, f_k[0], axis=1)(g_radius) * 4 * jnp.pi)
  rho_g = jnp.stack(rho_g)                              # [atom x y z]

  structure_factor = jnp.exp(-1.j * jnp.matmul(g_vector_grid, positions.transpose()))
  rho_g = rho_g * jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  return jnp.sum(rho_g, axis=0) * num_grids / vol


def core_charge_real(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  core_charge_grid: List[Float[Array, "r"]],
  vol: float,
  kmax: float = None,
) -> Float[Array, "x y z"]:
  """Real-space ``rho_core(r)``; its cell integral is the total core charge."""
  rho_g = core_charge_reciprocal(
    positions, g_vector_grid, r_grid, core_charge_grid, vol, kmax=kmax
  )
  dim = g_vector_grid.shape[-1]
  return jnp.real(jnp.fft.ifftn(rho_g, axes=range(-dim, 0)))
