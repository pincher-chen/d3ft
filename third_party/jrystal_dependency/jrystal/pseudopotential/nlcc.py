# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Non-linear core correction (NLCC) for plane waves.

The pseudo-core charge density rho_core(r) of an NLCC pseudopotential is placed on
the FFT grid exactly like the local potential: each atom's radial core charge is
spherical-Bessel transformed (l=0) to reciprocal space, multiplied by the structure
factor exp(-i G.R), summed over atoms, and inverse-FFT'd to real space.

The resulting rho_core(r) is added to the valence density *only* inside the exchange
-correlation functional:  rho_xc = rho_valence + rho_core.  Hartree, kinetic, local,
nonlocal and Ewald terms keep using rho_valence.
"""
from typing import List

import numpy as np
import jax.numpy as jnp
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

from ..sbt import sbt_numerical
from .utils import map_over_atoms


def core_charge_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  core_charge_grid: List[Float[Array, "r"]],
  vol: float,
) -> Complex[Array, "x y z"]:
  """Pseudo-core charge density in reciprocal space (sum over atoms).

  Mirrors :func:`jrystal.pseudopotential.local.potential_local_reciprocal` but for a
  bare radial density (no Coulomb tail). Inverse-FFT of the result gives rho_core(r)
  on the real-space grid, normalised so that the cell integral equals the total core
  charge of all atoms.

  Args:
    positions: Atomic positions in Cartesian coordinates, shape [atom, 3].
    g_vector_grid: Reciprocal-space grid, shape [x, y, z, 3].
    r_grid: Per-atom radial grids.
    core_charge_grid: Per-atom core charge density rho_core(r) on r_grid.
    vol: Unit cell volume.

  Returns:
    rho_core(G) on the reciprocal grid, shape [x, y, z].
  """
  g_radius = jnp.sqrt(jnp.sum(g_vector_grid**2, axis=-1))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))

  @map_over_atoms
  def transform(r, rho_c):
    # S(rho_c)[g] = int rho_c(r) j0(g r) r^2 dr ; times 4*pi for the 3D FT.
    kk, f_k = sbt_numerical(r, jnp.expand_dims(rho_c, axis=0),
                            l=0, kmax=np.max(g_radius))
    f_k = CubicSpline(kk, f_k[0], axis=1)(g_radius)
    return f_k * 4 * jnp.pi

  rho_g = transform(r_grid, core_charge_grid)
  rho_g = jnp.stack(rho_g)  # [na, x, y, z]

  # structure factor exp(-i G.R), shape [na, x, y, z]
  structure_factor = jnp.exp(
    -1.j * jnp.matmul(g_vector_grid, positions.transpose())
  )  # [x, y, z, na]
  structure_factor = jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  rho_g = rho_g * structure_factor
  rho_g = jnp.sum(rho_g, axis=0)  # sum over atoms

  rho_g = rho_g * num_grids / vol
  return rho_g


def core_charge_real(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  core_charge_grid: List[Float[Array, "r"]],
  vol: float,
) -> Float[Array, "x y z"]:
  """Real-space pseudo-core charge density rho_core(r) on the FFT grid.

  Returns a non-negative real grid whose cell integral is the total core charge.
  """
  rho_g = core_charge_reciprocal(
    positions, g_vector_grid, r_grid, core_charge_grid, vol
  )
  dim = g_vector_grid.shape[-1]
  rho_r = jnp.fft.ifftn(rho_g, axes=range(-dim, 0))
  return jnp.real(rho_r)
