"""D3FT's own norm-conserving pseudopotential path.

D3FT used to call into ``jrystal.pseudopotential`` for all of this. That path had
two defects that only show up on binary systems and on derivatives, so the pieces
D3FT depends on live here instead, with the fixes and with the differentiability
that direct optimisation needs:

* the Kleinman-Bylander term conjugated on the correct side, so an atom's
  projectors sit at ``+R`` rather than ``-R`` (:mod:`~d3ft.pseudopotential.nonlocal_`);
* the local-potential radial quadrature keeping ``[0, r_1]``
  (:mod:`~d3ft.pseudopotential.local`);
* NaN-free derivatives on the reciprocal grid, where ``|x|`` and ``atan2`` are
  singular at the origin and on the polar axis (:mod:`~d3ft.pseudopotential.geometry`);
* ``beta_l(|G+k|)`` evaluated inside the JAX graph so gradients reach the cell
  (:mod:`~d3ft.pseudopotential.beta`);
* the projectors stored on the plane-wave cutoff sphere rather than the FFT box,
  where 84.5% of the coefficients are zero by construction
  (:mod:`~d3ft.pseudopotential.nonlocal_`).

Everything not related to pseudopotentials -- plane-wave basis, crystal and grid
handling, FFTs, XC -- still comes from jrystal.
"""
from .beta import (
  beta_grid,
  beta_sbt_reciprocal,
  beta_sbt_table,
  build_tables,
  gk_vectors,
  kmax_for,
  kmax_for_grid,
  mask_indices,
)
from .geometry import (
  batch_sph_harm,
  batch_sph_harm_real,
  cartesian_to_spherical,
  safe_norm,
  spherical_harmonics_table,
)
from .local import energy_local, hamiltonian_local, potential_local_reciprocal
from .nlcc import core_charge_real, core_charge_reciprocal
from .nonlocal_ import (
  concat_d_matrix,
  scaled_d_matrix,
  energy_nonlocal,
  hamiltonian_nonlocal,
  projectors_reciprocal,
)

__all__ = [
  "beta_grid", "beta_sbt_reciprocal", "beta_sbt_table", "build_tables",
  "gk_vectors", "kmax_for", "kmax_for_grid", "mask_indices",
  "safe_norm", "cartesian_to_spherical", "batch_sph_harm", "batch_sph_harm_real",
  "spherical_harmonics_table",
  "potential_local_reciprocal", "hamiltonian_local", "energy_local",
  "core_charge_real", "core_charge_reciprocal",
  "concat_d_matrix", "scaled_d_matrix", "projectors_reciprocal", "hamiltonian_nonlocal",
  "energy_nonlocal",
]
