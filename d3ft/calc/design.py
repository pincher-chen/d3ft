"""Band edges as a differentiable function of strain, for inverse design.

Band-edge derivatives are harder than energy derivatives, and it is worth being
explicit about why. The stress is safe because the free energy is the quantity
being minimised: ``dF/d(params) = 0`` at the solution, so the total strain
derivative collapses to the partial one (Hellmann-Feynman). A band gap is not the
minimised quantity. It depends on the Hamiltonian, which depends on the density,
which depends on the parameters, so

    dE_gap/d(eps) = partial E_gap/partial eps |_params
                    + (partial E_gap/partial params) (d params/d eps)

and the second term does not vanish. Getting it requires either implicit
differentiation at the minimum (essentially DFPT) or differentiating through the
optimiser.

What this module computes is the **first term**: band edges from the Kohn-Sham
Hamiltonian rebuilt at a strained cell, with the electronic state frozen at its
converged value. That is the frozen-density (rigid-band) response. It is cheap,
exactly differentiable, and a well-defined approximation -- not the full
derivative. :func:`density_response_check` measures what it leaves out, by
comparing against central differences of a fully re-converged gap.

The gap itself goes through :mod:`d3ft.calc.bandgap`, never through
``min``/``max``; see that module for why.
"""
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jrystal._src.grid import g_vectors

from .. import pseudopotential as d3pp
from ..spectrum.ks_matvec import make_ks_hvp
from . import bandgap as bg
from .stress import _strained

__all__ = [
  "band_eigenvalues_at_strain", "soft_gap_at_strain",
  "band_edges_report", "density_response_check",
]


def band_eigenvalues_at_strain(
  strain,
  *,
  crystal,
  freq_mask,
  density_grid,
  k_frac_band,
  pseudopot=None,
  beta_tables=None,
  kmax: float = None,
  grid_kmax: float = None,
  core_density_grid=None,
  xc: str = "lda_x",
  n_bands: int = None,
):
  """Kohn-Sham eigenvalues at the given band k-points, for a strained cell.

  The density is held fixed on the FFT grid, i.e. frozen in fractional
  coordinates and carried along by the strain like the grid itself.

  Args:
    strain: symmetric ``[3, 3]`` strain tensor.
    density_grid: converged valence density, ``[x, y, z]`` or ``[spin, x, y, z]``.
    k_frac_band: band-path k-points in fractional coordinates, ``[nk, 3]``.
    n_bands: keep only this many lowest eigenvalues per k-point.

  Returns:
    ``[nk, n_bands]`` eigenvalues in Hartree.
  """
  grid_sizes = np.asarray(freq_mask.shape)
  cell, positions = _strained(
    jnp.asarray(crystal.cell_vectors), jnp.asarray(crystal.positions), strain
  )
  vol = jnp.abs(jnp.linalg.det(cell))
  g_vec = g_vectors(cell, grid_sizes)
  recip = 2 * jnp.pi * jnp.linalg.inv(cell).T
  k_cart = jnp.asarray(k_frac_band) @ recip

  rho = jnp.asarray(density_grid)
  if rho.ndim == 3:
    rho = rho[None, ...]

  pot_loc = d3pp.potential_local_reciprocal(
    positions, g_vec, pseudopot.r_grid, pseudopot.local_potential_grid,
    pseudopot.local_potential_charge, vol, kmax=grid_kmax,
  )
  gk_vec = d3pp.gk_vectors(g_vec, k_cart, freq_mask)
  beta_gk = d3pp.beta_grid(
    pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum, gk_vec, kmax, tables=beta_tables,
  )
  pot_nl = d3pp.projectors_reciprocal(
    positions, gk_vec, pseudopot.nonlocal_angular_momentum, beta_gk,
  )
  nl_d = d3pp.concat_d_matrix(pseudopot.nonlocal_d_matrix)

  eigs = []
  for ik in range(np.asarray(k_frac_band).shape[0]):
    pn_k = jax.tree.map(lambda x: x[ik], pot_nl)
    hop, n_g = make_ks_hvp(
      rho, pot_loc, pn_k, g_vec, k_cart[ik], vol, freq_mask,
      xc=xc, core_density_grid=core_density_grid, nl_d_matrix=nl_d,
    )
    mat = jax.vmap(hop, in_axes=1, out_axes=1)(jnp.eye(n_g, dtype=jnp.complex128))
    # eigvalsh only: its derivative is v_i^dag dA v_i, with no 1/(l_i - l_j)
    # terms, so a degeneracy does not blow it up. Individual eigenvalues are
    # still not differentiable where they cross, which is exactly why the gap
    # below is a soft extremum rather than min/max.
    w = jnp.linalg.eigvalsh(0.5 * (mat + jnp.conj(mat).T))
    eigs.append(w if n_bands is None else w[:n_bands])
  return jnp.stack(eigs)


def _occupation_from_count(eigenvalues, n_occ: int):
  """Hard 1/0 occupations for the ``n_occ`` lowest states at each k-point.

  Only used when a converged occupation array is not supplied. It is a step
  function of the eigenvalues, so it carries no gradient -- fine as a gate for
  the soft extremum, which is smoothed in the eigenvalues, but it does mean the
  gate itself cannot follow a change in band ordering. Prefer passing the
  converged smooth occupations.
  """
  nk, nb = eigenvalues.shape
  return jnp.where(jnp.arange(nb)[None, :] < n_occ, 1.0, 0.0) * jnp.ones((nk, 1))


def soft_gap_at_strain(strain, *, beta: float, n_occ: int = None,
                       occupations=None, **kwargs):
    """Smooth band gap of the strained cell. Differentiable in ``strain``."""
    eigs = band_eigenvalues_at_strain(strain, **kwargs)
    occ = (_occupation_from_count(eigs, n_occ) if occupations is None
           else jnp.asarray(occupations))
    return bg.soft_band_gap(eigs, occ, beta)


def band_edges_report(strain, *, beta: float, n_occ: int = None,
                      occupations=None, **kwargs) -> Dict[str, Any]:
    """Soft and hard edges side by side, for reporting rather than gradients."""
    eigs = band_eigenvalues_at_strain(strain, **kwargs)
    occ = (_occupation_from_count(eigs, n_occ) if occupations is None
           else jnp.asarray(occupations))
    svbm, scbm = bg.soft_band_edges(eigs, occ, beta)
    hvbm, hcbm = bg.hard_band_edges(eigs, occ)
    return {
        "eigenvalues": np.asarray(eigs),
        "soft_vbm": float(svbm), "soft_cbm": float(scbm),
        "soft_gap": float(scbm - svbm),
        "hard_vbm": float(hvbm), "hard_cbm": float(hcbm),
        "hard_gap": float(hcbm - hvbm),
        "bias_bound": bg.softening_bias_bound(occ, beta),
    }


def density_response_check(reconverged_gap_fn, frozen_grad, *,
                           h: float = 1e-3, component=(0, 0)) -> Dict[str, float]:
    """How much of ``dE_gap/d(strain)`` the frozen-density approximation misses.

    .. warning::
       Two confounds have to be controlled first, or the number this returns is
       meaningless. Both were measured on AlP and neither is small.

       **The basis must not change with the strain.** D3FT cuts on ``|G|``, so
       rebuilding a configuration at a different lattice constant moves G-vectors
       in and out of the sphere: a +-0.5% sweep changed ``n_g`` from 893 to 869.
       That is a discrete change in the variational space, and it shows up as
       steps in the energy. ``reconverged_gap_fn`` must therefore strain the
       *cell* at a fixed ``freq_mask``, the way :mod:`d3ft.calc.stress` does, not
       rebuild the configuration from a new lattice constant. (This is also the
       only sense in which the stress is free of a Pulay term.)

       **The ground state must stay on one branch.** Cold-starting each strain
       independently lands on different minima. Over a +-1% AlP sweep the total
       energy jumped 0.033 Ha and the gap 0.24 eV between eta = 0 and eta = 0.5%,
       and the resulting finite-difference derivative changed by 40% when the
       step size was doubled -- the signature of a discontinuity, not of
       curvature. Passing ``init_params`` to
       :func:`d3ft.calc.ground_state.calc` keeps the sweep continuous.

       Note that continuation alone is not sufficient either: at eta = +0.5% a
       cold start reached -9.3587 Ha while continuation from eta = 0 stayed at
       -9.3244 Ha. The lower energy is the cold one, so the branch continuation
       follows is not necessarily the global minimum. Until the optimiser
       reliably reaches the same minimum, a full band-edge strain derivative is
       not measurable on this code path.

    Args:
      reconverged_gap_fn: ``strain -> gap``, running a full ground state at each
        call, at fixed basis and with continuation. Expensive by construction;
        this is a diagnostic, not a loop body.
      frozen_grad: the frozen-density gradient this module produced, ``[3, 3]``.
      h: strain step for the central difference.
      component: which strain component to probe.

    Returns:
      the frozen and full derivatives, their ratio, and the difference. A ratio
      far from 1 means the rigid-band approximation is not usable for design on
      this system.
    """
    a, b = component
    ep = np.zeros((3, 3)); ep[a, b] += h
    em = np.zeros((3, 3)); em[a, b] -= h
    full = (reconverged_gap_fn(ep) - reconverged_gap_fn(em)) / (2 * h)
    frozen = float(np.asarray(frozen_grad)[a, b])
    return {
        "frozen": frozen,
        "full": float(full),
        "ratio": float(frozen / full) if full != 0 else float("nan"),
        "missing": float(full - frozen),
    }
