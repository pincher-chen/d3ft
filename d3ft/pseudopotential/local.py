"""Local pseudopotential in reciprocal space.

Adapted from ``jrystal.pseudopotential.local`` (Copyright 2025 Garena Online
Private Limited, Apache-2.0); see the project NOTICE.

Two things differ from upstream.

**The radial quadrature keeps r = 0.** The mesh handed in has already been masked
to ``r > 0``, so integrating over it silently drops ``[0, r_1]``. For the
integrand ``f(r) = 4 pi r^2 (V(r) + Z/r) j_0(qr)``, which tends to ``4 pi Z r`` as
``r -> 0``, the dropped area is ``4 pi Z r_1^2 / 2``: a q-INDEPENDENT,
Z-PROPORTIONAL offset added to ``Vhat_loc(G)`` at every G. Measured against DFTK
0.7.16 on the dojo Al/P meshes (dr = 0.01) the offsets were -2.17e-3 and -3.62e-3,
matching ``4 pi Z r_1^2 / 2`` to the residual Euler-Maclaurin endpoint term
``h^2/12 * f'(0)``. Re-inserting ``r = 0`` (where the integrand vanishes) and
using trapezoid weights removes both.

**Norms are NaN-safe**, so a gradient with respect to the cell survives; see
:mod:`d3ft.pseudopotential.geometry`.
"""
import os
from typing import List

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

from jrystal._src import braket
from jrystal._src.grid import g2r_vector_grid
from jrystal.sbt import sbt_numerical

from .geometry import safe_norm


def _compensation_real(z, r, mode: str, alpha: float):
  """Analytic long-range part subtracted before the radial transform."""
  if mode == "bare":
    return z / r
  if mode == "gaussian":
    return z * jsp.special.erf(alpha * r) / r
  raise ValueError(f"Invalid compensation {mode!r}. Expected 'bare' or 'gaussian'.")


def _trapezoid_weights(r_ext: np.ndarray) -> np.ndarray:
  w = np.empty_like(r_ext)
  w[0] = (r_ext[1] - r_ext[0]) / 2.
  w[-1] = (r_ext[-1] - r_ext[-2]) / 2.
  w[1:-1] = (r_ext[2:] - r_ext[:-2]) / 2.
  return w


def potential_local_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  local_potential_grid: List[Float[Array, "r"]],
  local_potential_charge: List[int],
  vol: float,
  fourier_transform_method: str = "sbt",
  r0_fix: bool = None,
  kmax: float = None,
  alpha_z: bool = True,
  r_ab=None,
) -> Complex[Array, "x y z"]:
  """Local pseudopotential on the reciprocal grid, in the FFT representation.

  The Coulomb tail is handled analytically: ``V(r) + Z/r`` is transformed
  numerically and ``4 pi Z / G^2`` subtracted afterwards, so the integrand decays.

  Args:
    positions: Cartesian atomic positions, shape ``[atom, 3]``.
    g_vector_grid: reciprocal grid, shape ``[x, y, z, 3]``.
    r_grid: per-atom radial meshes (already masked to r > 0).
    local_potential_grid: per-atom ``V_loc(r)`` in Hartree.
    local_potential_charge: per-atom valence charge Z.
    vol: cell volume.
    fourier_transform_method: ``"sbt"`` (default), ``"direct"`` (quadrature at the
      exact ``|G|``, no interpolation, NOT differentiable) or ``"fft"``.
    alpha_z: keep the finite G = 0 part of the local potential (QE's alpha Z
      term). Setting it False reproduces the upstream behaviour of zeroing all of
      G = 0, and exists only for regression work.
    kmax: upper end of the radial transform grid. Required when ``g_vector_grid``
      is traced (e.g. under a strain derivative), since it sizes a NumPy grid and
      so cannot be read off a tracer. Defaults to ``max |G|`` of the given grid.
    r0_fix: keep the ``[0, r_1]`` part of the radial integral. Defaults to the
      ``VLOC_R0_FIX`` environment variable, itself defaulting to on. Turning it
      off reproduces the upstream offset and exists only for regression work.

  Returns:
    ``Vhat_loc`` scaled to the FFT-array convention used by the Hamiltonian,
    i.e. ``ifftn(result)`` is the real-space local potential.
  """
  if r_ab is not None:
    if fourier_transform_method != "sbt" or not alpha_z:
      raise ValueError("QE local radial metadata requires default SBT/alpha-Z")
    from .qe_local import qe_local_reciprocal
    return qe_local_reciprocal(positions,g_vector_grid,r_grid,
                              local_potential_grid,local_potential_charge,
                              vol,r_ab=r_ab,kmax=kmax)

  if r0_fix is None:
    r0_fix = os.environ.get("VLOC_R0_FIX", "1") == "1"
  comp_mode = os.environ.get("VLOC_COMPENSATION", "bare").lower()
  comp_alpha = float(os.environ.get("VLOC_COMP_ALPHA", "1.0"))

  g_radius = safe_norm(g_vector_grid, axis=-1)
  r_radius = safe_norm(g2r_vector_grid(g_vector_grid), axis=-1)
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))
  if kmax is None:
    if isinstance(jnp.asarray(g_radius), jax.core.Tracer):
      raise ValueError(
        "potential_local_reciprocal needs an explicit kmax when g_vector_grid is "
        "traced: it sizes a NumPy tabulation grid, which is a discretisation "
        "choice rather than something to differentiate through."
      )
    kmax = float(np.max(np.asarray(g_radius)))

  def _one_atom(r, v_r, z):
    v_prime = v_r + _compensation_real(z, r, comp_mode, comp_alpha)

    if fourier_transform_method == "sbt":
      if r0_fix:
        # sbt() defaults to a LEFT-RECTANGLE rule over the supplied mesh. Prepend
        # r = 0 (where r^2 (V + Z/r) vanishes) and hand it trapezoid weights.
        r_ext = np.concatenate(([0.], np.asarray(r)))
        v_ext = np.concatenate(([0.], np.asarray(v_prime)))
        kk, f_k = sbt_numerical(
          r_ext, jnp.expand_dims(jnp.asarray(v_ext), 0), l=0,
          kmax=kmax, delta_r=_trapezoid_weights(r_ext),
        )
      else:
        kk, f_k = sbt_numerical(r, jnp.expand_dims(v_prime, 0), l=0, kmax=kmax)
      return CubicSpline(kk, f_k[0], axis=1)(g_radius) * 4 * jnp.pi

    if fourier_transform_method == "direct":
      # Quadrature at the exact |G| points: no spline, so no interpolation error.
      gg = np.asarray(g_radius).reshape(-1)
      rr = np.asarray(r)
      vp = np.asarray(v_prime)
      r_ext = np.concatenate(([0.], rr)) if r0_fix else rr
      out = np.zeros_like(gg, dtype=float)
      for st in range(0, gg.size, 2048):
        ed = min(st + 2048, gg.size)
        j0 = np.sinc((gg[st:ed][:, None] * rr[None, :]) / np.pi)
        integrand = vp[None, :] * j0 * (rr[None, :] ** 2)
        if r0_fix:
          integrand = np.concatenate(
            (np.zeros((integrand.shape[0], 1)), integrand), axis=1)
        out[st:ed] = 4 * np.pi * np.trapezoid(integrand, r_ext, axis=1)
      return jnp.asarray(out.reshape(np.asarray(g_radius).shape))

    if fourier_transform_method == "fft":
      v_3d = CubicSpline(r, v_prime)(r_radius)
      return jnp.fft.fftn(v_3d, axes=range(-3, 0)) / num_grids * 4 * jnp.pi

    raise ValueError(
      f"Invalid fourier_transform_method {fourier_transform_method!r}. "
      "Expected 'sbt', 'direct' or 'fft'."
    )

  v1_g = jnp.stack([
    _one_atom(r, v, z)
    for r, v, z in zip(r_grid, local_potential_grid, local_potential_charge)
  ])                                                   # [atom x y z]

  # Analytic Coulomb tail. G = 0 is neutralised below, so the placeholder there
  # only has to avoid a division by zero.
  g_radius_safe = g_radius.at[0, 0, 0].set(1e10)
  charges = jnp.expand_dims(jnp.array(local_potential_charge), axis=(1, 2, 3))
  v2_g = charges / jnp.expand_dims(g_radius_safe ** 2, 0) * 4 * jnp.pi
  if comp_mode == "gaussian":
    v2_g = v2_g * jnp.exp(-jnp.expand_dims(g_radius ** 2, 0) / (4 * comp_alpha ** 2))

  # G = 0. The 4 pi Z / G^2 Coulomb divergence in v2 is cancelled elsewhere --
  # jrystal's Hartree term zeroes its own G = 0 component and the Ewald sum
  # supplies the neutralising background -- but the FINITE part of the radial
  # transform is physical and must survive:
  #
  #     alpha_a = 4 pi int_0^inf r^2 (V_a(r) + Z_a / r) dr = v1_a(G -> 0)
  #
  # This is QE's "alpha Z" term. Dropping the whole of G = 0, as upstream does,
  # loses N_elec * sum_a alpha_a / Omega from the total energy -- 0.70 Ry for AlP
  # at a0 = 5.467 A -- and, because that scales as 1/Omega, an isotropic
  # -E_alphaZ/Omega from every diagonal component of the stress.
  v_g = v1_g - v2_g
  if alpha_z:
    v_g = v_g.at[:, 0, 0, 0].set(v1_g[:, 0, 0, 0])
  else:
    v_g = v_g.at[:, 0, 0, 0].set(0)

  structure_factor = jnp.exp(-1.j * jnp.matmul(g_vector_grid, positions.transpose()))
  v_g = v_g * jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  v_g = jnp.sum(v_g, axis=0)

  # The radial quadrature gives the continuous form factor Vhat(G), whose
  # plane-wave matrix element is Vhat(G-G')/Omega. ifftn carries a 1/N and the
  # operator is applied as FFT[v(r) * IFFT(c)], so the FFT-array representation
  # is N/Omega * Vhat(G).
  return v_g * num_grids / vol


def hamiltonian_local(wave_grid, potential_local_grid_reciprocal, vol):
  """Local-potential matrix elements in the band basis."""
  v_loc_real = jnp.fft.ifftn(potential_local_grid_reciprocal, axes=range(-3, 0))
  return braket.expectation(wave_grid, v_loc_real, vol, diagonal=False, mode="real")


def energy_local(reciprocal_density_grid, potential_local_grid_reciprocal, vol):
  """Local-potential energy from the reciprocal-space density."""
  return braket.reciprocal_braket(
    potential_local_grid_reciprocal, reciprocal_density_grid, vol
  )
