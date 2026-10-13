"""Kleinman-Bylander nonlocal pseudopotential.

Adapted from ``jrystal.pseudopotential.nloc`` (Copyright 2025 Garena Online
Private Limited, Apache-2.0); see the project NOTICE.

Conventions, which upstream gets wrong in two places and which every binary
system is sensitive to:

**Projector phase and placement.** In the QE/DFTK convention

.. math::
    \\langle G+k | \\beta_{lmi} \\rangle
      = 4\\pi (-i)^l\\, Y_{lm}(\\widehat{G+k})\\, \\beta_i(|G+k|)\\, e^{-i(G+k)\\cdot R}

with the KB coupling ``D_ij`` applied separately, not folded into the projectors.
Upstream folds in a square root of ``D`` obtained from an eigendecomposition,
which silently replaces ``D`` by its eigenvalues in the beta basis; that is only
equivalent when ``D`` is diagonal.

**Which side is conjugated.** ``H_nl = sum_ij |beta_i> D_ij <beta_j|`` means the
inner projection is ``<beta_j|psi> = sum_G conj(beta_j(G)) psi(G)`` and the outer
scatter uses an unconjugated ``beta_i(G)``. Swapping them is equivalent to
replacing every projector by its conjugate, i.e. flipping ``(-i)^l -> (+i)^l`` and
``e^{-i(G+k)R} -> e^{+i(G+k)R}``: the projectors of an atom at ``R`` land at
``-R``. That is harmless for an atom at the origin, but in zincblende the anion
sits at ``a/4(1,1,1)`` and ``2R`` is not a lattice vector, so ``-R`` is an empty
tetrahedral site. The deep state bound by that atom's local potential then never
sees the KB repulsion. See :func:`d3ft.spectrum.ks_matvec.make_ks_hvp` for the
matrix-free form of the same contraction.

**Storage: the cutoff sphere, not the FFT box.** The projectors are only ever
contracted against wavefunction coefficients, and those are zero outside the
frequency mask, so the grid points outside it contribute exactly nothing --
84.5% of them at cutoff 50 Ha on a 32^3 grid. Both functions below therefore
take an opaque set of trailing "grid" axes: ``[kpt, beta, m, x, y, z]`` over the
whole box, or ``[kpt, beta, m, n_g]`` over the sphere, and the second is what the
callers build (see :func:`d3ft.pseudopotential.beta.gk_vectors`). The full-grid
form is kept working because the parity probes and the gradient tests feed it
one, not because anything in a production path needs it.

The point is not only the 6.5x: ``n_g`` is fixed by the cutoff and the cell, so
it does not move when the FFT grid is refined, whereas ``N^3`` does. Grid 48 at
k=6, k=8 at grid 32, and Ge at all were each blocked by that growth.
"""
from typing import List, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from .geometry import spherical_harmonics_table


def concat_d_matrix(
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
) -> Float[Array, "beta beta"]:
  """Block-diagonal KB coupling for all atoms (no cross-atom terms)."""
  blocks = [jnp.asarray(d) for d in nonlocal_d_matrix]
  if len(blocks) == 1:
    return blocks[0]
  total = sum(b.shape[0] for b in blocks)
  out = jnp.zeros((total, total), dtype=blocks[0].dtype)
  start = 0
  for b in blocks:
    n = b.shape[0]
    out = out.at[start:start + n, start:start + n].set(b)
    start += n
  return out


def projectors_reciprocal(
  positions: Float[Array, "atom 3"],
  gk_vector_grid: Float[Array, "kpt *grid 3"],
  angular_momentum: List[Float[Array, "beta"]],
  beta_gk: List[Float[Array, "kpt beta *grid"]],
  concat: bool = True,
):
  """Projectors ``<G+k|beta_{lmi}>`` at the given ``G + k``.

  Args:
    positions: Cartesian atomic positions, shape ``[atom, 3]``.
    gk_vector_grid: ``G + k``, shape ``[kpt, *grid, 3]``. ``*grid`` is the whole
      FFT box ``x y z``, or a single ``n_g`` axis over the cutoff sphere as
      :func:`d3ft.pseudopotential.beta.gk_vectors` returns when given a mask.
    angular_momentum: per-atom list of ``l`` per projector.
    beta_gk: per-atom radial part ``beta_i(|G+k|)``, from
      :func:`d3ft.pseudopotential.beta.beta_grid`.
    concat: concatenate the atom axis into the projector axis.

  Returns:
    ``[kpt, beta, m, *grid]``, or a per-atom list when ``concat`` is False.
  """
  # Broadcasts rather than einsum, because none of these steps is a contraction:
  # every one multiplies by an axis that is constant over the grid. A broadcast
  # does not have to name the grid axes, which is what lets one body serve both
  # ``x y z`` and ``n_g``.
  n_grid = gk_vector_grid.ndim - 2                          # trailing grid axes
  l_max = int(max(int(np.max(np.asarray(l))) for l in angular_momentum))
  y_lm = spherical_harmonics_table(gk_vector_grid, l_max)   # [l kpt *grid m]

  out = []
  for pos, l_atom, beta_atom in zip(positions, angular_momentum, beta_gk):
    y = y_lm[jnp.asarray(l_atom)]                           # [beta kpt *grid m]
    # -> [kpt beta m *grid], the layout hamiltonian_nonlocal contracts on.
    psi = jnp.swapaxes(jnp.moveaxis(y, -1, 2), 0, 1)
    psi = psi * beta_atom[:, :, None]                       # [kpt beta 1 *grid]

    structure_factor = jnp.exp(-1.j * jnp.matmul(gk_vector_grid, pos))
    psi = psi * structure_factor[:, None, None]             # [kpt 1 1 *grid]

    imag_factor = (-1.j) ** jnp.asarray(l_atom)
    psi = psi * imag_factor.reshape((1, -1, 1) + (1,) * n_grid)
    out.append(psi * 4 * jnp.pi)

  return jnp.concatenate(out, axis=1) if concat else out


def _align_to_projectors(pw_coefficients, projectors, freq_mask):
  """Put coefficients and projectors on the same, flattened, G-axis.

  Returns ``([spin, kpt, band, n], [kpt, beta, m, n])``. When the projectors are
  stored on the cutoff sphere and the coefficients on the full FFT box, the
  coefficients are gathered down to the sphere. That is exact rather than an
  approximation: they are identically zero on the points being dropped, because
  ``freq_mask`` is what placed them on the grid to begin with.
  """
  n_grid_proj = projectors.ndim - 3
  n_grid_coeff = pw_coefficients.ndim - 3
  if n_grid_coeff != n_grid_proj:
    if freq_mask is None:
      raise ValueError(
        f"projectors carry {n_grid_proj} grid axis/axes and coefficients "
        f"{n_grid_coeff}; pass freq_mask so the coefficients can be restricted "
        "to the same G-vectors the projectors were evaluated on."
      )
    from .beta import mask_indices
    idx = jnp.asarray(mask_indices(freq_mask))
    pw_coefficients = jnp.reshape(
      pw_coefficients, pw_coefficients.shape[:3] + (-1,)
    )[..., idx]
  c = jnp.reshape(pw_coefficients, pw_coefficients.shape[:3] + (-1,))
  p = jnp.reshape(projectors, projectors.shape[:3] + (-1,))
  if c.shape[-1] != p.shape[-1]:
    raise ValueError(
      f"coefficients span {c.shape[-1]} G-vectors and projectors {p.shape[-1]}; "
      "they must be evaluated on the same set."
    )
  return c, p


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band *grid"],
  projectors: Complex[Array, "kpt beta m *grid"],
  vol: Float,
  d_matrix: Float[Array, "beta beta"],
  freq_mask=None,
) -> Complex[Array, "spin kpt band band"]:
  """``H_nl = sum_ij |beta_i> D_ij <beta_j| / Omega`` in the band basis.

  ``freq_mask`` is needed only in the mixed case -- projectors on the cutoff
  sphere, coefficients on the full grid -- which is what the SCF passes.
  """
  c, p = _align_to_projectors(pw_coefficients, projectors, freq_mask)
  # conj on the INNER projection: <beta_j|psi>. See the module docstring.
  f = jnp.einsum("skbg,kcmg->skbcm", c, jnp.conj(p))
  return jnp.einsum("skigm,gh,skjhm->skij", jnp.conj(f), d_matrix, f) / vol


def energy_nonlocal(
  pw_coefficients, projectors, vol, occupation, d_matrix, freq_mask=None,
) -> Float:
  """Nonlocal energy ``sum_nk f_nk <psi|H_nl|psi>``."""
  h = hamiltonian_nonlocal(
    pw_coefficients, projectors, vol, d_matrix, freq_mask=freq_mask
  )
  import jax
  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(h) * occupation).real


def scaled_d_matrix(pseudopot, theta_nonlocal=0.0, theta_nonlocal_l=None):
    """The KB coupling matrix with a learnable strength, uniform or per channel.

    ``theta_nonlocal`` scales every Kleinman-Bylander channel by ``1 + theta``.
    Measured on AlP PBE, that knob moves the **bulk modulus** hard --
    ``dB/d(theta) = +100 GPa``, taking B from 71.5 to 91.7 over ``theta`` in
    ``[-0.1, 0.1]`` -- and leaves **C44** alone, ``dC44/d(theta)`` of -0.2 to -0.8
    against a 0.3 noise floor. Scaling all channels together renormalises overall
    bond strength, not bond bending, and shear elasticity is bond bending.

    ``theta_nonlocal_l`` is the per-angular-momentum version: an array indexed by
    ``l``, so ``s`` and ``p`` can move independently. In a zincblende crystal the
    bond-angle stiffness is p-channel dominated, which is the reason to expect it
    to couple to C44 where the uniform scaling does not.

    Args:
        pseudopot: the pseudopotential dataclass; supplies ``nonlocal_d_matrix``
            and ``nonlocal_angular_momentum``.
        theta_nonlocal: uniform scalar. Used when ``theta_nonlocal_l`` is None.
        theta_nonlocal_l: per-``l`` values, indexable by the angular momenta in
            ``pseudopot.nonlocal_angular_momentum``. Overrides the scalar.

    Returns:
        The concatenated, scaled D matrix, shape ``[beta, beta]``.
    """
    d = concat_d_matrix(pseudopot.nonlocal_d_matrix)
    if theta_nonlocal_l is None:
        return d * (1.0 + theta_nonlocal)

    # concat_d_matrix lays the per-atom blocks out in order, so the angular
    # momenta concatenate the same way. The symmetrised form is exact for the
    # block-diagonal-in-l matrix this is, and stays symmetric if a
    # pseudopotential ever does couple two channels.
    ls = jnp.concatenate(
        [jnp.asarray(l).ravel() for l in pseudopot.nonlocal_angular_momentum]
    )
    t = jnp.asarray(theta_nonlocal_l)[ls]
    return d * (1.0 + 0.5 * (t[:, None] + t[None, :]))
