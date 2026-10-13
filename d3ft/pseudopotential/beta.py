"""Kleinman-Bylander projectors in reciprocal space, differentiable in the cell.

Adapted from ``jrystal.pseudopotential.beta`` (Copyright 2025 Garena Online
Private Limited, Apache-2.0); see the project NOTICE.

Upstream evaluates ``beta_l(|G + k|)`` entirely in NumPy/SciPy, so no derivative
reaches the cell. The split here is what makes it differentiable at no extra
cost: the spherical Bessel transform depends only on the UPF radial data and on
the tabulation range, never on the cell or on the atomic positions, so it stays
in NumPy and carries no gradient. Only the interpolation at ``|G + k|`` runs in
JAX, via ``interpax``.

Atomic positions never flow through this module. They enter the projector
through the structure factor ``exp(-i(G+k).R)`` in :mod:`d3ft.pseudopotential.nonlocal_`.

**The evaluation grid is the cutoff sphere, not the FFT box.** Everything here is
written against an opaque set of trailing "grid" axes, so ``gk_vector_grid`` may
be either ``[kpt, x, y, z, 3]`` (the whole FFT box) or ``[kpt, n_g, 3]`` (only the
G-vectors inside the wavefunction cutoff, from :func:`gk_vectors`). The projectors
are only ever contracted against wavefunction coefficients, which are zero outside
that sphere, so the two give the same numbers -- but the sphere holds ~15% of the
box, and ``n_g`` is set by the cutoff alone, so refining the grid stops growing it.
See :func:`d3ft.pseudopotential.nonlocal_.projectors_reciprocal`.
"""
import os
from typing import List, Sequence, Tuple

import interpax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from jrystal.sbt import sbt_numerical

from .geometry import safe_norm
from .quadrature import qe_simpson_weights, restored_mesh
from scipy.special import spherical_jn
from scipy.integrate import simpson


def beta_sbt_table(
  r_grid: Float[Array, "r"],
  beta_grid: Float[Array, "beta r"],
  angular_momentum: Sequence[int],
  kmax: float,
  r_ab=None,
  cutoff_size=None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Spherical Bessel transform of one atom's projectors on a fixed k grid.

  .. math::
      \\hat\\beta_l(k) = \\int_0^\\infty \\beta_l(r) j_l(kr) r^2 \\, dr

  Args:
    r_grid: radial mesh of the projectors.
    beta_grid: projector values, shape ``[beta, r]``.
    angular_momentum: ``l`` per projector.
    kmax: upper end of the tabulation grid. Must cover ``max |G + k|`` for every
      cell the caller will evaluate at. It is a discretisation choice, not a
      physical parameter, which is why it has to be a concrete float rather than
      something derived from a traced cell.

  Returns:
    ``(k, beta_k)`` with shapes ``[nk]`` and ``[beta, nk]``.
  """
  r=np.asarray(r_grid, dtype=float); b=np.asarray(beta_grid, dtype=float)
  if r_ab is not None:
    r,b,rab=restored_mesh(r,b,r_ab)
    if cutoff_size is not None:
      r,b,rab=r[:cutoff_size],b[:,:cutoff_size],rab[:cutoff_size]
    weights=qe_simpson_weights(rab)
  else:
    weights=simpson(np.eye(len(r)), x=r, axis=1)
  if not np.all(np.diff(r)>0) or b.shape != (len(angular_momentum),len(r)):
    raise ValueError("Invalid projector mesh or shape")
  k=np.arange(0., float(kmax)+0.010001, 0.01)
  values=np.empty((len(angular_momentum),len(k)))
  for start in range(0,len(k),512):
    q=k[start:start+512,None]*r[None,:]
    for i,l in enumerate(angular_momentum):
      values[i,start:start+512]=spherical_jn(int(l),q)@(b[i]*r*r*weights)
  return k,values


def beta_sbt_reciprocal(
  k_table: Float[Array, "nk"],
  beta_k_table: Float[Array, "beta nk"],
  gk_vector_grid: Float[Array, "kpt *grid 3"],
) -> Float[Array, "kpt beta *grid"]:
  """Evaluate ``beta_l(|G + k|)`` inside the JAX graph.

  ``interpax`` is used rather than ``scipy.interpolate`` precisely because it is
  differentiable; this is the step that carries the cell derivative.

  ``*grid`` is whatever trailing axes ``gk_vector_grid`` carries -- ``x y z`` for
  the FFT box, a single ``n_g`` for the cutoff sphere.
  """
  radius = safe_norm(gk_vector_grid, axis=-1)          # [kpt *grid]
  spline = interpax.CubicSpline(
    jnp.asarray(k_table), jnp.asarray(beta_k_table), axis=1, check=False
  )
  return jnp.swapaxes(spline(radius), 0, 1)            # [kpt beta *grid]


def beta_grid(
  r_grid: List[Float[Array, "r"]],
  beta_grids: List[Float[Array, "beta r"]],
  angular_momentum: List[Sequence[int]],
  gk_vector_grid: Float[Array, "kpt *grid 3"],
  kmax: float,
  tables: List[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[Float[Array, "kpt beta *grid"]]:
  """Per-atom projectors on a reciprocal grid, differentiable in the cell.

  Drop-in replacement for ``jrystal.calc.pre_calc.pre_calc_beta_sbt``.

  Args:
    tables: optional pre-built output of :func:`beta_sbt_table`, one per atom.
      Pass this when sweeping cells so the transform is not redone every step.
  """
  if tables is None:
    tables = [
      beta_sbt_table(r, b, l, kmax)
      for r, b, l in zip(r_grid, beta_grids, angular_momentum)
    ]
  return [
    beta_sbt_reciprocal(jnp.asarray(k), jnp.asarray(bk), gk_vector_grid)
    for k, bk in tables
  ]


def build_tables(pseudopot, kmax: float) -> List[Tuple[np.ndarray, np.ndarray]]:
  """Tabulate every atom's projectors once, for reuse across cells."""
  return [
    beta_sbt_table(r, b, l, kmax, r_ab=rab, cutoff_size=cut)
    for r, b, l, rab, cut in zip(pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
                       pseudopot.nonlocal_angular_momentum, pseudopot.r_ab,
                       getattr(pseudopot,"beta_cutoff_sizes",[None]*len(pseudopot.r_grid)))
  ]


def kmax_for(gk_vector_grid, headroom: float = 1.6) -> float:
  """A concrete ``kmax`` covering the given grid, with room for cell changes."""
  gk = np.asarray(gk_vector_grid)
  return float(np.max(np.sqrt(np.sum(gk ** 2, axis=-1)))) * headroom


def kmax_for_grid(
  g_vector_grid: Float[Array, "x y z 3"],
  k_vector: Float[Array, "kpt 3"],
  headroom: float = 1.6,
) -> float:
  """``kmax_for`` over the whole FFT box, without ever building ``G + k``.

  Equal to ``kmax_for(gk_vectors(g, k))`` to the last bit, but without building
  the full ``[kpt, x, y, z, 3]`` array -- 170 MB at 216 k-points on a 32^3 grid,
  570 MB at grid 48 -- which is the one remaining reason anything would have to.
  The loop is over k-points; each iteration touches one grid's worth of floats.

  **Deliberately the box and not the cutoff sphere.** ``kmax`` sizes the NumPy
  tabulation grid the SBT is evaluated on; a smaller one would move the spline
  nodes and shift every energy in the last few digits. Sampling the sphere would
  be defensible on its own, but it would make the masked projectors numerically
  different from the full-grid ones instead of identical, and the whole claim
  here is that the two are the same calculation. The unused radii cost a few
  table entries.
  """
  g = np.asarray(g_vector_grid).reshape(-1, 3)
  k = np.asarray(k_vector).reshape(-1, 3)
  return float(max(
    np.sqrt(np.sum((g + k_i) ** 2, axis=-1)).max() for k_i in k
  )) * headroom


def mask_indices(freq_mask) -> np.ndarray:
  """Flat indices of the active G-vectors, in the order every masked array uses.

  ``jrystal._src.utils.expand_coefficient`` scatters with ``.at[..., mask]``,
  which is C-order over the True entries; :func:`d3ft.spectrum.ks_matvec.make_ks_hvp`
  gathers with ``np.where(mask.reshape(-1))``. Same order, and this is the one
  place that says so, because a projector axis and a coefficient axis that
  disagree about it would still contract to a plausible-looking number.
  """
  return np.flatnonzero(np.asarray(freq_mask).astype(bool).reshape(-1))


def gk_vectors(
  g_vector_grid: Float[Array, "x y z 3"],
  k_vector: Float[Array, "kpt 3"],
  freq_mask=None,
) -> Float[Array, "kpt *grid 3"]:
  """``G + k`` for every k-point, on the cutoff sphere or on the whole box.

  With ``freq_mask`` the result is ``[kpt, n_g, 3]``, holding only the G-vectors
  the wavefunctions actually use; without it, the full ``[kpt, x, y, z, 3]``.
  Everything downstream of here -- the spherical harmonics table, ``beta_l``, the
  projectors themselves, and every autodiff intermediate along that chain --
  inherits that trailing shape, so this one call is what sizes all of them.

  ``PROJ_MASK=0`` in the environment ignores the mask and restores the full-box
  storage. It exists so the two can be run against each other in one binary on a
  real system, changing one thing: ``test_26`` proves they agree on synthetic
  data, and this is how that was confirmed on an actual ground state (and how the
  memory saving was measured). It is not a compatibility fallback.

  The projector arrays the two produce are bit-identical, but a total energy
  computed from them can differ by an ulp or two, because the contraction sums a
  different number of terms even though every dropped one is an exact zero.
  Measured on Si and Ge at cut 50 with ``init="random"``: 1-2 ulp, up to
  3e-14 Ha. Use random init for any such comparison -- ``init="atomic"``
  diagonalises a dense Hamiltonian built from these projectors, and an ulp there
  selects visibly different starting eigenvectors, so the two runs then take
  different paths to the same minimum (8e-11 Ha on Ge at 300 steps). That is the
  path/minimum distinction of PHASE3_DESIGN 6.2, not a defect here.
  """
  g = jnp.asarray(g_vector_grid)
  k = jnp.asarray(k_vector)
  if freq_mask is None or os.environ.get("PROJ_MASK", "1") == "0":
    return jnp.expand_dims(k, (1, 2, 3)) + jnp.expand_dims(g, 0)
  idx = jnp.asarray(mask_indices(freq_mask))
  return k[:, None, :] + jnp.reshape(g, (-1, 3))[idx][None, :, :]
