"""NaN-safe reciprocal-space geometry and real spherical harmonics.

Adapted from ``jrystal.pseudopotential.spherical`` (Copyright 2025 Garena Online
Private Limited, Apache-2.0); see the project NOTICE. The reason D3FT carries its
own copy is that the upstream helpers are not differentiable on a reciprocal
grid, which is exactly what direct differentiation needs them to be.

The trap is that a reciprocal grid always contains ``G + k = 0`` and, worse, a
whole polar axis where ``x = y = 0``:

* ``d|x|/dx = x/|x|`` is ``0/0`` at the origin;
* ``d atan2(y, x)`` is ``0/0`` wherever ``x = y = 0``;
* ``d arccos(t)/dt`` diverges at ``t = ±1``, which every point on the polar axis
  hits exactly.

Guarding these with ``jnp.where`` on the *result* does not work: the NaN is
already in the backward pass, and ``0 * NaN = NaN`` poisons the sum even when the
cotangent is zero. Every guard here therefore sits on the *argument*.
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from einops import einsum
from jax.scipy.special import sph_harm_y
from jaxtyping import Array, Float

from jrystal._src.utils import vmapstack

# Only used to keep the r = 0 point out of divisions; the polar angle no longer
# goes through arccos, so nothing here bounds a derivative.
_CLIP_EPS = 1e-10


def safe_norm(
  x: Float[Array, "*n 3"], axis: int = -1, keepdims: bool = False
) -> Float[Array, "*n"]:
  """Euclidean norm whose gradient is 0 (not NaN) at the origin."""
  sq = jnp.sum(x * x, axis=axis, keepdims=keepdims)
  is_zero = sq == 0.
  return jnp.where(is_zero, 0., jnp.sqrt(jnp.where(is_zero, 1., sq)))


def cartesian_to_spherical(
  x: Float[Array, "*n 3"], eps: float = _CLIP_EPS
) -> Float[Array, "*n 3"]:
  """Cartesian -> ``(r, theta, phi)``, differentiable everywhere on the grid.

  ``theta`` is the azimuthal angle in the x-y plane measured from x, in
  ``[0, 2*pi)``; ``phi`` is the polar angle from z, in ``[0, pi]``. Note this is
  the opposite naming convention from ``jax.scipy.special.sph_harm_y``.
  """
  r = safe_norm(x, axis=-1)

  # Polar angle via atan2(rho, z) rather than arccos(z / r).
  #
  # arccos has derivative -1/sqrt(1 - t^2), which diverges as t -> +-1, and every
  # grid point with G + k on the polar axis sits exactly there. Clipping t away
  # from +-1 bounds the derivative only for points that land *exactly* on the
  # axis; a point at t = 1 - 2e-10 slips past the clip and picks up a derivative
  # of order 1/sqrt(2e-10) ~ 5e4. That showed up as an off-diagonal stress of
  # 1.6e+04 Ha/Bohr^3 on a 3x3x3 k-mesh, where such near-axis points occur.
  #
  # atan2(rho, z) is well conditioned everywhere: d/d rho = z/(rho^2 + z^2) and
  # d/dz = -rho/(rho^2 + z^2) are both bounded away from the origin, and rho
  # itself comes from safe_norm, whose derivative is 0 on the axis -- the correct
  # regularisation, since the azimuthal direction is genuinely undefined there.
  rho = safe_norm(x[..., :2], axis=-1)
  z = x[..., 2]
  r_zero = r == 0.
  phi = jnp.where(r_zero, jnp.pi / 2, jnp.arctan2(rho, jnp.where(r_zero, 1., z)))

  xy_zero = (x[..., 0] ** 2 + x[..., 1] ** 2) == 0.
  x0 = jnp.where(xy_zero, 1., x[..., 0])
  x1 = jnp.where(xy_zero, 0., x[..., 1])
  theta = jnp.where(xy_zero, 0., jnp.arctan2(x1, x0))
  theta = jnp.mod(theta + 2 * jnp.pi, 2 * jnp.pi)

  return jnp.stack((r, theta, phi), axis=-1)


def batch_sph_harm(
  l: int, theta: Float[Array, "*b"], phi: Float[Array, "*b"]
) -> Float[Array, "*b m"]:
  """Complex spherical harmonics ``Y_lm`` for m = -l..l."""
  dim = theta.ndim
  m = np.arange(-int(l), int(l) + 1)
  n = np.array([l])

  @vmapstack(dim)
  def _fun(theta, phi):
    # sph_harm_y swaps the meaning of theta and phi relative to this module.
    return sph_harm_y(n, m, phi, theta)

  return _fun(theta, phi)


def batch_sph_harm_real(
  l: int, theta: Float[Array, "*b"], phi: Float[Array, "*b"]
) -> Float[Array, "*b m"]:
  """Real spherical harmonics for m = -l..l."""
  y = batch_sph_harm(l, theta, phi)
  m = jnp.arange(-l, l + 1)
  y_conj = einsum(y.conj(), (-1) ** m, "... m, m -> ... m")
  out = jnp.where(
    m >= 0,
    y.real * jnp.sqrt(2) * (-1) ** m,
    y_conj.imag * jnp.sqrt(2) * (-1) ** m,
  )
  return out.at[..., l].set(y[..., l].real)


def _real_sph_harm_cartesian(l: int, u: Float[Array, "*n 3"]) -> Float[Array, "*n m"]:
  """Real spherical harmonics as polynomials in the unit vector ``u``.

  Angles are avoided entirely. Going through ``theta``/``phi`` puts an associated
  Legendre function at the poles, whose derivative diverges there, and a
  reciprocal grid contains points exactly on the polar axis whenever the k-mesh
  includes Gamma. Clipping the angle bounds that derivative only by making the
  angle wrong; the Cartesian forms are smooth everywhere on the sphere, which is
  also why QE computes them this way (``ylmr2``).

  ``u`` must be unit-normalised. Ordering is m = -l .. l, matching
  :func:`batch_sph_harm_real`.
  """
  x, y, z = u[..., 0], u[..., 1], u[..., 2]
  pi = jnp.pi
  if l == 0:
    return jnp.stack([jnp.full_like(x, 0.5 * jnp.sqrt(1. / pi))], axis=-1)
  if l == 1:
    c = jnp.sqrt(3. / (4. * pi))
    return jnp.stack([c * y, c * z, c * x], axis=-1)
  if l == 2:
    c2 = jnp.sqrt(15. / (4. * pi))
    return jnp.stack([
      c2 * x * y,
      c2 * y * z,
      jnp.sqrt(5. / (16. * pi)) * (3. * z * z - 1.),
      c2 * x * z,
      jnp.sqrt(15. / (16. * pi)) * (x * x - y * y),
    ], axis=-1)
  if l == 3:
    return jnp.stack([
      jnp.sqrt(35. / (32. * pi)) * y * (3. * x * x - y * y),
      jnp.sqrt(105. / (4. * pi)) * x * y * z,
      jnp.sqrt(21. / (32. * pi)) * y * (5. * z * z - 1.),
      jnp.sqrt(7. / (16. * pi)) * z * (5. * z * z - 3.),
      jnp.sqrt(21. / (32. * pi)) * x * (5. * z * z - 1.),
      jnp.sqrt(105. / (16. * pi)) * z * (x * x - y * y),
      jnp.sqrt(35. / (32. * pi)) * x * (x * x - 3. * y * y),
    ], axis=-1)
  raise NotImplementedError(
    f"Cartesian real spherical harmonics are implemented for l <= 3, got l={l}. "
    "UPF projectors beyond l = 3 would need the missing forms added here; the "
    "angle-based batch_sph_harm_real is not a safe fallback because its "
    "derivative is singular on the polar axis."
  )


def spherical_harmonics_table(
  r_vector_grid: Float[Array, "*n 3"], l_max: int
) -> Float[Array, "l *n m"]:
  """Real ``Y_lm`` for every ``l <= l_max``, zero-padded in m.

  Returns shape ``[l_max + 1, *n, 2 * l_max + 1]``. For a given ``l`` only the
  first ``2l + 1`` m-slots are filled; the rest stay zero, so contracting over
  the full m axis is safe.
  """
  ndims = r_vector_grid.shape[:-1]
  r = safe_norm(r_vector_grid, axis=-1, keepdims=True)
  u = r_vector_grid / jnp.where(r == 0., 1., r)

  y_lm = jnp.zeros((l_max + 1, *ndims, 2 * l_max + 1)) + 0.j
  for i in range(l_max + 1):
    y_lm = y_lm.at[i, ..., :(2 * i + 1)].set(_real_sph_harm_cartesian(i, u))
  return y_lm
