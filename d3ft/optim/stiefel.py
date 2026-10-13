"""Stiefel manifold operations.

The complex Stiefel manifold :math:`\\mathrm{St}(n, p) = \\{X \\in \\mathbb{C}^{n \\times p}\\,:\\,X^{\\dagger}X = I_p\\}`
hosts the wavefunction coefficients of a single (spin, k-point) block. D3FT updates
parameters by:

1. Computing a Euclidean gradient via JAX AD.
2. Projecting the Euclidean gradient onto the tangent space at :math:`X`.
3. (Optionally) applying a preconditioner.
4. Stepping via a **retraction** that maps a tangent vector back onto the manifold.

This module provides the projection and retraction primitives.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex


def project_tangent(
    X: Complex[Array, "... n p"],
    V: Complex[Array, "... n p"],
) -> Complex[Array, "... n p"]:
    r"""Project a Euclidean ambient vector :math:`V` onto the tangent space at :math:`X`.

    The tangent space of :math:`\mathrm{St}(n,p)` at :math:`X` is
    :math:`T_X = \{V : X^{\dagger}V + V^{\dagger}X = 0\}`. The orthogonal projection is

    .. math::

        \mathrm{proj}_X(V) = V - X\,\mathrm{sym}(X^{\dagger}V),
        \quad \mathrm{sym}(A) = \tfrac{1}{2}(A + A^{\dagger}).

    Args:
        X: A point on the Stiefel manifold, shape ``[..., n, p]``.
        V: A Euclidean vector at :math:`X`, shape ``[..., n, p]``.

    Returns:
        The tangent-space projection of :math:`V`, shape ``[..., n, p]``.
    """
    XhV = jnp.einsum("...np,...nq->...pq", jnp.conj(X), V)
    sym = 0.5 * (XhV + jnp.swapaxes(jnp.conj(XhV), -1, -2))
    return V - jnp.einsum("...np,...pq->...nq", X, sym)


def retract_qr(
    X: Complex[Array, "... n p"],
    V: Complex[Array, "... n p"],
) -> Complex[Array, "... n p"]:
    r"""QR retraction: :math:`\mathcal{R}_X(V) = \mathrm{qr}(X + V)`.

    Cheap and well-conditioned for moderate step sizes. Returns the Q factor of the
    reduced QR decomposition.

    Args:
        X: A point on the Stiefel manifold, shape ``[..., n, p]``.
        V: A tangent vector at :math:`X`, shape ``[..., n, p]``.

    Returns:
        A new point on the Stiefel manifold, shape ``[..., n, p]``.
    """
    Y = X + V

    def _qr(M):
        Q, _ = jnp.linalg.qr(M, mode="reduced")
        return Q

    leading = Y.shape[:-2]
    if leading:
        Y_flat = Y.reshape((-1,) + Y.shape[-2:])
        Q_flat = jax.vmap(_qr)(Y_flat)
        return Q_flat.reshape(Y.shape)
    return _qr(Y)


def retract_cayley(
    X: Complex[Array, "... n p"],
    V: Complex[Array, "... n p"],
) -> Complex[Array, "... n p"]:
    r"""Cayley retraction.

    A second-order retraction that is more numerically stable than QR for very small
    step sizes. Defined as

    .. math::
        \mathcal{R}_X(V) = (I - \tfrac{1}{2} W)^{-1} (I + \tfrac{1}{2} W) X,
        \quad W = V X^{\dagger} - X V^{\dagger}.

    Args:
        X: A point on the Stiefel manifold, shape ``[..., n, p]``.
        V: A tangent vector at :math:`X`, shape ``[..., n, p]``.

    Returns:
        A new point on the Stiefel manifold, shape ``[..., n, p]``.

    Notes:
        For high-dimensional ``n`` the explicit Cayley formula above is expensive
        (:math:`O(n^3)`). A Woodbury-identity reformulation reduces this to
        :math:`O(np^2)`. The Woodbury variant is the recommended implementation.

        .. todo:: implement Woodbury-form Cayley retraction.
    """
    raise NotImplementedError("Use retract_qr for now; see docstring TODO.")


def stiefel_param_init(
    key: Array,
    num_bands: int,
    num_kpts: int,
    freq_mask: Bool[Array, "x y z"],
    spin_restricted: bool = True,
) -> Complex[Array, "spin kpt gpt band"]:
    r"""Initialise wavefunction coefficients *directly* as a point on the Stiefel manifold.

    Unlike Jrystal's ``pw.param_init`` (which returns unconstrained ``{w_re, w_im}``
    parameters that are orthonormalised downstream inside ``pw.coeff``), this returns a
    complex matrix ``X`` of shape ``(num_spin, num_kpts, num_g, num_bands)`` whose columns
    are already orthonormal for every (spin, k) block, i.e. :math:`X^{\dagger}X = I`.

    This is the parameterisation required by D3FT Contribution 1: each spin–k block is a
    point on the complex Stiefel manifold :math:`\mathrm{St}(n_g, n_b)`, so that
    :func:`project_tangent` and :func:`retract_qr` act with their intended geometry.

    Args:
        key: JAX PRNG key.
        num_bands: Number of bands :math:`n_b` (matrix columns).
        num_kpts: Number of k-points.
        freq_mask: 3D boolean frequency mask; ``num_g`` is the count of ``True`` entries.
        spin_restricted: If True, ``num_spin = 1`` else ``2``.

    Returns:
        Complex array of shape ``(num_spin, num_kpts, num_g, num_bands)`` with orthonormal
        columns per (spin, k) block.
    """
    from jrystal._src.unitary_module import unitary_matrix, unitary_matrix_param_init

    num_spin = 1 if spin_restricted else 2
    num_g = int(np.sum(np.asarray(freq_mask)))
    shape = (num_spin, num_kpts, num_g, num_bands)
    raw = unitary_matrix_param_init(key, shape, complex=True)
    return unitary_matrix(raw, complex=True)


def stiefel_coeff(
    X: Complex[Array, "spin kpt gpt band"],
    freq_mask: Bool[Array, "x y z"],
) -> Complex[Array, "spin kpt band x y z"]:
    r"""Scatter an already-orthonormal Stiefel point onto the 3D frequency grid.

    Equivalent to :func:`jrystal._src.pw.coeff` but **without** the internal QR
    orthonormalisation step, because ``X`` is maintained on the manifold by the
    Riemannian optimiser's retraction. Performing QR again would be redundant and would
    introduce spurious phase/sign churn between steps.

    Args:
        X: Orthonormal coefficients, shape ``(num_spin, num_kpts, num_g, num_bands)``.
        freq_mask: 3D boolean frequency mask used to place the ``num_g`` components.

    Returns:
        Complex array of shape ``(num_spin, num_kpts, num_bands, x, y, z)``, matching the
        output contract of :func:`jrystal._src.pw.coeff`.
    """
    from jrystal._src.utils import expand_coefficient

    return expand_coefficient(X, freq_mask)
