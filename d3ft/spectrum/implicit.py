"""Implicit differentiation through Lanczos eigenvalue extraction.

Standard reverse-mode AD through :func:`d3ft.spectrum.lanczos.lanczos_eigvals` is
inefficient and numerically unstable because the Lanczos iteration itself contains
re-orthogonalisations and divisions by small numbers. Instead, D3FT uses the **adjoint
method** based on first-order eigenvalue perturbation theory:

.. math::

    \\frac{\\partial \\lambda_i}{\\partial p} \\;=\\; \\langle v_i \\, | \\, \\partial H / \\partial p \\, | \\, v_i \\rangle,

where :math:`v_i` is the eigenvector associated with :math:`\\lambda_i`. JAX provides
the inner adjoint :math:`\\partial H / \\partial p` for free, so the cost of the gradient
is the same as one forward Lanczos plus a JAX-VJP through :math:`H`.

This file wires the perturbation formula into a custom JAX VJP so that
:func:`eigvals_with_grad` can be used inside any JAX-traced computation.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .lanczos import lanczos_ritz


def eigvals_with_grad(
    H_fn: Callable[[Any, jax.Array], jax.Array],
    params: Any,
    dim: int,
    k_iter: int,
    n_probes: int,
    key: jax.Array,
    k_return: int | None = None,
) -> jax.Array:
    r"""Eigenvalues of :math:`H(\text{params})` with implicit gradients in ``params``.

    Args:
        H_fn: A function ``H_fn(params, v) -> Hv`` that computes the action of
            :math:`H(\text{params})` on a vector. Must be JAX-traceable.
        params: A pytree of parameters that :math:`H` depends on.
        dim: Operator dimension.
        k_iter: Krylov subspace size.
        n_probes: Number of stochastic probes.
        key: PRNG key.
        k_return: How many of the lowest eigenvalues to return.

    Returns:
        The lowest ``k_return`` eigenvalues, with a custom VJP that produces
        gradients with respect to ``params`` via eigenvalue perturbation theory.

    Notes:
        Implementation (v0.1):

        1. Forward pass: run :func:`d3ft.spectrum.lanczos.lanczos_ritz` to get the
           lowest eigenvalues *and* their Ritz vectors :math:`v_i` from the Krylov
           subspace (with full re-orthogonalisation so the Ritz vectors are
           reliable). The Ritz vectors are saved as residuals and treated as
           **constants** in the backward pass.
        2. Custom VJP: for upstream cotangent :math:`\bar\lambda_i`, the gradient is

           .. math::
               \frac{\partial}{\partial p}\sum_i \bar\lambda_i\,
               \langle v_i | H(p) | v_i \rangle ,

           computed with a single JAX VJP through ``H_fn`` (Hellmann–Feynman /
           first-order eigenvalue perturbation theory). This avoids
           back-propagating through the Lanczos recurrence entirely.

        ``n_probes`` is accepted for API compatibility; the gradient path uses a
        single Krylov run (the lowest eigenpairs are deterministic given ``key``).
    """
    del n_probes  # gradient path uses a single Krylov run; see docstring.

    def _rayleigh(p, vecs):
        # Real Rayleigh quotients <v_i | H(p) | v_i> for each Ritz vector column.
        def one(v):
            return jnp.real(jnp.vdot(v, H_fn(p, v)))
        return jax.vmap(one, in_axes=1)(vecs)  # [k_return]

    @jax.custom_vjp
    def _wrapper(params):
        H_op = lambda v: H_fn(params, v)
        vals, _ = lanczos_ritz(H_op, dim, k_iter, key, k_return=k_return)
        return vals

    def fwd(params):
        H_op = lambda v: H_fn(params, v)
        vals, vecs = lanczos_ritz(H_op, dim, k_iter, key, k_return=k_return)
        # Recompute exact Rayleigh values for the returned eigenvalues so that the
        # primal of fwd matches `_wrapper` to first order; save Ritz vectors.
        return vals, (params, vecs)

    def bwd(res, eigs_bar):
        params, vecs = res
        vecs = jax.lax.stop_gradient(vecs)
        # d/dp [ sum_i eigs_bar_i * <v_i|H(p)|v_i> ]
        grad_p = jax.grad(
            lambda p: jnp.sum(eigs_bar * _rayleigh(p, vecs))
        )(params)
        return (grad_p,)

    _wrapper.defvjp(fwd, bwd)
    return _wrapper(params)
