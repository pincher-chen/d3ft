"""Stochastic Lanczos eigenvalue extraction.

This file implements the core of **Module 3 of D3FT v0.1 (D3FT-Lanczos)**.

Given an implicit Hermitian operator ``H_op(v)`` that returns :math:`H v` (provided by
JAX reverse-mode AD over the kinetic + effective-potential terms), :func:`lanczos_eigvals`
returns the lowest :math:`k` eigenvalues via :math:`k_{\\text{iter}}` steps of Lanczos
tridiagonalisation. Eigenvalues of the resulting tridiagonal matrix are computed with a
standard dense solver (cost :math:`O(k_{\\text{iter}}^2)`, negligible).

Total cost: :math:`O(k_{\\text{iter}} \\cdot \\mathrm{cost}(\\text{HVP}))`, which scales
as :math:`O(k_{\\text{iter}} \\cdot \\dim)` for a typical plane-wave Hamiltonian. This
is the headline asymptotic improvement over Jrystal's :math:`O(\\dim^2)` matrix
assembly.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def lanczos_tridiag(
    H_op: Callable[[Complex[Array, "n"]], Complex[Array, "n"]],
    dim: int,
    k_iter: int,
    key: jax.Array,
) -> tuple[Float[Array, "k"], Float[Array, "k_minus_1"]]:
    r"""Run :math:`k_{\text{iter}}` steps of Lanczos starting from a random unit vector.

    Args:
        H_op: A function that computes :math:`H v` given :math:`v` of shape ``[n]``.
        dim: Dimension :math:`n` of the operator.
        k_iter: Number of Lanczos iterations (Krylov subspace size).
        key: PRNG key for the random starting vector.

    Returns:
        ``(alphas, betas)`` — the diagonal (length ``k_iter``) and sub-diagonal
        (length ``k_iter - 1``) of the tridiagonal matrix in the Krylov subspace.

    Notes:
        For numerical stability at large ``k_iter`` it is common to run
        *full re-orthogonalisation* (re-project against all previous Krylov vectors).
        This costs an extra :math:`O(k_{\text{iter}}^2 \cdot n)` and is recommended for
        :math:`k_{\text{iter}} \gtrsim 32`.

        .. todo:: add a ``reorthogonalize: bool`` flag.
    """
    v = jax.random.normal(key, (dim,), dtype=jnp.complex64)
    v = v / jnp.linalg.norm(v)

    alphas = jnp.zeros(k_iter)
    betas = jnp.zeros(k_iter - 1)
    v_prev = jnp.zeros_like(v)
    beta_prev = 0.0

    for j in range(k_iter):
        w = H_op(v) - beta_prev * v_prev
        alpha = jnp.real(jnp.vdot(w, v))
        w = w - alpha * v
        alphas = alphas.at[j].set(alpha)

        if j < k_iter - 1:
            beta = jnp.linalg.norm(w)
            betas = betas.at[j].set(beta)
            v_prev = v
            v = w / (beta + 1.0e-14)
            beta_prev = beta

    return alphas, betas


def lanczos_eigvals(
    H_op: Callable[[Complex[Array, "n"]], Complex[Array, "n"]],
    dim: int,
    k_iter: int,
    n_probes: int,
    key: jax.Array,
    k_return: int | None = None,
    reorth: bool = True,
) -> Float[Array, "k"]:
    r"""Approximate the lowest eigenvalues of :math:`H` via stochastic Lanczos.

    Args:
        H_op: HVP function.
        dim: Operator dimension.
        k_iter: Krylov subspace size per probe.
        n_probes: Number of independent random probes; their eigenvalue spectra are
            averaged (variance reduction by Hutchinson-style trace estimation).
        key: PRNG key.
        k_return: How many of the lowest eigenvalues to return. Defaults to ``k_iter``.
        reorth: Use full re-orthogonalisation (recommended). Without it, converged
            eigenvalues replicate as spurious "ghost" copies once orthogonality is
            lost, which corrupts the lowest-eigenvalue estimates.

    Returns:
        A real array of length ``k_return`` containing the estimated lowest
        eigenvalues, sorted ascending.

    Notes:
        For band-structure calculations, typical values are
        ``k_iter ≈ 64`` and ``n_probes ≈ 8`` for plane-wave dimensions in the hundreds.
    """
    k_return = k_iter if k_return is None else k_return
    keys = jax.random.split(key, n_probes)

    all_eigs = []
    for k in keys:
        if reorth:
            alphas, betas, _ = lanczos_tridiag_basis(H_op, dim, k_iter, k)
        else:
            alphas, betas = lanczos_tridiag(H_op, dim, k_iter, k)
        T = jnp.diag(alphas) + jnp.diag(betas, k=1) + jnp.diag(betas, k=-1)
        eigs = jnp.linalg.eigvalsh(T)
        all_eigs.append(jnp.sort(eigs)[:k_return])

    return jnp.mean(jnp.stack(all_eigs, axis=0), axis=0)


def lanczos_tridiag_basis(
    H_op: Callable[[Complex[Array, "n"]], Complex[Array, "n"]],
    dim: int,
    k_iter: int,
    key: jax.Array,
    dtype: jnp.dtype = jnp.complex64,
) -> tuple[Float[Array, "k"], Float[Array, "k_minus_1"], Complex[Array, "n k"]]:
    r"""Lanczos with **full re-orthogonalisation**, also returning the Krylov basis.

    Unlike :func:`lanczos_tridiag`, this variant re-projects each new Lanczos
    vector against *all* previously generated vectors (cost
    :math:`O(k_{\text{iter}}^2 n)`). This is required whenever the Krylov basis
    ``V`` is needed downstream — e.g. to form Ritz vectors for eigenvalue
    gradients (see :mod:`d3ft.spectrum.implicit`). Without re-orthogonalisation
    the basis loses orthogonality and the Ritz vectors are unreliable.

    Returns:
        ``(alphas, betas, V)`` where ``V`` has shape ``[dim, k_iter]`` with the
        (approximately orthonormal) Lanczos vectors as columns.
    """
    v = jax.random.normal(key, (dim,), dtype=dtype)
    v = v / jnp.linalg.norm(v)

    alphas = jnp.zeros(k_iter)
    betas = jnp.zeros(k_iter - 1)
    V = jnp.zeros((dim, k_iter), dtype=dtype)
    V = V.at[:, 0].set(v)
    v_prev = jnp.zeros_like(v)
    beta_prev = 0.0

    for j in range(k_iter):
        w = H_op(v) - beta_prev * v_prev
        alpha = jnp.real(jnp.vdot(w, v))
        w = w - alpha * v
        # Full re-orthogonalisation against the stored basis (modified GS, twice
        # for numerical robustness).
        for _ in range(2):
            w = w - V @ (V.conj().T @ w)
        alphas = alphas.at[j].set(alpha)

        if j < k_iter - 1:
            beta = jnp.linalg.norm(w)
            betas = betas.at[j].set(beta)
            v_prev = v
            v = w / (beta + 1.0e-14)
            beta_prev = beta
            V = V.at[:, j + 1].set(v)

    return alphas, betas, V


def lanczos_ritz(
    H_op: Callable[[Complex[Array, "n"]], Complex[Array, "n"]],
    dim: int,
    k_iter: int,
    key: jax.Array,
    k_return: int | None = None,
    dtype: jnp.dtype = jnp.complex64,
) -> tuple[Float[Array, "k"], Complex[Array, "n k"]]:
    r"""Lowest Ritz pairs (eigenvalue estimates **and** Ritz vectors).

    The Ritz values are the eigenvalues of the :math:`k\times k` tridiagonal
    matrix; the Ritz vectors are :math:`V\,u_i`, where :math:`u_i` are the
    eigenvectors of the tridiagonal matrix and :math:`V` the Krylov basis. These
    Ritz vectors approximate the eigenvectors of :math:`H` and are used by the
    eigenvalue-gradient (Hellmann–Feynman) backward pass.

    Returns:
        ``(ritz_vals, ritz_vecs)`` with ``ritz_vals`` of length ``k_return`` and
        ``ritz_vecs`` of shape ``[dim, k_return]`` (columns are unit-norm).
    """
    k_return = k_iter if k_return is None else k_return
    alphas, betas, V = lanczos_tridiag_basis(H_op, dim, k_iter, key, dtype=dtype)
    T = jnp.diag(alphas) + jnp.diag(betas, k=1) + jnp.diag(betas, k=-1)
    evals, evecs = jnp.linalg.eigh(T)
    order = jnp.argsort(evals)[:k_return]
    ritz_vals = evals[order]
    ritz_vecs = V @ evecs[:, order]  # [dim, k_return]
    # Normalise (guards against basis non-orthogonality).
    ritz_vecs = ritz_vecs / (jnp.linalg.norm(ritz_vecs, axis=0, keepdims=True) + 1e-14)
    return ritz_vals, ritz_vecs
