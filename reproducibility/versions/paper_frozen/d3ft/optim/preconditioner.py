"""Physics-informed preconditioner for plane-wave DFT.

The kinetic operator :math:`T = \\tfrac{1}{2}|G+k|^2` dominates the curvature of the
free-energy landscape at large :math:`|G|`. A diagonal preconditioner
:math:`P^{-1} = (T + \\mu I)^{-1}` rescales the gradient so that high- and low-frequency
modes are updated on comparable timescales.

This is a long-standing heuristic in plane-wave DFT (Teter–Payne–Allan 1989); D3FT
adopts it inside the Riemannian Adam update.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def kinetic_preconditioner(
    g_vector_grid: Float[Array, "x y z 3"],
    kpts: Float[Array, "kpt 3"],
    freq_mask: Float[Array, "x y z"] | None = None,
    mu: float = 1.0,
) -> Float[Array, "kpt n_g"]:
    r"""Build :math:`(T + \mu I)^{-1}` as a per-(k, G) diagonal vector.

    Args:
        g_vector_grid: G-vector grid in reciprocal space, shape ``[x, y, z, 3]``.
        kpts: Sampled k-points, shape ``[kpt, 3]``.
        freq_mask: Optional boolean mask selecting which G-vectors are active
            (shape ``[x, y, z]``). If provided, the returned tensor is flattened over
            the active G-vectors.
        mu: Damping constant :math:`\mu \ge 0`.

    Returns:
        Diagonal preconditioner values, shape ``[kpt, n_g]`` where ``n_g`` is the
        number of active G-vectors.
    """
    if freq_mask is not None:
        # Restrict G before broadcasting over k.  Broadcasting the full FFT
        # grid first creates [n_k, nx, ny, nz, 3], even though the cutoff mask
        # immediately discards nearly all of it.  For diamond k=12/grid=100
        # that unnecessary temporary is 38.62 GiB on every GPU.
        mask = freq_mask.astype(bool)
        n_g = int(mask.sum())
        g = g_vector_grid.reshape(-1, g_vector_grid.shape[-1])[
            mask.reshape(-1)
        ]
        k = kpts[:, None, :]
        T = 0.5 * jnp.sum((g[None, :, :] + k) ** 2, axis=-1)
        return (1.0 / (T + mu)).reshape(kpts.shape[0], n_g)

    g = g_vector_grid[None, ...]
    k = kpts[:, None, None, None, :]
    T = 0.5 * jnp.sum((g + k) ** 2, axis=-1)
    return 1.0 / (T + mu)
