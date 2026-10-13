"""CPU unit test: implicit (Hellmann-Feynman) gradient through Lanczos eigenvalues.

Validates that `d3ft.spectrum.implicit.eigvals_with_grad` produces:
  (1) forward eigenvalues matching dense `eigvalsh` (lowest k), and
  (2) gradients w.r.t. parameters matching reverse-mode AD through dense `eigvalsh`.

Run on CPU:
    JAX_PLATFORMS=cpu python3 tests/test_05_lanczos_grad.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from d3ft.spectrum.implicit import eigvals_with_grad


def build_hermitian(params, A0):
    """A parametric Hermitian matrix: H(p) = A0 + sum_k p_k * (E_k + E_k^H),
    a smooth, well-defined family. Here we use a simple diagonal-scaling family.
    """
    # H(p) = A0 + diag(p)  (real diagonal perturbation -> stays Hermitian)
    return A0 + jnp.diag(params).astype(A0.dtype)


def main():
    key = jax.random.PRNGKey(0)
    dim = 24
    k_return = 4

    # Fixed Hermitian base.
    kk = jax.random.split(key, 3)
    M = jax.random.normal(kk[0], (dim, dim)) + 1j * jax.random.normal(kk[1], (dim, dim))
    A0 = (M + M.conj().T) / 2.0
    A0 = A0.astype(jnp.complex128)

    params = jax.random.normal(kk[2], (dim,), dtype=jnp.float64)

    def H_fn(p, v):
        return build_hermitian(p, A0) @ v

    # ---- Reference: dense lowest-k eigenvalues and their gradient ----
    def dense_lowest(p):
        H = build_hermitian(p, A0)
        ev = jnp.linalg.eigvalsh(H)  # ascending
        return ev[:k_return]

    ev_dense = dense_lowest(params)

    # scalar objective = sum of lowest-k eigenvalues (so grad is well-defined,
    # no degeneracy ambiguity for distinct eigenvalues)
    g_dense = jax.grad(lambda p: jnp.sum(dense_lowest(p)))(params)

    # ---- Lanczos implicit-gradient ----
    # k_iter = dim ensures the Krylov subspace spans the whole space => exact.
    def lanczos_lowest(p):
        return eigvals_with_grad(
            H_fn, p, dim=dim, k_iter=dim, n_probes=1, key=key, k_return=k_return
        )

    ev_lanczos = lanczos_lowest(params)
    g_lanczos = jax.grad(lambda p: jnp.sum(lanczos_lowest(p)))(params)

    ev_err = float(jnp.max(jnp.abs(jnp.sort(ev_lanczos) - ev_dense)))
    g_err = float(jnp.max(jnp.abs(g_lanczos - g_dense)))
    g_rel = g_err / (float(jnp.max(jnp.abs(g_dense))) + 1e-12)

    print(f"dim={dim} k_return={k_return} k_iter={dim}")
    print(f"eigvals dense   : {ev_dense}")
    print(f"eigvals lanczos : {jnp.sort(ev_lanczos)}")
    print(f"max |Δeigval|        = {ev_err:.3e}")
    print(f"max |Δgrad|          = {g_err:.3e}")
    print(f"max |Δgrad| relative = {g_rel:.3e}")

    assert ev_err < 1e-6, f"eigenvalue mismatch too large: {ev_err}"
    assert g_rel < 1e-5, f"gradient mismatch too large: {g_rel}"
    print("PASS: Lanczos implicit gradient matches dense eigvalsh autodiff.")


if __name__ == "__main__":
    main()
