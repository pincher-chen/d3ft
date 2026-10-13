"""Level-3: Stochastic Lanczos correctness. CPU only, medium speed (~30 s).

Run on the login node:
    python tests/test_03_lanczos.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _random_hermitian(n: int, key) -> jnp.ndarray:
    A = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    return (A + jnp.conj(A).T) / 2


def test_lanczos_tridiag_shape():
    """lanczos_tridiag should return (alphas, betas) with correct lengths."""
    from d3ft.spectrum.lanczos import lanczos_tridiag

    n, k_iter = 64, 16
    key = jax.random.PRNGKey(0)
    H = _random_hermitian(n, key)
    H_op = lambda v: H @ v

    alphas, betas = lanczos_tridiag(H_op, n, k_iter, jax.random.PRNGKey(1))
    print(f"  alphas.shape={alphas.shape}  betas.shape={betas.shape}")
    assert alphas.shape == (k_iter,), f"Expected ({k_iter},), got {alphas.shape}"
    assert betas.shape == (k_iter - 1,), f"Expected ({k_iter-1},), got {betas.shape}"


def test_lanczos_eigvals_accuracy():
    """Lanczos lowest eigenvalue should approximate true eigvalsh to within 5%."""
    from d3ft.spectrum.lanczos import lanczos_eigvals

    n = 128
    key = jax.random.PRNGKey(42)
    H = _random_hermitian(n, key)
    H_op = lambda v: H @ v

    true_eigs = jnp.sort(jnp.linalg.eigvalsh(H))
    lanczos_eigs = lanczos_eigvals(H_op, n, k_iter=64, n_probes=8,
                                   key=jax.random.PRNGKey(7), k_return=5)

    rel_err = float(jnp.max(jnp.abs(lanczos_eigs - true_eigs[:5])) /
                    (jnp.abs(true_eigs[:5]).mean() + 1e-8))
    print(f"  True  lowest 5 eigs: {np.array(true_eigs[:5]).round(4)}")
    print(f"  Lanczos lowest 5:    {np.array(lanczos_eigs).round(4)}")
    print(f"  Max relative error:  {rel_err:.4f}  (threshold 0.10)")
    assert rel_err < 0.10, f"Lanczos accuracy insufficient: rel_err={rel_err:.4f}"


def test_lanczos_real_symmetric():
    """On a real symmetric matrix, Lanczos should also work (via complex path)."""
    from d3ft.spectrum.lanczos import lanczos_eigvals

    n = 64
    key = jax.random.PRNGKey(99)
    A = jax.random.normal(key, (n, n))
    H = (A + A.T) / 2
    H_op = lambda v: H @ v.real + 0j

    true_eigs = jnp.sort(jnp.linalg.eigvalsh(H))
    est_eigs = lanczos_eigvals(H_op, n, k_iter=32, n_probes=4,
                               key=jax.random.PRNGKey(5), k_return=4)
    rel_err = float(jnp.max(jnp.abs(est_eigs - true_eigs[:4])) /
                    (jnp.abs(true_eigs[:4]).mean() + 1e-8))
    print(f"  Real-symmetric rel_err = {rel_err:.4f}")
    assert rel_err < 0.10


TESTS = [
    test_lanczos_tridiag_shape,
    test_lanczos_eigvals_accuracy,
    test_lanczos_real_symmetric,
]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            print(f"[RUN] {fn.__name__}")
            fn()
            print(f"[OK]  {fn.__name__}\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {e}\n")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} tests passed.")
