"""Level-1: Stiefel manifold math correctness. CPU only, fast (<5 s).

Run on the login node:
    python tests/test_01_stiefel.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_project_tangent_is_tangent():
    """proj(X, V) should satisfy X†·result + result†·X ≈ 0."""
    from d3ft.optim.stiefel import project_tangent
    key = jax.random.PRNGKey(0)
    n, p = 64, 8
    X_raw, V_raw = jax.random.normal(key, (2, n, p)) + 1j * jax.random.normal(key, (2, n, p))
    X, _ = jnp.linalg.qr(X_raw)
    V = V_raw

    TV = project_tangent(X, V)
    skew = jnp.conj(X).T @ TV + jnp.conj(TV).T @ X
    err = float(jnp.max(jnp.abs(skew)))
    print(f"  tangent skew-sym error = {err:.2e}  (should be < 1e-10)")
    assert err < 1e-10, f"Tangent projection failed: err={err}"


def test_retract_qr_on_manifold():
    """retract_qr(X, V)†·retract_qr(X, V) ≈ I_p."""
    from d3ft.optim.stiefel import project_tangent, retract_qr
    key = jax.random.PRNGKey(1)
    n, p = 128, 12
    X_raw = jax.random.normal(key, (n, p)) + 1j * jax.random.normal(key, (n, p))
    X, _ = jnp.linalg.qr(X_raw)
    V = 0.01 * (jax.random.normal(jax.random.PRNGKey(2), (n, p)) +
                1j * jax.random.normal(jax.random.PRNGKey(3), (n, p)))
    V = project_tangent(X, V)

    X_new = retract_qr(X, V)
    ortho_err = float(jnp.max(jnp.abs(jnp.conj(X_new).T @ X_new - jnp.eye(p))))
    print(f"  QR retraction orthogonality error = {ortho_err:.2e}  (should be < 1e-12)")
    assert ortho_err < 1e-12, f"QR retraction failed: err={ortho_err}"


def test_riemannian_adam_init():
    """Riemannian Adam initialises without error on a simple pytree."""
    from d3ft.optim.riemannian_adam import riemannian_adam
    import optax
    key = jax.random.PRNGKey(0)
    n, p = 32, 4
    X_raw = jax.random.normal(key, (n, p)) + 1j * jax.random.normal(key, (n, p))
    X, _ = jnp.linalg.qr(X_raw)
    params = {"w_re": X.real, "w_im": X.imag}

    opt = riemannian_adam(learning_rate=0.01)
    state = opt.init(params)
    print(f"  Riemannian Adam init OK, count={state.count}")


def test_preconditioner_shape():
    """Kinetic preconditioner returns correct shape [kpt, n_g]."""
    from d3ft.optim.preconditioner import kinetic_preconditioner
    import jrystal
    from jrystal._src.grid import g_vectors, k_vectors
    import numpy as np

    # Tiny 2-atom Si-like cell
    cell = jnp.eye(3) * 10.26  # bohr
    g_vec = g_vectors(cell, [8, 8, 8])
    from jrystal._src.grid import spherical_mask
    mask = spherical_mask(cell, [8, 8, 8], cutoff_energy=10.0)
    kpts = k_vectors(cell, [2, 2, 2])

    P_inv = kinetic_preconditioner(g_vec, kpts, freq_mask=mask, mu=1.0)
    n_g = int(mask.sum())
    n_k = kpts.shape[0]
    assert P_inv.shape == (n_k, n_g), f"Expected ({n_k}, {n_g}), got {P_inv.shape}"
    assert float(jnp.min(P_inv)) > 0, "Preconditioner must be positive"

    # The memory-efficient path must be identical to forming the full
    # [kpt, x, y, z] tensor and masking it afterwards.
    full = 1.0 / (
        0.5 * jnp.sum(
            (g_vec[None, ...] + kpts[:, None, None, None, :]) ** 2,
            axis=-1,
        )
        + 1.0
    )
    reference = full.reshape(n_k, -1)[:, mask.reshape(-1)]
    assert bool(jnp.array_equal(P_inv, reference)), "Masked preconditioner changed"
    print(f"  preconditioner shape = {P_inv.shape}  min={float(jnp.min(P_inv)):.3f}  OK")


TESTS = [
    test_project_tangent_is_tangent,
    test_retract_qr_on_manifold,
    test_riemannian_adam_init,
    test_preconditioner_shape,
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
