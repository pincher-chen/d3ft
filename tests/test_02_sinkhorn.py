"""Level-2: Sinkhorn occupation correctness. CPU only, fast (<10 s).

Run on the login node:
    python tests/test_02_sinkhorn.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_occupation_sums_to_n_electrons():
    """Sinkhorn occupation should sum to N_e."""
    from d3ft.occupation.sinkhorn import sinkhorn_occupation

    num_electrons = 8.0
    spin, kpt, band = 1, 1, 10
    # Fake energies: lower bands lower energy
    energies = jnp.ones((spin, kpt, band)) * jnp.arange(band)[None, None, :]
    occ = sinkhorn_occupation(energies, num_electrons=num_electrons, epsilon=0.1,
                              spin_restricted=True, n_iter=100)
    total = float(jnp.sum(occ))
    print(f"  sum(occ) = {total:.6f}  (expected {num_electrons})")
    assert abs(total - num_electrons) < 0.01, f"Occupation sum error: {total}"


def test_occupation_range():
    """Each occupation should be in [0, capacity=2] for spin-restricted."""
    from d3ft.occupation.sinkhorn import sinkhorn_occupation

    num_electrons = 6.0
    energies = jax.random.normal(jax.random.PRNGKey(42), (1, 2, 8))
    occ = sinkhorn_occupation(energies, num_electrons=num_electrons, epsilon=0.05,
                              spin_restricted=True, n_iter=200)
    min_o = float(jnp.min(occ))
    max_o = float(jnp.max(occ))
    print(f"  occ range: [{min_o:.4f}, {max_o:.4f}]  (expected [0, 2])")
    assert min_o >= -1e-6, f"Occupation below 0: {min_o}"
    assert max_o <= 2.0 + 1e-6, f"Occupation above 2: {max_o}"


def test_epsilon_schedule_log():
    """Log schedule should be monotonically decreasing and bounded."""
    from d3ft.occupation.sinkhorn import epsilon_schedule

    eps_vals = [epsilon_schedule(t, 1.0, 1e-3, 1000, "log") for t in range(0, 1001, 50)]
    print(f"  eps_schedule log: {[f'{e:.4f}' for e in eps_vals[:6]]} ...")
    assert all(eps_vals[i] >= eps_vals[i+1] - 1e-10 for i in range(len(eps_vals)-1)), \
        "Log schedule not monotonically decreasing"
    assert eps_vals[-1] >= 1e-3 - 1e-10, f"Schedule went below epsilon_final: {eps_vals[-1]}"
    assert eps_vals[0] <= 1.0 + 1e-10


def test_occupation_differentiable():
    """jax.grad should work through sinkhorn_occupation."""
    from d3ft.occupation.sinkhorn import sinkhorn_occupation

    num_electrons = 4.0
    energies = jnp.array([[[0.1, 0.5, 1.0, 1.5, 2.0, 3.0]]])  # [1, 1, 6]

    def f(e):
        occ = sinkhorn_occupation(e, num_electrons=num_electrons, epsilon=0.2,
                                  spin_restricted=True, n_iter=50)
        return jnp.sum(occ)

    grad = jax.grad(f)(energies)
    print(f"  grad shape: {grad.shape}  max|grad|={float(jnp.max(jnp.abs(grad))):.4f}")
    assert grad.shape == energies.shape, "Gradient shape mismatch"
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient!"


TESTS = [
    test_occupation_sums_to_n_electrons,
    test_occupation_range,
    test_epsilon_schedule_log,
    test_occupation_differentiable,
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
