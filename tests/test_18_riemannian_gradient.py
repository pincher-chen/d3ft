"""A reported gradient has to be a direction the energy actually falls along.

``variational_residual`` is the only number that qualifies the Hellmann-Feynman
stress -- ``stress_tensor`` assumes ``dF/d(params) = 0`` and that residual is the
measurement of how false the assumption is. It reported 2.6e-3 on a converged AlP
state, flat against 8x more optimiser effort, and that number was used to argue
that the stress's premise was insufficient.

It was measuring nothing. JAX returns the conjugate Wirtinger gradient for a
real-valued function of complex input; the magnitude is unaffected, so a
magnitude-only check passes, but ``project_tangent`` applied to the unconjugated
vector keeps a component the optimiser never moves along. Corrected, the same
state reports 5.5e-5, and walking along it reproduces the first-order energy drop
to 0.999. ``d3ft.optim.riemannian_adam`` had the conjugation and the comment
explaining it; ``variational_residual`` had the tangent-space projection and the
comment explaining that. Each was half right.

The guard here is operational rather than analytical, because that is what caught
it: take the reported descent direction, walk along it, and require the energy to
fall by the predicted first-order amount. A gradient that fails that test is not
a gradient whatever its magnitude says.

Pure JAX on a Rayleigh quotient -- no DFT, no GPU, well under a second.
"""

import jax

# Without this the whole file runs in float32 and the 1e-10 tolerances below are
# meaningless -- test_14 was written with the same omission and passed for the
# wrong reason until it was caught.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after the x64 switch, deliberately)

from d3ft.optim.stiefel import project_tangent, retract_qr  # noqa: E402

N, P, SEED = 12, 3, 0


def _problem():
    """A complex Hermitian Rayleigh quotient on St(N, P), and a point off its minimum.

    ``F(X) = Re tr(X^H A X)`` is minimised by the P lowest eigenvectors of A. Any
    other orthonormal X has a genuine descent direction, which is what the checks
    below need.
    """
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(SEED), 4)
    M = jax.random.normal(k1, (N, N)) + 1j * jax.random.normal(k2, (N, N))
    A = 0.5 * (M + M.conj().T)
    Y = jax.random.normal(k3, (N, P)) + 1j * jax.random.normal(k4, (N, P))
    X, _ = jnp.linalg.qr(Y)

    def F(Z):
        return jnp.real(jnp.trace(jnp.conj(Z).T @ A @ Z))

    # A is returned so the minimiser check uses the same operator F closes over;
    # rebuilding it from the seed gave a different matrix and a bogus failure.
    return F, X, A


def _line_search(F, X, direction, alphas):
    """``(dE, predicted)`` for a step ``-alpha * direction``, retracted to the manifold."""
    e0 = float(F(X))
    gg = float(jnp.real(jnp.vdot(direction, direction)))
    out = []
    for a in alphas:
        dE = float(F(retract_qr(X, -a * direction))) - e0
        out.append((a, dE, -a * gg))
    return out


def check_conjugated_gradient_descends(failures):
    """The conjugated, projected gradient must reproduce the first-order slope."""
    F, X, _ = _problem()
    g = jax.grad(F)(X)
    right = project_tangent(X, jnp.conj(g))
    rows = _line_search(F, X, right, (1e-4, 1e-3, 1e-2))
    print(f"  conjugated: max|g| = {float(jnp.max(jnp.abs(right))):.4e}")
    for a, dE, pred in rows:
        ratio = dE / pred
        print(f"    alpha={a:<8.0e} dE={dE:+.6e} predicted={pred:+.6e} ratio={ratio:+.4f}")
        if not 0.9 < ratio < 1.1:
            failures.append(
                f"conjugated gradient at alpha={a} gave ratio {ratio:.4f}, "
                "expected ~1: it is not a descent direction")


def check_unconjugated_gradient_does_not(failures):
    """The bug's signature: same magnitude, no descent. This must stay detectable."""
    F, X, _ = _problem()
    g = jax.grad(F)(X)
    wrong = project_tangent(X, g)
    right = project_tangent(X, jnp.conj(g))
    m_wrong = float(jnp.max(jnp.abs(wrong)))
    m_right = float(jnp.max(jnp.abs(right)))
    rows = _line_search(F, X, wrong, (1e-3,))
    ratio = rows[0][1] / rows[0][2]
    print(f"  unconjugated: max|g| = {m_wrong:.4e} vs conjugated {m_right:.4e} "
          f"(magnitudes are the same order -- that is why it hid)")
    print(f"    alpha=1e-03 ratio={ratio:+.4f}")
    if 0.9 < ratio < 1.1:
        failures.append(
            "the unconjugated projection also descends, so this test no longer "
            "distinguishes the two conventions and cannot guard the bug")


def check_raw_magnitude_cannot_tell_them_apart(failures):
    """Why a magnitude-only check passed for months: conjugation preserves it."""
    F, X, _ = _problem()
    g = jax.grad(F)(X)
    a = float(jnp.max(jnp.abs(g)))
    b = float(jnp.max(jnp.abs(jnp.conj(g))))
    print(f"  max|g| = {a:.6e}   max|conj g| = {b:.6e}   difference {abs(a-b):.1e}")
    if abs(a - b) > 1e-12:
        failures.append("conjugation changed the magnitude, which it must not")


def check_projection_is_a_projection(failures):
    """``project_tangent`` must be idempotent and land in the tangent space."""
    F, X, _ = _problem()
    v = project_tangent(X, jnp.conj(jax.grad(F)(X)))
    again = project_tangent(X, v)
    idem = float(jnp.max(jnp.abs(again - v)))
    XhV = jnp.conj(X).T @ v
    tang = float(jnp.max(jnp.abs(XhV + jnp.conj(XhV).T)))
    print(f"  idempotent to {idem:.2e}; tangency |X^H V + V^H X| = {tang:.2e}")
    if idem > 1e-10:
        failures.append(f"project_tangent is not idempotent: {idem:.2e}")
    if tang > 1e-10:
        failures.append(f"projected vector is not tangent: {tang:.2e}")


def check_minimum_reports_a_small_residual(failures):
    """At the true minimum the corrected residual must collapse; the buggy one need not."""
    F, X, A = _problem()
    _, V = jnp.linalg.eigh(A)
    Xmin = V[:, :P]                      # the exact minimiser
    g = jax.grad(F)(Xmin)
    right = float(jnp.max(jnp.abs(project_tangent(Xmin, jnp.conj(g)))))
    print(f"  at the exact minimiser: corrected residual = {right:.3e}")
    if right > 1e-8:
        failures.append(
            f"corrected residual at the exact minimum is {right:.2e}, not ~0")


def main():
    failures = []
    for check in (
        check_conjugated_gradient_descends,
        check_unconjugated_gradient_does_not,
        check_raw_magnitude_cannot_tell_them_apart,
        check_projection_is_a_projection,
        check_minimum_reports_a_small_residual,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_18_riemannian_gradient")
        return 1
    print("[OK]  the reported Riemannian gradient is a direction the energy falls along")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
