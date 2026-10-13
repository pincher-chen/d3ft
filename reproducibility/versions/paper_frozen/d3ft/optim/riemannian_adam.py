"""Kinetic-preconditioned tangent-projected Adam with QR retraction.

This file implements **Module 1 of D3FT v0.1 (D3FT-Riem)**.

The update at iteration :math:`t` reads:

1. Get Euclidean gradient :math:`g_t = \\nabla F(X_t)` (from JAX AD).
2. Project to tangent space: :math:`\\tilde g_t = \\mathrm{proj}_{X_t}(g_t)`.
3. Update ambient Adam moments from the tangent-projected gradients:
   :math:`m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\tilde g_t`,
   :math:`v_t = \\beta_2 v_{t-1} + (1-\\beta_2) |\\tilde g_t|^2`.
4. Form the adaptively normalised direction:
   :math:`d_t = -\\alpha \\frac{\\hat m_t}{\\sqrt{\\hat v_t} + \\epsilon}`.
5. Apply the kinetic factor after normalisation and project the result again:
   :math:`d_t^{\\mathrm{tan}} = \\mathrm{proj}_{X_t}(P^{-1}\\odot d_t)`.
6. Retract back to the manifold: :math:`X_{t+1} = \\mathcal{R}_{X_t}(d_t^{\\mathrm{tan}})`.

The moments remain in ambient coordinates; this implementation does not parallel
transport them and does not inherit a convergence theorem for a different
Riemannian Adam construction.

Returned as an ``optax.GradientTransformation`` so it composes with the rest of the
Optax pipeline (scheduling, clipping, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

from .stiefel import project_tangent, retract_qr


class RiemannianAdamState(NamedTuple):
    count: jnp.ndarray
    mu: Any
    nu: Any


def riemannian_adam(
    learning_rate: Any = 0.05,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1.0e-8,
    preconditioner: Optional[Any] = None,
    retraction: str = "qr",
) -> optax.GradientTransformation:
    """Construct the tangent-projected Adam ``GradientTransformation``.

    Args:
        learning_rate: Step size :math:`\\alpha`. A float, or a callable of the
            step count for a schedule. Constant rates do not converge; see the
            note in ``update_fn``.
        b1: First-moment exponential decay :math:`\\beta_1`.
        b2: Second-moment exponential decay :math:`\\beta_2`.
        eps: Numerical stabiliser in the denominator.
        preconditioner: Optional ``pytree`` of diagonal preconditioner values with the
            same structure as the wavefunction parameters. ``None`` means no
            preconditioning (pure Riemannian Adam).
        retraction: Retraction method. Currently only ``"qr"`` is implemented.

    Returns:
        An ``optax.GradientTransformation`` whose ``update`` step performs
        steps 2–6 of the description above. The caller is responsible for invoking
        :func:`stiefel.retract_qr` on the returned updates when applying them to the
        parameters (see :mod:`d3ft.calc.ground_state` for the wiring).
    """
    if retraction != "qr":
        raise NotImplementedError("Only QR retraction is wired up in v0.1.")

    tree_map = jax.tree_util.tree_map

    def init_fn(params):
        mu = tree_map(jnp.zeros_like, params)
        nu = tree_map(jnp.zeros_like, params)
        return RiemannianAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "Riemannian Adam requires the current params to project onto the "
                "tangent space. Pass them via `optimizer.update(grads, state, params)`."
            )

        count = state.count + 1

        # JAX returns the *conjugate* Wirtinger gradient for real-valued functions of
        # complex inputs (this is why Jrystal's real {w_re, w_im} parameterisation never
        # hit it). Conjugate first to recover the standard Euclidean gradient; otherwise
        # the projected search direction ascends the energy. See tests/debug_stiefel_gd.py.
        updates = tree_map(jnp.conj, updates)

        proj_grads = tree_map(project_tangent, params, updates)

        mu = tree_map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, proj_grads
        )
        nu = tree_map(
            lambda n, g: b2 * n + (1.0 - b2) * (g.real ** 2 + g.imag ** 2),
            state.nu, proj_grads,
        )

        mu_hat = tree_map(lambda m: m / (1.0 - b1 ** count), mu)
        nu_hat = tree_map(lambda n: n / (1.0 - b2 ** count), nu)

        # A callable learning rate is a schedule of the step count, the same
        # convention optax uses. This matters more than it looks: with a constant
        # rate Adam never settles, it orbits the minimum at a fixed radius. On
        # AlP the variational residual sat at 2.4e-3 to 2.8e-3 from 4000 steps to
        # 32000 without improving, the energy wandered by 0.3 mRy, and the
        # acoustic sum rule went 2.9e-2, 8.6e-3, 3.9e-3, 2.9e-2 eV/A -- not
        # converging, drifting.
        lr = learning_rate(count) if callable(learning_rate) else learning_rate

        direction = tree_map(
            lambda m, n: -lr * m / (jnp.sqrt(n) + eps),
            mu_hat, nu_hat,
        )

        # Apply the diagonal preconditioner AFTER Adam's per-coordinate normalisation.
        # If applied to the raw gradient (as the proposal states literally), the factor
        # cancels in mu/sqrt(nu); applying it here gives each frequency mode a distinct
        # effective step size (small for high-|G| / high-curvature modes), which is the
        # intended Teter-Payne-Allan effect.
        if preconditioner is not None:
            direction = tree_map(lambda d, p: d * p, direction, preconditioner)

        direction = tree_map(project_tangent, params, direction)

        return direction, RiemannianAdamState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
