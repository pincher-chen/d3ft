"""Fit a Hamiltonian parameter to a measured elastic constant.

.. warning::

   **The two AlP fits this module produced are retracted.** They converged
   cleanly -- 3 Newton steps to theta = 0.4798 (gradient step 0.10) and 0.4757
   (step 0.15), C44 landing on the 70 GPa target to 0.2% -- but they were fitting
   an artefact. The gradient they used, ``dC44/d(theta) = +17.31 GPa``, comes
   from the stress route, and the relaxed-energy curvature puts the true value
   between -0.2 and -0.8 GPa: **C44 does not respond to a uniform scaling of the
   Kleinman-Bylander D matrix.** A target 8.6 GPa away from the start is then
   reachable only because the objective it is descending is not C44.

   :mod:`d3ft.calc.stationary` carries the measurements and the diagnosis. The
   Newton machinery here is not implicated -- ``test_15_fit`` exercises it against
   a synthetic linear response and it is exact -- it needed a parameter that
   actually couples to the constant being fitted.

   **The retracted target is now reachable, with a per-channel parameter.**
   Fitting ``theta_nonlocal_l`` to ``C44 = 65`` GPa converged in three Newton steps
   to ``[0.0938, -0.0853, -0.0038]``, 0.042 GPa from target, taking C44 from the
   relaxed-energy curvature and the gradient from
   :func:`d3ft.calc.stationary.dfree_energy_dtheta_l`. The s and p channels move in
   opposite directions, which is exactly what a single uniform theta cannot do.

   **With one, it works.** Fitting theta to a target *bulk modulus* converged in
   two Newton steps to ``theta = 0.09565``, against an independently measured
   ``0.10 +/- 0.005``, using the stationary gradient from
   :mod:`d3ft.calc.stationary`: 6 ground states where a finite-difference gradient
   would have taken 18. The gradient moved from +97.91 to +91.17 GPa between the
   two steps, so the curvature is real and Newton absorbed it.

The objective is a number an experiment reports:

    L(theta) = (C44_relaxed(theta) - C44_target)^2

and its gradient has to run through the variational state, through force balance
at the relaxed internal coordinate, and out to the shear stress.

**How that gradient is obtained here: finite differences over theta, with each
point fully re-relaxed.** The implicit chain in :mod:`d3ft.calc.implicit` was
meant to supply it cheaply and does not -- it omits the electronic response
``d(psi)/d(theta)``, which a mixed second derivative needs and the variational
property does not give for free; see that module's docstring for the measured
failure. Until a Sternheimer or implicit-differentiation solve replaces it, the
gradient costs two full relaxations per parameter.

That cost is the honest state of the method and it bounds what can be fitted: a
handful of parameters, not a network. What makes even this affordable is Newton
relaxation -- pass ``hessian`` through ``relax_kwargs`` and each relaxation takes
about two steps instead of seventeen.

**The ionic relaxation is not optional in the objective.** Clamped-ion C44 for
AlP is 84.9 GPa against an experimental 61.5; relaxed it is 61.4. A fit against
the clamped quantity would be fitting the wrong number by 38%.

``theta`` scales the Kleinman-Bylander D matrix, ``D -> (1 + theta) D``. It is a
stand-in for any learnable Hamiltonian parameter, chosen because it is a single
scalar acting directly on the nonlocal potential, so the optimisation landscape
can be looked at without a network in the way.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .elastic import _HA_BOHR3_TO_GPA
from .forces import relax_internal_coordinates
from .stress import stress_from_ground_state


def c44_and_gradient(
    config,
    symbols: Sequence[str],
    theta: float,
    delta: float = 1.0e-2,
    theta_step: float = 0.02,
    relax_kwargs: Optional[dict] = None,
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Relaxed-ion C44 at ``theta`` and its derivative, both in GPa.

    Two shear directions, each relaxed and each carrying an implicit-chain
    derivative. Returns the clamped-ion pieces as well, since the contrast
    between them is the point.

    Args:
        config: base config at the reference geometry.
        symbols: chemical symbols in crystal order.
        theta: current value of the parameter.
        delta: shear amplitude for the finite difference defining C44.
        theta_step: central-difference step in theta.
        relax_kwargs: passed to the internal relaxation; supply ``hessian`` to
            take Newton steps, which cuts each relaxation from about 17 steps to
            2 and is what makes an iterated fit affordable at all.
        workdir: scratch directory.
        verbose: report each solve.

    Returns:
        Dict with ``C44``, ``dC44_dtheta``, and the two displaced values.
    """
    relax_kwargs = dict(relax_kwargs or {})
    relax_kwargs.setdefault("max_steps", 22)
    relax_kwargs.setdefault("force_tol", 1.0e-4)
    relax_kwargs.setdefault("step", 1.0)

    def c44_at(th):
        """Relaxed-ion C44 at a given theta, by relaxing both shear directions."""
        cfg = copy.deepcopy(config)
        cfg.d3ft.theta_nonlocal = float(th)
        sig = {}
        for sign, tag in ((+1, "plus"), (-1, "minus")):
            strain = np.zeros((3, 3))
            strain[1, 2] = strain[2, 1] = sign * delta
            r = relax_internal_coordinates(
                cfg, symbols, strain=strain, workdir=workdir, verbose=False,
                **relax_kwargs,
            )
            sig[tag] = float(np.asarray(
                stress_from_ground_state(r["ground_state"]))[1, 2])
            if verbose:
                print(f"    theta={th:+.5f} eps_yz={sign * delta:+.4f}: "
                      f"{r['steps']} steps, converged={r['converged']}, "
                      f"sigma_yz={sig[tag]:+.4e}")
        return 0.5 * (sig["plus"] - sig["minus"]) * _HA_BOHR3_TO_GPA / (2.0 * delta)

    c44 = c44_at(theta)
    # Central difference over theta. Each point is two relaxations, so this is
    # four ground-state solves per gradient with Newton steps and thirty-four
    # without -- the reason the hessian argument exists.
    cp = c44_at(theta + theta_step)
    cm = c44_at(theta - theta_step)
    dc44 = (cp - cm) / (2.0 * theta_step)

    return {
        "theta": float(theta),
        "C44": float(c44),
        "dC44_dtheta": float(dc44),
        "C44_plus": float(cp),
        "C44_minus": float(cm),
    }


def fit_theta_to_c44(
    config,
    symbols: Sequence[str],
    target: float,
    theta0: float = 0.0,
    max_steps: int = 6,
    tol: float = 0.5,
    damping: float = 1.0,
    theta_bounds: tuple = (-0.5, 0.5),
    **kwargs,
) -> Dict[str, Any]:
    """Newton iteration on ``theta`` to hit a target C44, in GPa.

    ``theta <- theta - damping * (C44 - target) / (dC44/dtheta)``. Newton rather
    than gradient descent because the derivative is the expensive object here --
    once it is computed, using only its sign would waste it.

    Args:
        config: base config.
        symbols: chemical symbols.
        target: C44 to hit, GPa. AlP's experimental value is 61.5.
        theta0: starting parameter.
        max_steps: cap on iterations.
        tol: stop when ``|C44 - target|`` falls below this, GPa.
        damping: scale on the Newton step; below 1 for a landscape that
            overshoots.
        theta_bounds: clip, since a large ``theta`` rescales the nonlocal
            potential enough to change which states are bound.
        **kwargs: forwarded to :func:`c44_and_gradient`.

    Returns:
        Dict with the ``history`` of (theta, C44, dC44/dtheta), the final
        ``theta`` and ``C44``, and whether it ``converged``.
    """
    theta = float(theta0)
    history = []
    converged = False
    for step in range(max_steps):
        r = c44_and_gradient(config, symbols, theta, **kwargs)
        grad = r["dC44_dtheta"]
        residual = r["C44"] - target
        history.append({
            "step": step, "theta": theta, "C44": r["C44"],
            "residual": residual, "dC44_dtheta": r["dC44_dtheta"],
            "grad_used": grad,
        })
        if kwargs.get("verbose", True):
            print(f"  [{step}] theta = {theta:+.6f}  C44 = {r['C44']:8.3f}  "
                  f"residual = {residual:+8.3f}  dC44/dtheta = {grad:+.4g}")
        if abs(residual) < tol:
            converged = True
            break
        if abs(grad) < 1e-9:
            if kwargs.get("verbose", True):
                print("  gradient vanished; stopping")
            break
        theta = float(np.clip(theta - damping * residual / grad, *theta_bounds))

    return {
        "theta": theta,
        "C44": history[-1]["C44"] if history else None,
        "converged": converged,
        "steps": len(history),
        "history": history,
        "target": target,
    }
