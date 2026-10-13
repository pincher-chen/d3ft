"""Parameter derivatives of relaxed-ion elastic constants, via stationarity.

.. note::

   **The formula is verified and closed end to end -- on the bulk modulus.
   On C44 it is not usable, and the reason is the parameter, not the method.**
   All numbers below are AlP PBE at a = 5.5076 Ang, cutoff 20 Ha, Gamma-centred
   6x6x6, basis pinned to the zero-strain mask.

   *The stationarity identity itself.* ``dF_relaxed/d(theta)`` from
   :func:`dfree_energy_dtheta` matches a Richardson-extrapolated finite difference
   of the relaxed free energy over ``theta`` to **4e-6 relative**.

   *Per channel, C44 has a knob after all.* The uniform scaling is not a bad
   parameter for C44 because C44 is insensitive to the nonlocal strength -- it is
   a bad parameter because the channels cancel. With ``theta_nonlocal_l``
   (:func:`d3ft.pseudopotential.scaled_d_matrix`) the sensitivities separate:

   ===========  =====================  =====================
   channel      ``dB/d(theta_l)``      ``dC44/d(theta_l)``
   ===========  =====================  =====================
   ``l = 0`` s  +46.4                  **+30.0**
   ``l = 1`` p  +54.1                  **-24.3**
   ``l = 2`` d  -2.6                   -1.0
   sum          +97.9                  +4.8
   ===========  =====================  =====================

   The B column sums to the uniform +97.909 exactly, and the assignment was
   confirmed by bumping one channel at a time and finite-differencing B (+4.32
   measured against +4.64 predicted for s, +5.31 against +5.41 for p, the
   shortfall being the known curvature of B(theta)). The C44 column is 8 and 6
   sigma against the estimator's +/-3.8 GPa noise at ``delta = 0.02``, and sums to
   within noise of zero: **s and p pull C44 in opposite directions and the uniform
   direction sits in the null space.** The same table also shows that s is nearly a
   pure bulk-modulus knob -- ``dP/d(theta_s) = -3.7`` against
   ``dB/d(theta_s) = +46.4`` -- so per-channel parameters give directions that
   change one property while leaving another fixed, which a single uniform scaling
   cannot do.

   *Closed on B.* Isotropic strain leaves the internal coordinate fixed by cubic
   symmetry, so each point is one ground state and there is no relaxation chain.
   ``dB/d(theta)`` from the second difference of ``g`` is **+97.909 GPa** at
   ``delta = 0.02`` against **+97.6** from recomputing B at ``theta = +/-0.1``,
   i.e. **0.3%**. The signal scales as it should: the second difference goes
   9.579e-04 -> 3.377e-03 as delta doubles (ratio 3.53, ideal 4). A Newton fit of
   theta to a target B converged in two steps to ``theta = 0.09565`` where the
   independent answer is ``0.10 +/- 0.005`` (the reference's own delta spread),
   and it cost **6 ground states against 18** for a finite-difference gradient.

   *And with P = 3.* A Gauss-Newton fit of ``theta_l`` to a target
   ``(B, P) = (90, 0)`` -- both read off the same three ground states, B from the
   second difference of the energy and the pressure from the first -- converged in
   three steps to ``B = 90.004``, ``P = 0.011``, at **9 ground states against 63**.
   The Jacobian is 2x3 and every entry of it comes from the three reverse passes,
   so the scaling claim is measured at P > 1 rather than extrapolated from P = 1:
   3 solves plus 3 reverse passes against 3(2P+1) solves.

   *And on C44 itself, which is what this module was written for.* A Newton fit
   of ``theta_l`` to a target ``C44 = 65`` GPa converged in three steps --
   61.415 -> 64.165 -> 65.042, landing 0.042 GPa from the target -- at **18 ground
   states against about 126**. C44 was taken from the relaxed-energy curvature at
   every step, never from the stress, for the reason in the paragraph after next.
   The gradient is quantitatively right, not merely right in sign: step 0 predicted
   +3.585 and delivered +2.750, and the trapezoidal integral of the gradient along
   that step is +2.885, a 5% account of the shortfall; step 1 predicted +0.835 and
   delivered +0.877. **A scalar theta cannot reach this target at all** -- its
   sensitivity is zero to within noise, so the Newton step divides by zero. Here
   multiple parameters are not an optimisation, they are the difference between
   feasible and infeasible.

   *Why the uniform knob fails.* ``dC44/d(theta)`` from the same construction is noise
   dominated: three seeds gave +11.40, +23.02 and -7.42 GPa, spread 1.7x the mean,
   one sign flip, and enlarging delta does not help because the second difference
   does not grow as ``delta^2``. It does not grow because **C44 barely depends on
   theta**. From the relaxed-energy curvature, ``C44(theta = -0.1, 0, +0.1) =
   60.37, 61.21, 60.32`` GPa at delta = 0.01 and ``60.49, 61.42, 60.33`` at
   delta = 0.02: ``dC44/d(theta)`` is -0.2 to -0.8 against a 0.3 noise floor,
   while ``dB/d(theta)`` is +100. A uniform scaling of the Kleinman-Bylander D
   matrix renormalises overall bond strength, not bond bending. Choose the
   observable the parameter actually couples to, and compute the signal-to-noise
   ratio before spending GPU time: it was ~2 for C44 and ~115 for B at the delta
   each was run at.

   *The +17.31 GPa that :mod:`d3ft.calc.fit` used as its C44 gradient is an
   artefact.* It comes from the stress route, which at ``theta != 0`` carries a
   systematic error: sigma is 2.7% smaller in magnitude than its own energy
   surface's slope at ``|theta| = 0.1``, 0.2% at ``theta = 0``, and odd in theta.
   It is systematic, not noise: four optimiser settings spanning 8x in effort give
   2.773, 2.695, 2.920 and 2.307%. Ruled out as causes: the stress AD (finite
   differences of the same function converge to it as h^2, to five digits), the
   two energy expressions (they agree to 1e-8), the basis (pinned, 1243 at every
   point), the strain step (0.01 and 0.02 agree), and the optimiser trajectory.

   What is left is the variational response term ``(dF/dparams).(dparams/deps)``
   that ``stress_tensor`` assumes away, and the honest status of that attribution
   is **magnitude-compatible but unmeasured**. The residual at the relaxed states
   is 1.7e-5 to 1.7e-4 (max component; ``|g| = 9.3e-4`` at ``theta = 0``), so
   closing the 6.9e-4 Ha per unit strain gap needs ``|dparams/deps|`` of roughly
   0.25 to 0.6 -- a 0.6-1.4% change in the coefficients per unit strain, plausible
   but never checked. And the residual is **not** odd in theta (1.66, 0.84,
   0.68 e-4 at theta = -0.1, 0, +0.1) while the gap changes sign, so it does not
   account for the sign structure either.

   Measuring ``dparams/deps`` means solving the Sternheimer response equation --
   which is also what would remove this error rather than merely explain it. The
   problem and its fix are the same object; see ``ad-dfpt``'s
   ``solve_OmegaplusK_split`` for a reference implementation.

   *An earlier version of this note blamed the same term on the strength of a
   2.1-2.5e-3 residual.* That number was wrong: ``variational_residual`` was
   projecting the unconjugated JAX gradient, inflating it 47x, and it is fixed
   with ``test_18_riemannian_gradient`` guarding it. The corrected residual is
   still large enough for the attribution to survive on magnitude, but the
   argument is now the weaker one written above.

   On B the same stress error is present but is 2.5 GPa against a 20 GPa signal,
   which is why it does no harm there and is fatal for C44.

The obvious route -- differentiate the stress with respect to a Hamiltonian
parameter -- does not work, and :mod:`d3ft.calc.implicit` records the measured
failure. ``sigma`` is itself ``dF/d(eps)``, so ``d(sigma)/d(theta)`` is a mixed
second derivative and needs the electronic response ``d(psi)/d(theta)``, which
the variational property does not supply.

Writing the same elastic constant as a second difference of the **energy**
removes the problem, because what then has to be differentiated with respect to
``theta`` is a *first* derivative of a doubly stationary quantity:

    F_relaxed(eps, theta) = F(R*(eps, theta), psi*(eps, theta), eps, theta)

    dF_relaxed/d(theta) = partial F / partial theta
                        + (partial F / partial R)   . dR*/d(theta)
                        + (partial F / partial psi) . dpsi*/d(theta)

At a geometry relaxed to zero force and a state at the variational minimum, the
second and third terms vanish -- the forces *are* ``partial F / partial R``, and
electronic stationarity is what licenses Hellmann-Feynman in the first place. So

    dF_relaxed/d(theta) = partial F / partial theta   at frozen R and psi,

which is plain autodiff through the converged coefficients. Then, with
``C(theta)`` taken as a second difference of ``F_relaxed`` over strain,

    dC/d(theta) = [g(+delta) - 2 g(0) + g(-delta)] / (Omega delta^2),
    g(eps) = dF_relaxed/d(theta) at the geometry relaxed for that strain.

**What this buys, and it is the point of the module.** The three relaxations are
the ones the elastic constant needs anyway. Adding a parameter costs one more
reverse-mode pass through an already-converged state, not another relaxation, so
the cost is O(1) in the number of parameters where finite differences over theta
are O(N). That is the difference between fitting a handful of physical constants
and fitting a learned correction.

**Two things to watch.**

*Residual force.* ``partial F / partial R = 0`` holds only as well as the
relaxation converged, so the neglected term is ``(residual force) . dR*/dtheta``,
about 1e-4 x 1e-3 = 1e-7 Ha at the tolerances used here. It is first order in a
small quantity, where the stress route was wrong at zeroth order -- wrong sign,
360% to 520%.

*Second differences of energy are numerically softer than first differences of
stress*, which is the usual argument for the stress route. Here it cuts the other
way: the energy is variational, so it is second-order accurate in the error of
the state, while the stress is only first-order. With the cosine schedule the
seed-to-seed energy spread is 0.001 mRy against a signal of roughly 0.086 mRy at
``delta = 0.01`` and four times that at 0.02.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _HA_BOHR3_TO_GPA
from .forces import relax_internal_coordinates
from .implicit import _kwargs_from
from .stress import free_energy_at_strain


def dfree_energy_dtheta(out) -> float:
    """``dF/d(theta)`` at a converged, force-relaxed state, in Ha per unit theta.

    Exact by Hellmann-Feynman provided ``out`` is both electronically converged
    and at zero force; see the module docstring for why both are required and
    what the residual force costs.

    Args:
        out: a :class:`GroundStateOutput` from a relaxed geometry.

    Returns:
        The derivative, a scalar.
    """
    kwargs = _kwargs_from(out)
    natom = np.asarray(out.crystal.positions).shape[0]
    other = {k: v for k, v in kwargs.items()
             if k not in ("params_pw", "params_occ")}
    # The state's own theta, removed from the keyword set so the closure below
    # owns that argument. The derivative has to be taken AT this value, not at
    # zero: differentiating at zero while the state was converged at theta gives
    # a partial of one Hamiltonian evaluated on another one's solution.
    if other.get("theta_nonlocal_l", None) is not None:
        raise ValueError(
            "this state was solved with per-channel theta_nonlocal_l, so the "
            "scalar theta_nonlocal does not enter its Hamiltonian and its "
            "derivative would be identically zero. Use dfree_energy_dtheta_l."
        )
    other.pop("theta_nonlocal_l", None)
    theta0 = float(other.pop("theta_nonlocal", 0.0))
    zero_strain = jnp.zeros((3, 3))
    zero_disp = jnp.zeros((natom, 3))

    def energy(theta):
        return free_energy_at_strain(
            zero_strain, kwargs["params_pw"], kwargs["params_occ"],
            displacement=zero_disp, theta_nonlocal=theta, **other,
        )

    return float(jax.grad(energy)(theta0))


def dfree_energy_dtheta_l(out, n_channels: Optional[int] = None):
    """``dF/d(theta_l)`` for every angular-momentum channel, in one reverse pass.

    The per-channel counterpart of :func:`dfree_energy_dtheta`, and the reason the
    stationarity argument is worth the trouble: the cost of this call does not
    grow with the number of channels, so P parameters cost one pass rather than
    ``2P`` extra ground states. Same licence as the scalar version -- the state
    must be at zero force and at the variational minimum.

    Args:
        out: a converged, force-relaxed :class:`GroundStateOutput`.
        n_channels: length of the returned vector; inferred from the
            pseudopotential's angular momenta when omitted.

    Returns:
        ``numpy`` array of shape ``(n_channels,)``, Hartree per unit theta_l.
    """
    kwargs = _kwargs_from(out)
    natom = np.asarray(out.crystal.positions).shape[0]
    other = {k: v for k, v in kwargs.items()
             if k not in ("params_pw", "params_occ")}
    other.pop("theta_nonlocal", None)
    theta0 = other.pop("theta_nonlocal_l", None)
    if theta0 is None:
        n = n_channels or (1 + max(
            int(np.max(np.asarray(l)))
            for l in out.pseudopot.nonlocal_angular_momentum))
        theta0 = np.zeros(n)
    theta0 = jnp.asarray(theta0, dtype=float)

    zero_strain = jnp.zeros((3, 3))
    zero_disp = jnp.zeros((natom, 3))

    def energy(tl):
        return free_energy_at_strain(
            zero_strain, kwargs["params_pw"], kwargs["params_occ"],
            displacement=zero_disp, theta_nonlocal_l=tl, **other,
        )

    return np.asarray(jax.grad(energy)(theta0))


def dc44_dtheta(
    config,
    symbols: Sequence[str],
    theta: float = 0.0,
    delta: float = 1.0e-2,
    relax_kwargs: Optional[dict] = None,
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Relaxed-ion C44 and ``dC44/d(theta)``, both in GPa, without a theta sweep.

    Three shear strains -- ``-delta``, ``0``, ``+delta`` -- each relaxed to zero
    force. C44 comes from the stresses as usual; its theta-derivative comes from
    the second difference of ``dF/d(theta)`` over the same three points.

    Args:
        config: base config at the reference geometry.
        symbols: chemical symbols in crystal order.
        theta: parameter value to evaluate at.
        delta: shear amplitude.
        relax_kwargs: passed to the relaxation; supply ``hessian`` for Newton
            steps, which is what makes three relaxations cheap.
        workdir: scratch directory.
        verbose: report each relaxation.

    Returns:
        Dict with ``C44``, ``dC44_dtheta``, the three ``g`` values, and the
        relaxation diagnostics.
    """
    relax_kwargs = dict(relax_kwargs or {})
    relax_kwargs.setdefault("max_steps", 22)
    relax_kwargs.setdefault("force_tol", 1.0e-4)
    relax_kwargs.setdefault("step", 1.0)

    cfg = copy.deepcopy(config)
    cfg.d3ft.theta_nonlocal = float(theta)

    # Zero strain first, so its basis can be pinned for the two strained points.
    # |G| <= cutoff is evaluated on the deformed lattice, so letting each point
    # pick its own drops 16 plane waves at |eps_yz| = 0.02 on AlP and shifts the
    # energy by ~0.03 mRy -- comparable to the differences being taken here.
    mask = None
    points = {}
    for tag, sign in (("zero", 0.0), ("minus", -1.0), ("plus", +1.0)):
        strain = np.zeros((3, 3))
        strain[1, 2] = strain[2, 1] = sign * delta
        r = relax_internal_coordinates(
            cfg, symbols, strain=strain, workdir=workdir, verbose=False,
            freq_mask=mask, **relax_kwargs,
        )
        mask = r["freq_mask"] if mask is None else mask
        out = r["ground_state"]
        from .stress import stress_from_ground_state

        sigma = float(np.asarray(stress_from_ground_state(out))[1, 2])
        g = dfree_energy_dtheta(out)
        vol = float(abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))
        points[tag] = {"sigma_yz": sigma, "g": g, "volume": vol,
                       "steps": r["steps"], "converged": r["converged"],
                       "max_force": r["max_force"]}
        if verbose:
            print(f"  eps_yz = {sign * delta:+.4f}: {r['steps']} steps, "
                  f"converged={r['converged']}, max|F|={r['max_force']:.2e}, "
                  f"sigma_yz={sigma:+.4e}, dF/dtheta={g:+.6e}")

    scale = _HA_BOHR3_TO_GPA / (2.0 * delta)
    c44 = 0.5 * (points["plus"]["sigma_yz"] - points["minus"]["sigma_yz"]) * scale

    # Second difference of dF/dtheta over the shear strain. The Voigt factor
    # matches the one C44 carries: eps_4 = 2 eps_yz, so the curvature in the
    # tensorial strain is divided by four to become the Voigt constant.
    vol0 = points["zero"]["volume"]
    curvature = (points["plus"]["g"] - 2.0 * points["zero"]["g"]
                 + points["minus"]["g"]) / (delta ** 2)
    dc44 = 0.25 * curvature / vol0 * _HA_BOHR3_TO_GPA

    return {
        "theta": float(theta),
        "C44": float(c44),
        "dC44_dtheta": float(dc44),
        "g": {k: points[k]["g"] for k in points},
        "points": points,
    }


def verify_against_theta_sweep(
    config,
    symbols: Sequence[str],
    theta_step: float = 0.1,
    **kwargs,
) -> Dict[str, Any]:
    """Check ``dC44/d(theta)`` against re-relaxing at shifted theta.

    The reference is what :mod:`d3ft.calc.fit` uses as its gradient: C44
    recomputed from scratch at ``theta = +/- theta_step``, every geometry
    relaxed again. It shares no intermediate with the stationary route, so
    agreement is not circular.

    Pick ``theta_step`` with the reference's own noise in mind. At 0.02 two runs
    of the same AlP calculation disagreed by a factor of 1.9, because the
    difference being resolved was 0.8% of the stress baseline; 0.1 puts it near
    4%.
    """
    base = dc44_dtheta(config, symbols, theta=0.0, **kwargs)
    shifted = {}
    for sign, tag in ((+1, "plus"), (-1, "minus")):
        shifted[tag] = dc44_dtheta(
            config, symbols, theta=sign * theta_step, **kwargs
        )
    fd = (shifted["plus"]["C44"] - shifted["minus"]["C44"]) / (2.0 * theta_step)
    stationary = base["dC44_dtheta"]
    rel = abs(stationary - fd) / abs(fd) if fd else float("nan")
    return {
        "dC44_dtheta_stationary": stationary,
        "dC44_dtheta_finite_difference": fd,
        "relative_error": rel,
        "C44": base["C44"],
        "C44_plus": shifted["plus"]["C44"],
        "C44_minus": shifted["minus"]["C44"],
        "base": base,
    }
