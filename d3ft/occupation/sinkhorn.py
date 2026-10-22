"""Capacity-bounded, log-domain occupation balancing.

This file implements **Module 2 of D3FT v0.1 (D3FT-OT)**.

Reformulates the occupation-number assignment as a *discrete optimal-transport problem
with capacity constraints*:

* Sources: :math:`N_e` units of electronic mass (one per electron, possibly fractional).
* Sinks: bands, each with capacity :math:`c_n \\le 1` (spin-restricted: :math:`\\le 2`).
* Cost: band energy :math:`\\varepsilon_n`.

The one-bin entropy model yields capped exponential occupations:

.. math::

    f_n^{(\\epsilon)} \\;=\\; \\min\\{c_n, c_n\\exp[(\\mu-\\varepsilon_n)/\\epsilon]\\}.

Hard capacities make the exact map piecewise smooth, even at finite regularisation.
Its zero-regularisation limit is capacity-constrained filling, not the physical
finite-temperature Fermi--Dirac distribution. The implemented finite balancing
and charge-root iterations preserve forward feasibility but do not yet implement
the implicit derivative of the mass root; JAX gradient finiteness alone is not
a certificate of the physical occupation Jacobian.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def epsilon_schedule(
    step: int,
    epsilon_init: float,
    epsilon_final: float,
    total_steps: int,
    schedule: str = "log",
) -> float:
    r"""Anneal the entropic regularisation :math:`\epsilon_t`.

    Args:
        step: Current iteration index (0-based).
        epsilon_init: Initial regularisation :math:`\epsilon_0`.
        epsilon_final: Final regularisation (also a floor on the schedule).
        total_steps: Total number of optimisation steps.
        schedule: One of:

            * ``"log"``: :math:`\epsilon_t = \epsilon_0 / \log(t + e)`, clipped at
              ``epsilon_final``. **This schedule cannot reach its target.** It gives
              0.1447 at t = 1000, 0.1174 at t = 5000, 0.1040 at t = 15000; reaching
              ``epsilon_final = 1e-3`` would take :math:`t = e^{1000}`. Under it
              ``epsilon_final`` is decorative and the occupations stay smeared by
              ~0.12 Ha = 3.2 eV, which for AlP is *wider than the 1.6 eV band gap*.
              Measured cost on AlP: 0.053 Ha above the native occupation scheme.
            * ``"exp"``: exponential decay from ``epsilon_init`` to ``epsilon_final``.
              Reaches its target, and with ``epsilon_final = 3e-2`` closes 97% of
              the gap to the native scheme (-9.283403 Ha against -9.284819).
              **The implementation goes NaN and Adam then poisons every
              band-energy parameter.** Over three targets at 3000 steps the onset
              was eps = 0.0217, 0.0183, 0.0156, which looked like a floor near
              2e-2. It is not: the same ``epsilon_final = 3e-2`` that survives
              3000 steps goes NaN at 12000. The controlling quantity is
              ``band_energies / epsilon``, and the band-energy parameters grow
              with optimisation, so *any* epsilon fails given enough steps. Until
              this is diagnosed the Sinkhorn path is not usable for production
              runs; ``occupation.enabled = False`` (the native scheme) is, and it
              reproduces QE to 0.0011 Ry in energy and 2.5% in pressure. A
              stop_gradient on the bisection bracket was tried and did not help.
            * ``"constant"``: returns ``epsilon_init`` always.

    Returns:
        The regularisation value at iteration ``step``.

    Notes:
        The log schedule is the only one with a known theoretical convergence
        guarantee for vanishing-regularisation Sinkhorn (Bregman projection geometry).
    """
    if schedule == "constant":
        return float(epsilon_init)
    if schedule == "log":
        e = float(epsilon_init) / max(jnp.log(step + jnp.e), 1.0)
        return float(max(e, epsilon_final))
    if schedule == "exp":
        frac = step / max(total_steps, 1)
        return float(
            epsilon_init * (epsilon_final / epsilon_init) ** frac
        )
    raise ValueError(f"Unknown schedule: {schedule!r}")


def sinkhorn_occupation(
    band_energies: Float[Array, "spin kpt band"],
    num_electrons: float,
    epsilon: float,
    spin_restricted: bool = True,
    n_iter: int = 50,
) -> Float[Array, "spin kpt band"]:
    r"""Compute occupation numbers via capacity-bounded entropic OT.

    Applies finite Sinkhorn-style dual balancing in log space, followed by a
    scalar mass correction, to obtain occupations of the form

    .. math::

        f_n \;=\; \mathrm{clip}\bigl(\exp(-(\varepsilon_n - \alpha)/\epsilon),\,0,\,c\bigr),
        \qquad \sum_n f_n = N_e.

    The capacity :math:`c` is 1 for unrestricted spin and 2 for spin-restricted
    calculations.

    Args:
        band_energies: Single-particle band energies, shape ``[spin, kpt, band]``.
        num_electrons: Number of electrons :math:`N_e`.
        epsilon: Entropic regularisation strength :math:`\epsilon > 0`.
        spin_restricted: If True, the per-band capacity is 2; else 1.
        n_iter: Number of Sinkhorn iterations.

    Returns:
        Occupation tensor of shape ``[spin, kpt, band]`` with

        * each entry in :math:`[0, c]`,
        * total sum equal to ``num_electrons`` up to scalar-root roundoff.

    Notes:
        Implementation is fully in log-domain for stability as :math:`\epsilon \to 0`.
    """
    spin, kpt, band = band_energies.shape
    # Per-state Pauli capacity. The density pipeline weights every k-point
    # equally (sum over the kpt axis with no explicit 1/num_kpts factor) and the
    # total occupation is constrained to `num_electrons`. Each Bloch state may
    # therefore hold at most capacity/num_kpts, matching Jrystal's convention
    # (`simplex_projector` divides by num_kpts). Without the 1/num_kpts factor,
    # a k-mesh lets the optimiser pile electrons into a few lowest states far
    # beyond the physical bound, collapsing the energy below the true ground
    # state.
    capacity = (2.0 if spin_restricted else 1.0) / kpt
    n_bins = spin * kpt * band

    log_cost = band_energies.reshape(-1) / epsilon

    log_alpha = jnp.zeros(())
    log_beta = jnp.zeros(n_bins)

    log_capacity = jnp.log(capacity)
    log_Ne = jnp.log(num_electrons)

    def body(carry, _):
        log_alpha, log_beta = carry

        log_f = -log_cost + log_alpha + log_beta
        log_clipped = jnp.minimum(log_f, log_capacity)
        log_sum = jax.scipy.special.logsumexp(log_clipped)
        log_alpha_new = log_alpha + (log_Ne - log_sum)

        log_f = -log_cost + log_alpha_new + log_beta
        log_beta_new = jnp.minimum(0.0, log_capacity - (-log_cost + log_alpha_new))

        return (log_alpha_new, log_beta_new), None

    (log_alpha, log_beta), _ = jax.lax.scan(body, (log_alpha, log_beta), None, length=n_iter)

    # Final EXACT mass projection. The alternating Sinkhorn updates conflict near
    # the capacity boundary, so after a fixed `n_iter` the mass constraint
    # ``sum_n f_n = N_e`` is only approximately satisfied -- badly so in the
    # small-`epsilon` / small-gap regime (e.g. eps=1e-3, n_iter=50 can yield
    # ``sum(f)=12`` instead of 18 for GaAs). An under-normalised occupation
    # under-charges the density, under-screens the Hartree potential and produces
    # spurious over-bound states. We therefore solve for a single global potential
    # shift ``s`` such that ``sum_n min(exp(log_f_n + s), capacity) = N_e`` exactly.
    # The total mass is monotone increasing in ``s``, so a bisection converges to
    # machine precision; this guarantees electron-number conservation regardless
    # of ``epsilon`` or ``n_iter``.
    if n_bins * capacity < num_electrons - 1e-12:
        raise ValueError(
            f"Infeasible occupation problem: {n_bins} states x capacity "
            f"{capacity} = {n_bins * capacity} cannot hold {num_electrons} "
            "electrons. Increase empty_bands."
        )

    base = -log_cost + log_alpha + log_beta  # = log_f

    # ``exp`` before the clip overflows and the clip then hides it: the forward
    # value is right (``min(inf, capacity) == capacity``) while the reverse pass
    # is not. ``minimum`` sends a zero cotangent down the branch it did not
    # select, ``d(exp)/dx`` is ``inf`` there, and ``0 * inf`` is NaN. Since
    # ``exp`` is monotone, ``min(exp(x), c) == exp(min(x, log c))`` exactly, so
    # clipping the argument is the same function with nothing to overflow. This
    # is the same class of error as the earlier NaN cell gradients -- **the guard
    # has to sit on the argument, never on the result** -- in the one place it
    # had not been checked.
    def total_mass(s):
        return jnp.sum(jnp.exp(jnp.minimum(base + s, log_capacity)))

    def bisect(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_small = total_mass(mid) < num_electrons
        lo = jnp.where(too_small, mid, lo)
        hi = jnp.where(too_small, hi, mid)
        return (lo, hi), None

    # The bracket has to follow the data, not be a fixed +-60. ``base`` lives on
    # the scale of ``band_energies / epsilon``, which the optimiser is free to
    # make large: measured spans of ~1e4 are routine once the band-energy
    # parameters have grown. With a fixed +-60 bracket the root simply sits
    # outside it, ``total_mass`` stays below ``num_electrons`` at every midpoint,
    # the search returns the upper end, and the result is silently short of
    # charge -- by exactly 0.5 electrons in the case that first exposed this,
    # where total_mass(+60) = 7.5 and the root was near s = 1000. That
    # under-charges the density, under-screens Hartree, and lowers the energy, so
    # runs that lost the most charge looked like the best minima.
    #
    # Both ends are chosen analytically so the root is bracketed by construction:
    #   at ``hi`` even the smallest term is clipped to capacity, so
    #   mass = n_bins * capacity >= N_e;
    #   at ``lo`` every term is at most N_e/(e * n_bins), so mass <= N_e/e < N_e.
    # stop_gradient on the bracket. It is a search device, not physics: the
    # physical content is that ``s`` satisfies the constraint, and its derivative
    # belongs to the constraint, not to the mechanics of locating it. With the
    # old fixed +-60 bracket this was automatic, since constants carry no
    # gradient. Making the ends data-dependent (necessary for the forward value)
    # would otherwise route a gradient through max/min of ``base`` and 200
    # halvings, which is ill-conditioned; with the exp epsilon schedule it went
    # NaN at eps ~ 1.2e-3 and Adam then poisoned every band-energy parameter.
    lo0 = jax.lax.stop_gradient(jnp.log(num_electrons / n_bins) - jnp.max(base) - 1.0)
    hi0 = jax.lax.stop_gradient(log_capacity - jnp.min(base) + 1.0)
    (lo, hi), _ = jax.lax.scan(bisect, (lo0, hi0), None, length=200)
    s = 0.5 * (lo + hi)

    # Same reordering as in ``total_mass``, and for the same reason.
    f = jnp.exp(jnp.minimum(base + s, log_capacity))
    return f.reshape(spin, kpt, band)
