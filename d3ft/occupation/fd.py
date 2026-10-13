"""Fermi--Dirac occupation coordinates with an implicit electron-number root.

The temperature here is in Hartree. Occupations include equal k-point weights,
as required by the current density API. This candidate changes occupation
coordinates only; it does not identify latent level parameters with KS energies.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def fermi_dirac_occupation(band_energies, num_electrons, epsilon,
                          spin_restricted=True, n_iter=50):
    """Return c*sigmoid((mu-z)/T); differentiate the charge equation implicitly.

    This API supports derivatives with respect to latent levels at fixed T,
    capacities, and electron number. n_iter is retained for API compatibility.
    """
    if not isinstance(epsilon, jax.core.Tracer) and epsilon <= 0:
        raise ValueError("FD occupation temperature must be positive")
    spin, nk, nb = band_energies.shape
    capacity = (2.0 if spin_restricted else 1.0) / nk
    if not 0 < num_electrons < spin * nk * nb * capacity:
        raise ValueError("FD coordinates need electrons strictly inside total capacity")
    levels = jnp.asarray(band_energies).reshape(-1)
    temp = jnp.asarray(epsilon, dtype=levels.dtype)
    cap = jnp.asarray(capacity, dtype=levels.dtype)

    @jax.custom_jvp
    def solve(z):
        lo = jax.lax.stop_gradient(jnp.min(z) - 100 * temp)
        hi = jax.lax.stop_gradient(jnp.max(z) + 100 * temp)

        def step(interval, _):
            low, high = interval
            mid = (low + high) / 2
            mass = jnp.sum(cap * jax.nn.sigmoid((mid - z) / temp))
            small = mass < num_electrons
            return (jnp.where(small, mid, low),
                    jnp.where(small, high, mid)), None

        (lo, hi), _ = jax.lax.scan(step, (lo, hi), None, length=200)
        return cap * jax.nn.sigmoid(((lo + hi) / 2 - z) / temp)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        (z,), (dz,) = primals, tangents
        f = solve(z)
        weights = f * (1 - f / cap) / temp
        total = jnp.sum(weights)
        # In exact arithmetic total>0 at finite T. Machine-saturated insulating
        # states can have total=0; their numerically resolvable level derivative
        # is zero. Record such cases separately in validation.
        dmu = jnp.sum(weights * dz) / jnp.maximum(total, 1e-250)
        return f, weights * (dmu - dz)

    return solve(levels).reshape(spin, nk, nb)
