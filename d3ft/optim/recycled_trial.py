"""Recycle an Armijo trial rate without changing the acceptance condition."""
from typing import Any, NamedTuple
import jax.numpy as jnp


class RecycledTrialState(NamedTuple):
    count: Any
    mu: Any
    nu: Any
    trial_rate: Any
    success_streak: Any


def advance_trial(rate, reductions, accepted, previous_streak, upper):
    floor = jnp.asarray(upper) * 2.0 ** -24
    clean = accepted & (reductions == 0)
    streak = jnp.where(clean, previous_streak + 1, 0)
    grow = streak >= 8
    proposal = jnp.where(grow, 2.0 * rate, rate)
    proposal = jnp.where(accepted, proposal, rate / 2.0)
    proposal = jnp.clip(proposal, floor, upper)
    return proposal, jnp.where(grow, 0, streak)
