"""K-pool projector construction must equal the historical eager expression.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` to exercise two
CPU shards on hosts without GPUs.
"""
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from d3ft import pseudopotential as d3pp
from d3ft.calc.ground_state import _projectors_reciprocal_kpool
from d3ft.parallel import kpool_mesh


g_vec = jnp.zeros((1, 1, 1, 3))
k_vec = jnp.asarray([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1],
])
mask = np.ones((1, 1, 1), dtype=bool)
positions = jnp.zeros((1, 3))
k_table = np.linspace(0.0, 2.0, 17)
tables = [(k_table, np.stack([np.exp(-k_table)]))]
pseudopot = SimpleNamespace(
    r_grid=[np.linspace(0.01, 2.0, 8)],
    nonlocal_beta_grid=[np.ones((1, 8))],
    nonlocal_angular_momentum=[np.asarray([0])],
)
crystal = SimpleNamespace(positions=positions)

gk = d3pp.gk_vectors(g_vec, k_vec, mask)
beta = d3pp.beta_grid(
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    gk,
    2.0,
    tables=tables,
)
eager = d3pp.projectors_reciprocal(
    positions, gk, pseudopot.nonlocal_angular_momentum, beta
)
sharded = _projectors_reciprocal_kpool(
    pseudopot, crystal, g_vec, k_vec, mask, 2.0, tables, kpool_mesh()
)
jax.block_until_ready(sharded)

np.testing.assert_allclose(sharded, eager, rtol=0.0, atol=0.0)
assert sum(shard.data.shape[0] for shard in sharded.addressable_shards) == len(k_vec)
print(
    "test_29: eager/k-pool projectors agree; local shapes",
    [shard.data.shape for shard in sharded.addressable_shards],
)
