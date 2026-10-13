"""The fused k-pool density must match Jrystal in value and reverse mode.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` so this also
checks the cross-pool sum on hosts without GPUs.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from d3ft.parallel import kpool_mesh, kpool_sharding, plain_fft_patch
from d3ft import pseudopotential as d3pp
from jrystal._src import energy, pw


rng = np.random.default_rng(3109)
coeff_host = (
    rng.normal(size=(1, 4, 3, 5, 4, 3))
    + 1j * rng.normal(size=(1, 4, 3, 5, 4, 3))
).astype(np.complex128)
occupation_host = rng.uniform(size=(1, 4, 3))
vol = 123.4
g_vector_grid = jnp.asarray(rng.normal(size=(5, 4, 3, 3)))
kpts_host = rng.normal(size=(4, 3))
mask = np.zeros((5, 4, 3), dtype=bool)
mask.reshape(-1)[:17] = True
projectors_host = (
    rng.normal(size=(4, 2, 2, 17))
    + 1j * rng.normal(size=(4, 2, 2, 17))
).astype(np.complex128)
d_matrix = jnp.asarray([[1.2, 0.1], [0.1, 0.8]])

# Reference is the unmodified Jrystal expression on host arrays.
with plain_fft_patch(k_pool=False):
    reference = pw.density_grid(jnp.asarray(coeff_host), vol, jnp.asarray(occupation_host))
    reference_grad = jax.grad(
        lambda c: jnp.sum(pw.density_grid(c, vol, jnp.asarray(occupation_host)) ** 2)
    )(jnp.asarray(coeff_host))
    reference_kinetic = energy.kinetic(
        g_vector_grid, jnp.asarray(kpts_host), jnp.asarray(coeff_host),
        jnp.asarray(occupation_host),
    )
    reference_kinetic_grad = jax.grad(
        lambda c: energy.kinetic(
            g_vector_grid, jnp.asarray(kpts_host), c,
            jnp.asarray(occupation_host),
        )
    )(jnp.asarray(coeff_host))
    reference_nonlocal = d3pp.energy_nonlocal(
        jnp.asarray(coeff_host), jnp.asarray(projectors_host), vol,
        jnp.asarray(occupation_host), d_matrix, freq_mask=mask,
    )
    reference_nonlocal_grad = jax.grad(
        lambda c: d3pp.energy_nonlocal(
            c, jnp.asarray(projectors_host), vol,
            jnp.asarray(occupation_host), d_matrix, freq_mask=mask,
        )
    )(jnp.asarray(coeff_host))

mesh = kpool_mesh()
coeff = jax.device_put(jnp.asarray(coeff_host), kpool_sharding(mesh, 6, 1))
occupation = jax.device_put(jnp.asarray(occupation_host), kpool_sharding(mesh, 3, 1))
kpts = jax.device_put(jnp.asarray(kpts_host), kpool_sharding(mesh, 2, 0))
projectors = jax.device_put(
    jnp.asarray(projectors_host), kpool_sharding(mesh, 4, 0)
)
with plain_fft_patch(k_pool=True):
    density = jax.jit(lambda c, o: pw.density_grid(c, vol, o))(coeff, occupation)
    grad = jax.jit(jax.grad(
        lambda c, o: jnp.sum(pw.density_grid(c, vol, o) ** 2), argnums=0
    ))(coeff, occupation)
    kinetic = jax.jit(energy.kinetic)(g_vector_grid, kpts, coeff, occupation)
    kinetic_grad = jax.jit(jax.grad(energy.kinetic, argnums=2))(
        g_vector_grid, kpts, coeff, occupation
    )
    nonlocal_energy = jax.jit(
        lambda c, p, o: d3pp.energy_nonlocal(
            c, p, vol, o, d_matrix, freq_mask=mask
        )
    )(coeff, projectors, occupation)
    nonlocal_grad = jax.jit(jax.grad(
        lambda c, p, o: d3pp.energy_nonlocal(
            c, p, vol, o, d_matrix, freq_mask=mask
        ), argnums=0
    ))(coeff, projectors, occupation)

np.testing.assert_allclose(density, reference, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(grad, reference_grad, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(kinetic, reference_kinetic, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(
    kinetic_grad, reference_kinetic_grad, rtol=1e-12, atol=1e-12
)
np.testing.assert_allclose(nonlocal_energy, reference_nonlocal, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(
    nonlocal_grad, reference_nonlocal_grad, rtol=1e-12, atol=1e-12
)
assert all(shard.data.shape[1] == 2 for shard in grad.addressable_shards)
print("test_31: k-pool density value/gradient match; local k=2")
