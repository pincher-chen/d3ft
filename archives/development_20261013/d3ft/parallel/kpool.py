"""k-point pool parallelism -- the level that speeds up a *single* ground state.

Quantum ESPRESSO parallelises over images, k-point pools, band groups and plane
waves, in that order of increasing communication. The same hierarchy maps onto
this code, and the levels are not interchangeable:

* **Image level** -- independent geometries: the 6N displaced cells behind a
  force-constant matrix, the four strained cells behind a set of elastic
  constants. Handled by ``workflows/parallel/``. It cannot touch an internal
  relaxation, where each step needs the geometry the previous one produced.
* **k-point pools** -- *this module*. Splits the 216 k-points of a Gamma 6x6x6
  mesh across devices, which makes each individual ground state faster and so
  shortens the sequential chain itself. This is the level Amdahl's law demands
  once the embarrassingly parallel parts are done.
* **Band groups** -- the bands at one k-point are coupled through the Stiefel
  orthonormality constraint, so splitting them means a distributed QR every
  step. High communication, low return.
* **Plane waves / FFT grid** -- heaviest communication; QE reaches for it last
  and so should this.

The energy has exactly the pool structure. Kinetic energy, the nonlocal
projectors and the per-band FFTs are independent per k; they are summed into one
density; Hartree, XC and the local potential then act on that density and know
nothing about k. **One collective per evaluation**, on the density.

Sizes for AlP at Gamma 6x6x6, cutoff 20 Ha: the nonlocal projector array is
``[216, 4, 3, 24, 24, 24]`` complex128, 573 MB, and it is entirely per-k. Over
eight devices that is 72 MB each, which is also why a cutoff-50 run that ran out
of memory on one device is not intrinsically too large.

**Why the FFT has to be swapped out.** ``jrystal._src.pw`` imports ``ifftn3d``
from its ``spmd`` package, a custom-partitioned FFT registered through
``xb.register_plugin_callbacks``. That registration does not take effect for the
CUDA plugin in the JAX build here, so merely making a second device visible fails
with ``No registered implementation for custom call to
CustomSPMDPartitioningJrystal``. The custom partitioner exists to split the 3D
grid across devices -- and grid splitting is precisely what k-pool parallelism
does *not* want. Each device holds whole 3D grids for its own subset of k-points,
so a plain local ``jnp.fft.ifftn`` over the last three axes is both correct and
what the hardware wants; k is a leading axis and GSPMD treats it as a batch
dimension with nothing to negotiate.

So the fix is to stop using the custom partitioner rather than to repair it.
:func:`plain_fft_patch` swaps in the plain transform for the duration of a block.
It patches the attribute on ``jrystal._src.pw`` rather than editing jrystal,
which stays reference-only.

**Status: implemented, correct, and not worth using. Default off.**

It works. AlP at a Gamma 4x4x4 mesh, 600 steps, gives -18.48564538 Ry on one
device and the same number to eight decimals on two and four -- the collectives
are in the right places. It is simply slower:

| devices | k_pool | time |
|---|---|---|
| 1 | off | **71.5 s** |
| 2 | on | 78.0 s |
| 4 | on | 80.7 s |

The per-k work is too small to pay for the communication. Sixty-four k-points of
1243 plane waves and 4 bands is not much arithmetic, while the density
all-reduce over a 24^3 grid happens every step and Hartree, XC and the local
potential stay replicated. QE's pools pay off because its systems bring
thousands of k-points and tens of thousands of plane waves each; ours do not
reach the point where splitting them helps.

**The 216-point production mesh cannot even compile it**, for a reason worth
recording separately: ``update``'s jit captures the per-k arrays as compile-time
constants -- 2.87 GB of them, the nonlocal projectors alone being
``[216, 4, 3, 24, 24, 24]`` complex128, 573 MB -- and partitioning pushes the
compilation result past XLA's 2 GB protobuf limit. The failure surfaces as
``UnspecifiedValue has no attribute addressable_devices_indices_map`` from an
unrelated part of JAX, which is what made it take so long to find; the warning
naming the real cause was three lines above and I was filtering it out. Passing
those arrays as arguments instead of closing over them would fix the compile,
shorten compilation, and reduce memory whether or not anything is sharded. That
is worth doing on its own merits, and it is not what makes a k-pool pay.

Two things did have to be fixed before any of this could be measured, and both
are real:

* ``jrystal._src.spmd.custom_sharding`` calls ``jax.tree_map``, removed in JAX
  0.6.0. Reached only when GSPMD has real partitioning to do, so nothing sharded
  means nothing broken. See :func:`jrystal_jax_compat`.
* ``jrystal._src.pw`` imports a custom-partitioned ``ifftn3d`` whose CUDA
  partitioner is unregistered here. It exists for grid sharding, the opposite of
  what a pool wants. See :func:`plain_fft_patch`.

Keep them: they are what let ``calc`` run at all with more than one device
visible, which the image-level parallelism in ``workflows/parallel/`` needs.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _plain_ifftn3d(x):
    """``ifftn`` over the last three axes, with no custom partitioning.

    Identical to the function jrystal's custom partitioner wraps; the wrapper is
    what breaks, not the transform.
    """
    return jnp.fft.ifftn(x, axes=range(-3, 0))


def _local_occupied_density(coeff, vol, occupation):
    """Form one pool's density and reduce it before leaving ``shard_map``.

    Keeping the absolute-square tensor inside the map is essential.  If the
    sharded wavefunction is returned to an enclosing replicated computation,
    GSPMD may all-gather ``[spin, k, band, x, y, z]`` before the k reduction.
    GaP k=12 then asks every GPU for a 44.65-GiB float64 buffer.  Here each GPU
    materialises only its local k batch and the only value crossing the map
    boundary is the small, replicated ``[spin, x, y, z]`` density.
    """
    wave = _plain_ifftn3d(coeff)
    wave *= np.prod(coeff.shape[-3:]) / jnp.sqrt(vol)
    dens = jnp.real(wave * jnp.conj(wave))
    local = jnp.einsum("skb...,skb->s...", dens, occupation)
    return jax.lax.psum(local, "k")


def _local_kinetic(g_vector_grid, kpts, coeff, occupation):
    """Evaluate the kinetic scalar within each k pool, then sum pools."""
    dim = g_vector_grid.shape[-1]
    g = jnp.expand_dims(g_vector_grid, axis=range(3))
    k = jnp.expand_dims(kpts, axis=[0] + [i + 2 for i in range(dim + 1)])
    gk2 = jnp.sum((g + k) ** 2, axis=-1)
    orbital = jnp.sum(
        gk2 * jnp.real(coeff * jnp.conj(coeff)),
        axis=range(3, dim + 3),
    )
    return jax.lax.psum(jnp.sum(orbital * occupation) / 2, "k")


def jrystal_jax_compat():
    """Restore ``jax.tree_map``, which JAX 0.6.0 removed and jrystal still calls.

    ``jrystal._src.spmd.custom_sharding._infer_sharding_from_operands`` uses
    ``jax.tree_map``. That function is only reached when GSPMD actually has to
    partition something, so nothing sharded means nothing broken -- which is
    exactly why an unsharded ``calc`` runs happily on eight devices and the same
    call with one array sharded does not.

    The failure it produces is badly misleading. The partitioner raises inside
    XLA's compilation callback, the sharding is left unresolved, and JAX reports
    ``UnspecifiedValue has no attribute addressable_devices_indices_map`` from
    somewhere else entirely. Six different ways of expressing the sharding all
    failed with that message before the real line came out of a stack trace.

    Idempotent, and a no-op on JAX versions that still have the attribute.
    """
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    return hasattr(jax, "tree_map")


@contextlib.contextmanager
def plain_fft_patch(*, k_pool=False):
    """Use the plain FFT inside this block.

    Required before any multi-device work: jrystal's custom-partitioned FFT
    raises on CUDA as soon as more than one device is visible. See the module
    docstring for why replacing it is the right move rather than a workaround.
    """
    jrystal_jax_compat()
    from jrystal._src import energy, pw
    from d3ft import pseudopotential as d3pp

    original = pw.ifftn3d
    original_density_grid = pw.density_grid
    original_kinetic = energy.kinetic
    original_energy_nonlocal = d3pp.energy_nonlocal
    if k_pool:
        from jax.experimental.shard_map import shard_map
        # GSPMD's FFT lowering can all-gather even an untransformed batch
        # axis. Make the local FFT explicit; its transpose then stays local
        # too. Density reductions outside this map retain their collectives.
        spec = P(None, 'k', None, None, None, None)
        local_fft = shard_map(_plain_ifftn3d, mesh=kpool_mesh(),
                              in_specs=spec, out_specs=spec, check_rep=False)

        def transform(x):
            if x.ndim != 6:
                raise ValueError('k-pool wave FFT expects [spin,k,band,x,y,z]')
            return local_fft(x)

        pw.ifftn3d = transform

        # Patch the complete occupied-density operation as well as its FFT.
        # Patching only the FFT leaves the absolute square and k reduction in a
        # replicated outer graph, which can recreate the full k batch per GPU.
        local_density = shard_map(
            # The VJP otherwise saves two 10.81-GiB local complex FFT buffers
            # until the backward sweep at GaP k=12. Recomputing the FFT there
            # is substantially cheaper than keeping those activations live.
            jax.checkpoint(_local_occupied_density),
            mesh=kpool_mesh(),
            in_specs=(spec, P(), P(None, "k", None)),
            out_specs=P(None, None, None, None),
            check_rep=False,
        )

        def density_grid(coeff, vol, occupation=None):
            if occupation is None:
                # Jrystal's no-occupation API deliberately returns every
                # orbital density, so there is no k reduction to localise.
                return original_density_grid(coeff, vol, occupation)
            if coeff.ndim != 6 or occupation.ndim != 3:
                raise ValueError(
                    "k-pool density expects coeff [spin,k,band,x,y,z] and "
                    "occupation [spin,k,band]"
                )
            return local_density(coeff, vol, occupation)

        pw.density_grid = density_grid

        # Jrystal's kinetic expression also forms a full-grid float64
        # ``|c|^2`` tensor before reducing k. Keep that tensor pool-local for
        # the same reason as the real-space density above.
        local_kinetic = shard_map(
            jax.checkpoint(_local_kinetic),
            mesh=kpool_mesh(),
            in_specs=(P(None, None, None, None), P("k", None), spec,
                      P(None, "k", None)),
            out_specs=P(),
            check_rep=False,
        )

        def kinetic(g_vector_grid, kpts, coeff, occupation=None):
            if occupation is None:
                return original_kinetic(g_vector_grid, kpts, coeff, occupation)
            return local_kinetic(g_vector_grid, kpts, coeff, occupation)

        energy.kinetic = kinetic

        # Restricting full-grid coefficients to the cutoff sphere is a gather.
        # If it sits in the replicated outer graph, reverse mode implements its
        # transpose as a scatter-add into a *global* FFT-sized zero tensor.  For
        # GaP k=12 that hidden cotangent is the remaining 44.65-GiB allocation.
        # Keeping the complete nonlocal contraction inside the map makes both
        # the gather and its transpose operate on the local 216-point batch.
        def energy_nonlocal(
            coeff, projectors, vol, occupation, d_matrix, freq_mask=None
        ):
            if freq_mask is None:
                return original_energy_nonlocal(
                    coeff, projectors, vol, occupation, d_matrix,
                    freq_mask=freq_mask,
                )
            # ``mask_indices`` intentionally uses NumPy, so the mask must stay
            # a static closure while shard_map traces the local contraction.
            def nonlocal_body(c, p, v, o, d):
                value = original_energy_nonlocal(
                    c, p, v, o, d, freq_mask=freq_mask
                )
                return jax.lax.psum(value, "k")

            local_energy_nonlocal = shard_map(
                jax.checkpoint(nonlocal_body),
                mesh=kpool_mesh(),
                in_specs=(
                    spec,
                    P("k", None, None, None),
                    P(),
                    P(None, "k", None),
                    P(None, None),
                ),
                out_specs=P(),
                check_rep=False,
            )
            return local_energy_nonlocal(
                coeff, projectors, vol, occupation, d_matrix
            )

        d3pp.energy_nonlocal = energy_nonlocal
    else:
        pw.ifftn3d = _plain_ifftn3d
    try:
        yield
    finally:
        pw.ifftn3d = original
        pw.density_grid = original_density_grid
        energy.kinetic = original_kinetic
        d3pp.energy_nonlocal = original_energy_nonlocal


def kpool_mesh(devices: Optional[Sequence] = None) -> Mesh:
    """A one-dimensional device mesh named ``"k"``.

    Args:
        devices: devices to use; all local devices by default.

    Returns:
        The mesh to build shardings against.
    """
    devices = list(jax.devices()) if devices is None else list(devices)
    return Mesh(np.asarray(devices), ("k",))


def kpool_sharding(mesh: Mesh, ndim: int, k_axis: int = 1) -> NamedSharding:
    """Shard ``k_axis`` over the mesh and replicate everything else.

    Args:
        mesh: from :func:`kpool_mesh`.
        ndim: rank of the array being sharded.
        k_axis: which axis carries k. 1 for ``[spin, kpt, n_g, band]``
            coefficients, 0 for arrays that lead with k.

    Returns:
        The sharding to pass to ``jax.device_put``.
    """
    spec = [None] * ndim
    spec[k_axis] = "k"
    return NamedSharding(mesh, P(*spec))


def shard_over_k(tree, mesh: Mesh, k_axis: int = 1, k_size: Optional[int] = None):
    """Place every array in ``tree`` whose ``k_axis`` matches ``k_size`` onto the mesh.

    Arrays that do not carry a k axis of the right length are left replicated,
    which is what the density-side quantities need.

    Args:
        tree: pytree of arrays.
        mesh: from :func:`kpool_mesh`.
        k_axis: axis carrying k.
        k_size: expected length of that axis; inferred from the first array with
            enough dimensions when omitted.

    Returns:
        The tree with eligible leaves sharded.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if k_size is None:
        for leaf in leaves:
            arr = jnp.asarray(leaf)
            if arr.ndim > k_axis:
                k_size = arr.shape[k_axis]
                break
    n_dev = mesh.devices.size
    if k_size is None or k_size % n_dev:
        raise ValueError(
            f"k-point count {k_size} is not divisible by {n_dev} devices. "
            "Pick a mesh size that divides it, or a k-grid that the device "
            "count divides -- a Gamma 6x6x6 mesh gives 216, which suits 1, 2, "
            "3, 4, 6 or 8 devices."
        )
    out = []
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.ndim > k_axis and arr.shape[k_axis] == k_size:
            out.append(jax.device_put(arr, kpool_sharding(mesh, arr.ndim, k_axis)))
        else:
            out.append(arr)
    return jax.tree_util.tree_unflatten(treedef, out)
