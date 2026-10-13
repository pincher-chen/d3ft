"""Brillouin-zone meshes, and the point-group closure they have to satisfy.

D3FT differentiates the free energy with respect to a Cartesian strain, so the
free energy has to carry the crystal's point symmetry *as a function*, not only
in the value it happens to take. A k-mesh that is not closed under the point
group breaks that, and the breakage lands exactly where symmetry should have
forced a zero: the off-diagonal components of the stress tensor.

Jrystal samples with ``ase.dft.kpoints.monkhorst_pack``, whose even-n meshes are
half-shifted off Gamma. Taken in the **primitive** basis of an fcc cell that
shift is not symmetry-preserving: of the 48 cubic operations, 36 map the mesh
onto a different set of points. Measured consequence for AlP at 4x4x4, LDA-PZ,
cutoff 20 Ha, native occupations::

    mesh                nk    |sigma_offdiag|   as % of the diagonal
    ASE half-shifted    64    2.0178e-05        89.8%
    Gamma-centred       64    1.2690e-07         0.3%
    Gamma-centred      216    2.1465e-07         0.8%

Two orders of magnitude, from the mesh alone. The residual sits at the level of
the diagonal's own optimisation noise (std 1.5e-7 to 5.1e-7), i.e. consistent
with zero.

The half-shifted mesh is not *wrong* -- it converges the BZ integral faster at
fixed nk, which is why it is the usual choice, and codes that use it restore the
symmetry afterwards by symmetrising forces and stresses with the crystal point
group. D3FT does not do that, and for a differentiable code it should not: the
functional being differentiated is what has to be symmetric, not just its output.
Symmetrising the mesh instead costs a factor of 4 in nk (64 -> 256 at n=4), which
buys nothing over simply taking a Gamma-centred mesh one step denser, so
Gamma-centred is the default here.

The cost is real and worth stating: at fixed n the Gamma-centred mesh integrates
the BZ less well. For AlP at n=4 it gives -18.5346 Ry against the half-shifted
mesh's -18.5466 (QE: -18.5486); n=6 recovers -18.5463.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np


def cubic_point_group() -> np.ndarray:
    """The 48 signed permutation matrices, in Cartesian coordinates.

    Returns:
        Array of shape (48, 3, 3).
    """
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            op = np.zeros((3, 3))
            for row, col in enumerate(perm):
                op[row, col] = signs[row]
            ops.append(op)
    return np.asarray(ops)


def _fold_key(frac: np.ndarray) -> tuple:
    """Fold fractional coordinates into [0, 1) and quantise for set lookup.

    Folding into [0, 1) rather than (-1/2, 1/2] matters: on the half-open
    symmetric interval the points +1/2 and -1/2 are the same k-point with two
    representations, and a set membership test misses the pair. That artefact
    makes a Gamma-centred mesh -- which does contain 1/2 for even n -- look like
    it breaks 42 of the 48 operations when it in fact breaks none.
    """
    folded = np.mod(np.asarray(frac), 1.0)
    folded = np.where(np.abs(folded - 1.0) < 1e-9, 0.0, folded)
    return tuple(np.round(folded, 8) + 0.0)


def _as_triple(grid_sizes) -> np.ndarray:
    sizes = np.asarray(grid_sizes).reshape(-1)
    if sizes.size == 1:
        sizes = np.repeat(sizes, 3)
    return sizes.astype(int)


def k_mesh_fractional(
    grid_sizes, mesh: str = "gamma", fold: bool = True
) -> np.ndarray:
    """Uniform BZ mesh in fractional (primitive reciprocal) coordinates.

    Args:
        grid_sizes: One integer, or three.
        mesh: ``"gamma"`` for a Gamma-centred mesh, closed under the point group
            at every n. ``"mp"`` for the ASE half-shifted Monkhorst-Pack mesh,
            which is not -- see the module docstring.
        fold: Put the representatives in (-1/2, 1/2] rather than [0, 1). This is
            not cosmetic. D3FT masks on ``|G|`` -- a sphere fixed at the origin,
            not ``|G+k|`` -- so ``E(k)`` is *not* invariant under
            ``k -> k + G``: the representative chosen for each mesh point picks
            out a different set of plane waves. Representatives near the origin
            keep the sphere centred on the states it has to describe. It also
            makes the mesh as close to exactly closed as it can be; see
            :func:`exactly_broken_ops`.

    Returns:
        Array of shape (prod(grid_sizes), 3).
    """
    sizes = _as_triple(grid_sizes)
    if mesh == "gamma":
        axes = [np.arange(n) / n for n in sizes]
    elif mesh == "mp":
        axes = [(np.arange(n) + 0.5) / n - 0.5 for n in sizes]
    else:
        raise ValueError(f"unknown k-mesh convention {mesh!r}; use 'gamma' or 'mp'")
    if fold:
        axes = [np.where(a > 0.5, a - 1.0, a) for a in axes]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def k_vectors(
    cell_vectors, grid_sizes, mesh: str = "gamma", fold: bool = True
) -> np.ndarray:
    """Cartesian k-vectors, the drop-in for ``jrystal``'s ``k_vectors``."""
    cell = np.asarray(cell_vectors)
    recip = 2 * np.pi * np.linalg.inv(cell).T
    return k_mesh_fractional(grid_sizes, mesh, fold) @ recip


def exactly_broken_ops(
    k_frac: np.ndarray,
    cell_vectors,
    ops: Sequence[np.ndarray] | None = None,
) -> List[int]:
    """Closure of the mesh *as stored*, with no folding modulo the reciprocal lattice.

    :func:`symmetry_broken_ops` asks whether the mesh maps onto itself as a set of
    k-points. That is the right question for a code whose basis is ``|G+k|``
    bounded, where ``k`` and ``k+G`` describe the same states. D3FT's basis is a
    ``|G|`` sphere fixed at the origin, so they do not: the two representatives
    select different plane waves and give different energies. The condition that
    actually makes the free energy point-group symmetric is closure of the stored
    representatives, which is what this measures.

    No box-shaped mesh in an fcc primitive basis satisfies it. Measured, folded
    into (-1/2, 1/2]: 36/48 broken for odd ``n``, 42/48 for even, at every ``n``
    from 3 to 8. Closure under negation and permutation of the axis sets -- which
    odd ``n`` does have -- is not the relevant property, because the point group
    acts in *Cartesian* coordinates and its operations in this basis are integer
    matrices that mix the components.

    It does not appear to matter. Once the mesh is closed modulo the reciprocal
    lattice, ``|sigma_offdiag|`` falls below the *standard deviation* of the
    diagonal at every ``n`` measured, with no odd/even pattern -- i.e. consistent
    with the zero cubic symmetry requires. Kept as a diagnostic, not a
    requirement.
    """
    cell = np.asarray(cell_vectors)
    recip = 2 * np.pi * np.linalg.inv(cell).T
    recip_inv = np.linalg.inv(recip)
    k_frac = np.asarray(k_frac)
    k_cart = k_frac @ recip
    members = {tuple(np.round(x, 8) + 0.0) for x in k_frac}
    ops = cubic_point_group() if ops is None else np.asarray(ops)

    broken = []
    for index, op in enumerate(ops):
        image = (k_cart @ op.T) @ recip_inv
        if any(tuple(np.round(v, 8) + 0.0) not in members for v in image):
            broken.append(index)
    return broken


def symmetry_broken_ops(
    k_frac: np.ndarray,
    cell_vectors,
    ops: Sequence[np.ndarray] | None = None,
) -> List[int]:
    """Indices of point-group operations that do not map the mesh onto itself.

    A k-mesh is acceptable for strain derivatives only when this is empty.

    Args:
        k_frac: Mesh in fractional coordinates, shape (nk, 3).
        cell_vectors: Real-space lattice vectors, rows.
        ops: Cartesian operations to test; defaults to the full cubic group.

    Returns:
        Indices into ``ops`` of the operations that break closure.
    """
    cell = np.asarray(cell_vectors)
    recip = 2 * np.pi * np.linalg.inv(cell).T
    recip_inv = np.linalg.inv(recip)
    k_frac = np.asarray(k_frac)
    k_cart = k_frac @ recip
    members = {_fold_key(x) for x in k_frac}
    ops = cubic_point_group() if ops is None else np.asarray(ops)

    broken = []
    for index, op in enumerate(ops):
        image = (k_cart @ op.T) @ recip_inv
        if any(_fold_key(v) not in members for v in image):
            broken.append(index)
    return broken
