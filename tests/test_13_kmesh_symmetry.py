"""The k-mesh must be closed under the crystal point group.

Not a stylistic preference. D3FT differentiates the free energy with respect to a
Cartesian strain, so the free energy has to be symmetric *as a function* of that
strain -- otherwise the derivative is faithful (finite differences agree to 5e-9)
and the answer is still wrong, because the functional being differentiated does
not carry the symmetry the crystal does.

Where it surfaces is the off-diagonal stress, which cubic symmetry forces to
zero. Measured for AlP at 4x4x4, LDA-PZ, cutoff 20 Ha, native occupations:

    ASE half-shifted   |sigma_offdiag| = 2.0178e-05   89.8% of the diagonal
    Gamma-centred      |sigma_offdiag| = 1.2690e-07    0.3%

Pure geometry -- no SCF, no GPU -- so it is cheap enough to guard the convention
on every run.
"""

import numpy as np

from d3ft.grid import (
    cubic_point_group,
    k_mesh_fractional,
    k_vectors,
    symmetry_broken_ops,
)

BOHR = 0.529177210903


def fcc_cell(a_angstrom=5.467):
    """Primitive fcc lattice vectors, rows, in Bohr."""
    h = a_angstrom / BOHR / 2
    return np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])


def check_point_group(failures):
    ops = cubic_point_group()
    if ops.shape != (48, 3, 3):
        failures.append(f"point group has shape {ops.shape}, expected (48, 3, 3)")
        return
    if len({op.tobytes() for op in ops}) != 48:
        failures.append("point group contains duplicate operations")
    worst = max(float(np.abs(op @ op.T - np.eye(3)).max()) for op in ops)
    if worst > 1e-12:
        failures.append(f"point group elements are not orthogonal (max dev {worst:.2e})")
    print(f"  point group: 48 distinct orthogonal elements (max dev {worst:.1e})")


def check_gamma_is_closed(failures):
    """A Gamma-centred mesh is closed at every n, even or odd."""
    cell = fcc_cell()
    for n in (3, 4, 6, 8):
        k_frac = k_mesh_fractional(n, "gamma")
        if len(k_frac) != n ** 3:
            failures.append(f"gamma mesh n={n} has {len(k_frac)} points, expected {n**3}")
        broken = symmetry_broken_ops(k_frac, cell)
        print(f"  gamma  n={n}: nk={len(k_frac):4d}  broken={len(broken):2d}/48")
        if broken:
            failures.append(f"gamma mesh n={n} breaks {len(broken)}/48 operations")


def check_half_shift_breaks(failures):
    """The defect this module exists to prevent, pinned so it cannot come back.

    36 of 48 for every even n tested. Odd n is excluded: ASE's shift vanishes
    there, so the mesh is Gamma-centred and symmetric.
    """
    cell = fcc_cell()
    for n in (4, 6, 8):
        broken = symmetry_broken_ops(k_mesh_fractional(n, "mp"), cell)
        print(f"  mp     n={n}: nk={n**3:4d}  broken={len(broken):2d}/48")
        if len(broken) != 36:
            failures.append(
                f"half-shifted mesh n={n} breaks {len(broken)}/48, expected 36 -- "
                "if this dropped to 0 the convention changed underneath the test"
            )


def check_folding_convention(failures):
    """A Gamma-centred mesh contains +1/2 for even n, and -1/2 is the same point.

    Folding into (-1/2, 1/2] leaves those with two different representations, and
    a set-membership test then reports 42 of 48 operations broken for a mesh that
    breaks none. This guards the [0, 1) folding that avoids the artefact.
    """
    cell = fcc_cell()
    k_frac = k_mesh_fractional(4, "gamma")
    if not np.any(np.isclose(k_frac, 0.5)):
        failures.append("gamma mesh n=4 has no 1/2 component; the guard is vacuous")
    broken = symmetry_broken_ops(k_frac, cell)
    if broken:
        failures.append(
            f"gamma mesh n=4 reports {len(broken)}/48 broken -- folding artefact is back"
        )
    print("  folding: 1/2 present and closure still 0/48")


def check_k_vectors_agree(failures):
    cell = fcc_cell()
    recip = 2 * np.pi * np.linalg.inv(cell).T
    for mesh in ("gamma", "mp"):
        dev = float(
            np.abs(k_vectors(cell, 4, mesh) - k_mesh_fractional(4, mesh) @ recip).max()
        )
        if dev > 1e-12:
            failures.append(f"k_vectors({mesh}) disagrees with the fractional mesh ({dev:.2e})")
    print("  k_vectors matches the fractional mesh for both conventions")


def check_mp_matches_ase(failures):
    """"mp" has to stay comparable to Jrystal's sampling to be a fair control."""
    try:
        from ase.dft.kpoints import monkhorst_pack
    except ImportError:
        print("  ase unavailable; skipping the Jrystal-parity check")
        return
    dev = float(np.abs(k_mesh_fractional(4, "mp") - monkhorst_pack((4, 4, 4))).max())
    if dev > 1e-12:
        failures.append(f"'mp' no longer reproduces ase.monkhorst_pack ({dev:.2e})")
    print(f"  'mp' reproduces ase.monkhorst_pack (max dev {dev:.1e})")


def check_symmetrisation_cost(failures):
    """Records why Gamma-centred is the default rather than a symmetrised shift.

    Closing the half-shifted mesh under the point group takes 64 points to 256 --
    more than a Gamma-centred 6x6x6 (216), which already brings the off-diagonal
    stress down to 2.1e-7. The cheaper fix is the denser symmetric mesh.
    """
    cell = fcc_cell()
    recip = 2 * np.pi * np.linalg.inv(cell).T
    recip_inv = np.linalg.inv(recip)
    k_cart = k_mesh_fractional(4, "mp") @ recip

    orbit = set()
    for vec in k_cart:
        for op in cubic_point_group():
            frac = np.mod((vec @ op.T) @ recip_inv, 1.0)
            frac = np.where(np.abs(frac - 1.0) < 1e-9, 0.0, frac)
            orbit.add(tuple(np.round(frac, 8) + 0.0))
    print(f"  symmetrising the n=4 half-shifted mesh: 64 -> {len(orbit)} k-points")
    if len(orbit) != 256:
        failures.append(f"symmetrised half-shifted mesh has {len(orbit)} points, expected 256")


def main():
    print("[test_13] k-mesh closure under the cubic point group")
    failures = []
    for check in (
        check_point_group,
        check_gamma_is_closed,
        check_half_shift_breaks,
        check_folding_convention,
        check_k_vectors_agree,
        check_mp_matches_ase,
        check_symmetrisation_cost,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_13_kmesh_symmetry")
        return 1
    print("[OK]  gamma-centred meshes are point-group closed; the half shift is not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
