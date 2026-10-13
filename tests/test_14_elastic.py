"""Elastic-constant reduction: Voigt bookkeeping and the fixed-basis requirement.

The expensive half of `d3ft.calc.elastic` is five ground states. The half that is
easy to get wrong is the arithmetic on top of them -- the Voigt factor of two on
C44 above all -- and that half needs no DFT at all: feed the reduction the
stresses an ideal cubic solid would produce and check the constants come back.

Also pins the geometry helpers, since a silent error in the deformed cell would
show up only as a wrong elastic constant many GPU-minutes later.
"""

import numpy as np

from d3ft.calc.elastic import (
    RY_BOHR3_TO_GPA,
    _write_xyz,
    reduce_cubic,
)
from d3ft.calc.stress import _strained

BOHR = 0.529177210903


def cubic_stress(strain, c11, c12, c44):
    """sigma for an ideal cubic solid, in Voigt form, given a tensorial strain.

    sigma_xx = c11 eps_xx + c12 (eps_yy + eps_zz), cyclic; sigma_yz = c44 * 2 eps_yz.
    The factor of two on the shear is the whole point of the test.
    """
    e = np.asarray(strain, dtype=float)
    sigma = np.zeros((3, 3))
    diag = np.array([e[0, 0], e[1, 1], e[2, 2]])
    for i in range(3):
        others = sum(diag[j] for j in range(3) if j != i)
        sigma[i, i] = c11 * diag[i] + c12 * others
    for i, j in ((0, 1), (0, 2), (1, 2)):
        sigma[i, j] = sigma[j, i] = c44 * 2.0 * e[i, j]
    return sigma


def check_reduction_recovers_a_known_solid(failures):
    """Round-trip C11/C12/C44 through the finite-difference reduction."""
    c11, c12, c44 = 145.0, 62.0, 72.0
    delta = 5.0e-3

    e_xx = np.zeros((3, 3))
    e_xx[0, 0] = delta
    e_yz = np.zeros((3, 3))
    e_yz[1, 2] = e_yz[2, 1] = delta

    got = reduce_cubic(
        cubic_stress(e_xx, c11, c12, c44),
        cubic_stress(-e_xx, c11, c12, c44),
        cubic_stress(e_yz, c11, c12, c44),
        cubic_stress(-e_yz, c11, c12, c44),
        delta,
        unit=1.0,
    )
    for name, want in (("C11", c11), ("C12", c12), ("C44", c44),
                       ("B", (c11 + 2 * c12) / 3.0)):
        dev = abs(got[name] - want)
        print(f"  {name}: got {got[name]:.6f}, want {want:.6f}")
        if dev > 1e-9:
            failures.append(f"{name} off by {dev:.3e} on an exactly linear solid")


def check_voigt_factor_is_not_silently_absorbed(failures):
    """A reduction that dropped the 1/2 on C44 would still pass a symmetric test.

    So assert the asymmetry directly: doubling only the shear stiffness must
    double C44 and leave C11 and C12 untouched.
    """
    delta = 5.0e-3
    e_xx = np.zeros((3, 3))
    e_xx[0, 0] = delta
    e_yz = np.zeros((3, 3))
    e_yz[1, 2] = e_yz[2, 1] = delta

    def run(c44):
        return reduce_cubic(
            cubic_stress(e_xx, 145.0, 62.0, c44),
            cubic_stress(-e_xx, 145.0, 62.0, c44),
            cubic_stress(e_yz, 145.0, 62.0, c44),
            cubic_stress(-e_yz, 145.0, 62.0, c44),
            delta, unit=1.0,
        )

    one, two = run(72.0), run(144.0)
    if abs(two["C44"] - 2.0 * one["C44"]) > 1e-9:
        failures.append("C44 does not scale with the shear stiffness")
    if abs(two["C11"] - one["C11"]) > 1e-9 or abs(two["C12"] - one["C12"]) > 1e-9:
        failures.append("changing C44 leaked into C11 or C12")
    print(f"  C44 scales 72 -> 144 as {one['C44']:.3f} -> {two['C44']:.3f}")


def check_unit_constants(failures):
    """Ry/Bohr^3 is half Ha/Bohr^3; QE reports in the former, d3ft in the latter."""
    from d3ft.calc.elastic import _HA_BOHR3_TO_GPA
    if abs(RY_BOHR3_TO_GPA * 2.0 - _HA_BOHR3_TO_GPA) > 1e-9:
        failures.append("RY_BOHR3_TO_GPA is not half of _HA_BOHR3_TO_GPA")
    print(f"  Ha/Bohr^3 = {_HA_BOHR3_TO_GPA:.4f} GPa, "
          f"Ry/Bohr^3 = {RY_BOHR3_TO_GPA:.4f} GPa")


def check_strain_is_applied_as_a_cartesian_deformation(failures):
    """(I + eps) acting on the cell, with the atoms carried affinely."""
    h = 5.467 / BOHR / 2
    cell = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    pos = np.array([[0.0, 0.0, 0.0], [h / 2, h / 2, h / 2]])

    strain = np.zeros((3, 3))
    strain[0, 0] = 0.01
    new_cell, new_pos = _strained(cell, pos, strain)
    new_cell = np.asarray(new_cell)
    new_pos = np.asarray(new_pos)

    expected = cell @ (np.eye(3) + strain).T
    if np.abs(new_cell - expected).max() > 1e-12:
        failures.append("strained cell is not cell @ (I + eps).T")

    # Fractional coordinates must be preserved: that is what "clamped ion" means.
    frac_before = pos @ np.linalg.inv(cell)
    frac_after = new_pos @ np.linalg.inv(new_cell)
    dev = np.abs(frac_before - frac_after).max()
    print(f"  fractional coordinates preserved to {dev:.2e}")
    if dev > 1e-12:
        failures.append(f"fractional coordinates moved by {dev:.2e} under strain")

    # A pure volumetric strain must scale the volume by (1 + delta)^3.
    iso = np.eye(3) * 0.01
    vol0 = abs(np.linalg.det(cell))
    vol1 = abs(np.linalg.det(np.asarray(_strained(cell, pos, iso)[0])))
    want = vol0 * 1.01 ** 3
    if abs(vol1 - want) / want > 1e-12:
        failures.append("volumetric strain does not scale the volume by (1+d)^3")


def check_xyz_writer_round_trips(failures):
    """The deformed geometry reaches the solver through a file; check it survives."""
    import tempfile
    from pathlib import Path

    h = 5.467 / BOHR / 2
    cell = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    pos = np.array([[0.0, 0.0, 0.0], [h / 2, h / 2, h / 2]])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cell.xyz"
        _write_xyz(path, cell, pos, ["Al", "P"])
        text = path.read_text().splitlines()
        if text[0].strip() != "2":
            failures.append("xyz header does not declare 2 atoms")
        lattice = [float(x) for x in text[1].split('"')[1].split()]
        back = np.array(lattice).reshape(3, 3) / BOHR
        dev = np.abs(back - cell).max()
        print(f"  cell round-trips through extXYZ to {dev:.2e} Bohr")
        if dev > 1e-9:
            failures.append(f"cell changed by {dev:.2e} Bohr through the file")
        if not text[2].startswith("Al") or not text[3].startswith("P"):
            failures.append("species order not preserved")


def main():
    # d3ft runs in float64 everywhere (calc enables it through set_env_params),
    # but these checks call _strained directly, so nothing has enabled it yet.
    # Without this the geometry assertions run in float32, where a 5 Bohr cell
    # carries ~5e-6 of absolute error against a 1e-12 tolerance. The
    # fractional-coordinate check passes anyway because its round-trip cancels
    # the error -- which is exactly how a precision problem hides.
    import jax
    jax.config.update("jax_enable_x64", True)

    print("[test_14] elastic-constant reduction and strain geometry")
    failures = []
    for check in (
        check_reduction_recovers_a_known_solid,
        check_voigt_factor_is_not_silently_absorbed,
        check_unit_constants,
        check_strain_is_applied_as_a_cartesian_deformation,
        check_xyz_writer_round_trips,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_14_elastic")
        return 1
    print("[OK]  reduction recovers a known cubic solid; strain geometry is affine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
