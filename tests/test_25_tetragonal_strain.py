"""C11 and C12, and the three conversion factors that must not be confused.

``screen`` measures three independent combinations under three strains, and each
carries a different denominator:

    isotropic    eps = d I                          B          d2E / (9 V0)
    tetragonal   eps = diag(d, d, (1+d)^-2 - 1)     C11 - C12  d2E / (6 V0)
    shear        eps_yz = eps_zy = d                C44        d2E / (4 V0)

None is interchangeable with another. The shear branch once used the isotropic
factor and returned C44 scaled by 9/4, which is why every one of them is pinned
here against a synthetic quadratic energy rather than left to review.

The tetragonal strain earns its place twice over. It needs no relaxation -- both
sublattices stay on S4 axes, and S4 sends a displacement to minus itself, so
symmetry forbids any internal shift exactly as cubic symmetry does at zero
strain. So C11 and C12, like a0 and B, are available on systems whose forces
cannot be trusted; C44 alone is not. And it is the only strain that separates
C11 from C12 at all: the isotropic screen sees only the combination C11 + 2 C12,
so B and C44 together leave the pair underdetermined.

No DFT: ``ground_state_at_strain``, ``relax_internal_coordinates`` and
``sensitivity`` are replaced, so what is exercised is the strain construction,
the arithmetic, and the assembly in ``cubic_constants``.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3ft.calc import screening  # noqa: E402
from d3ft.calc.screening import cubic_constants  # noqa: E402

_HA_BOHR3_TO_GPA = screening._HA_BOHR3_TO_GPA
_CELL = 10.0
_VOL = _CELL ** 3


class _Cfg:
    def __init__(self):
        self.xc = "gga_x_pbe+gga_c_pbe"
        self.d3ft = {"xc_params": None}


def _fake_out(energy):
    crystal = types.SimpleNamespace(cell_vectors=np.eye(3) * _CELL)
    return types.SimpleNamespace(
        total_energy=energy, freq_mask=np.ones((2, 2, 2), dtype=bool),
        crystal=crystal,
    )


def _install(energy_of_d):
    """Replace the expensive calls; return the strain matrices actually built."""
    seen = []

    def fake_gs(config, strain, symbols, **kw):
        strain = np.asarray(strain, dtype=float)
        seen.append(strain)
        # Recover the scalar parameter from the strain the branch constructed.
        d = float(strain[0, 0])
        return _fake_out(energy_of_d(d))

    def fake_relax(config, symbols, strain=None, **kw):
        strain = np.asarray(strain, dtype=float)
        seen.append(strain)
        d = float(strain[1, 2])
        return {"ground_state": _fake_out(energy_of_d(d)), "converged": True,
                "max_force": 1.0e-5, "steps": 3}

    screening.ground_state_at_strain = fake_gs
    screening.relax_internal_coordinates = fake_relax
    screening.sensitivity = lambda out, xc_base, nc: np.zeros(
        len(screening.parameter_names(xc_base, nc))
    )
    return seen


def _run(mode, k, delta=0.02):
    """Curvature k about d = 0, run through `screen`; returns (result, strains)."""
    orig = (screening.ground_state_at_strain,
            screening.relax_internal_coordinates, screening.sensitivity)
    try:
        seen = _install(lambda d: 0.5 * k * d * d)
        r = screening.screen(_Cfg(), ["Al", "P"], mode=mode, delta=delta,
                             hessian=np.eye(6), verbose=False)
        return r, seen
    finally:
        (screening.ground_state_at_strain,
         screening.relax_internal_coordinates, screening.sensitivity) = orig


def test_tetragonal_strain_conserves_volume():
    """det(I + eps) = 1, or the measurement is contaminated by a bulk response.

    The exact ``(1+d)^-2 - 1`` matters: the linearised ``-2d`` leaves a residual
    volume change of 3 d^2, second order in precisely the quantity whose second
    derivative is being taken.
    """
    _, seen = _run("tetragonal", k=1.0)
    assert len(seen) == 3, seen
    for eps in seen:
        det = float(np.linalg.det(np.eye(3) + eps))
        assert abs(det - 1.0) < 1e-12, (eps, det)
    # And it really is the exact form, not the linearisation.
    d = 0.02
    built = [e for e in seen if abs(e[0, 0] - d) < 1e-15][0]
    assert abs(built[2, 2] - ((1.0 + d) ** -2 - 1.0)) < 1e-15, built
    assert abs(built[2, 2] - (-2.0 * d)) > 1e-4, "linearised eps_zz would pass det only to O(d^2)"


def test_tetragonal_needs_no_relaxation():
    """It must go through the plain ground state, never the relaxation path.

    That is the whole reason C11 and C12 survive on a system whose forces are
    unusable. If this branch ever starts relaxing, they stop being available and
    nothing would say so.
    """
    _, seen = _run("tetragonal", k=1.0)
    for eps in seen:
        assert abs(eps[1, 2]) < 1e-15 and abs(eps[2, 1]) < 1e-15, eps
        assert abs(eps[0, 0] - eps[1, 1]) < 1e-15, eps      # x and y move together


def test_the_three_factors_are_distinct():
    """9 V0, 6 V0 and a quarter, each against the same synthetic curvature."""
    k = 3.0
    iso, _ = _run("isotropic", k=k)
    tet, _ = _run("tetragonal", k=k)
    she, _ = _run("shear", k=k)

    scale = k * _HA_BOHR3_TO_GPA / _VOL
    assert np.isclose(iso["observable"], scale / 9.0, rtol=1e-12), iso["observable"]
    assert np.isclose(tet["observable"], scale / 6.0, rtol=1e-12), tet["observable"]
    assert np.isclose(she["observable"], scale * 0.25, rtol=1e-12), she["observable"]

    # The substitution that actually happened: isotropic factor into shear.
    assert not np.isclose(she["observable"], iso["observable"])
    assert np.isclose(she["observable"] / iso["observable"], 9.0 / 4.0, rtol=1e-12)
    # And the new one is not a disguised copy of either.
    assert np.isclose(tet["observable"] / iso["observable"], 9.0 / 6.0, rtol=1e-12)


def test_cubic_constants_round_trip():
    """C11, C12 -> the two measured combinations -> back to C11, C12."""
    for c11, c12 in ((166.0, 64.0), (125.0, 53.4), (443.0, 100.0)):
        b = (c11 + 2.0 * c12) / 3.0
        out = cubic_constants(b, c11 - c12, c44=79.6)
        assert np.isclose(out["C11"], c11, rtol=1e-12), out
        assert np.isclose(out["C12"], c12, rtol=1e-12), out
        assert np.isclose(out["shear_modulus"], (c11 - c12) / 2.0, rtol=1e-12), out
        assert out["stable"], out


def test_born_criteria_catch_signs_and_not_factors():
    """What the stability check is actually worth -- which is less than it looks.

    For positive B and positive C11 - C12 the cubic Born criteria hold
    identically: C11 + 2 C12 = 3B > 0 by construction, and C11 > |C12| reduces
    to 2B > -(C11 - C12)/3. So a wrong *factor* sails straight through. The 9/4
    error this codebase has already made returns C11 = 251, C12 = 21.5 and a
    stable crystal. Asserting otherwise was the first thing written here, and
    this test is what disproved it.

    Signs they do catch, and that is the version worth keeping.
    """
    good = cubic_constants(98.0, 102.0, c44=79.6)
    assert good["stable"], good

    # The 9/4 slip: passes stability, so nothing here can flag it.
    scaled = cubic_constants(98.0, 102.0 * 9.0 / 4.0, c44=79.6)
    assert scaled["stable"], scaled
    assert np.isclose(scaled["C11"], 251.0) and np.isclose(scaled["C12"], 21.5), scaled

    # A sign on C11 - C12 does fail, and names the criterion it failed.
    flipped = cubic_constants(98.0, -102.0, c44=79.6)
    assert not flipped["stable"], flipped
    assert any("C11 > |C12|" in v for v in flipped["violations"]), flipped["violations"]

    # A negative bulk modulus fails both of the B-dependent ones.
    negative_b = cubic_constants(-98.0, 102.0, c44=79.6)
    assert not negative_b["stable"], negative_b
    assert len(negative_b["violations"]) >= 2, negative_b["violations"]

    # And a non-positive C44, when one is supplied at all.
    assert not cubic_constants(98.0, 102.0, c44=-1.0)["stable"]


def test_c44_is_optional_and_anisotropy_follows_it():
    """Without a trustworthy force there is no C44, and the rest must still work."""
    out = cubic_constants(98.0, 102.0)
    assert out["C44"] is None and out["anisotropy"] is None, out
    assert np.isclose(out["C11"], 166.0, rtol=1e-12), out
    assert out["stable"], out

    with_c44 = cubic_constants(98.0, 102.0, c44=79.6)
    assert np.isclose(with_c44["anisotropy"], 2 * 79.6 / 102.0, rtol=1e-12), with_c44


def test_unknown_mode_is_rejected_by_name():
    """The error has to name the modes, since there are now three of them."""
    try:
        screening.screen(_Cfg(), ["Al", "P"], mode="orthorhombic", verbose=False)
    except ValueError as exc:
        msg = str(exc)
        for m in ("isotropic", "tetragonal", "shear"):
            assert m in msg, msg
    else:
        raise AssertionError("an unknown strain mode was accepted")


if __name__ == "__main__":
    test_tetragonal_strain_conserves_volume()
    test_tetragonal_needs_no_relaxation()
    test_the_three_factors_are_distinct()
    test_cubic_constants_round_trip()
    test_born_criteria_catch_signs_and_not_factors()
    test_c44_is_optional_and_anisotropy_follows_it()
    test_unknown_mode_is_rejected_by_name()
    print("test_25_tetragonal_strain: ok")
