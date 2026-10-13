"""An internal relaxation that runs out of steps must not pass for a converged one.

``screen(mode="shear")`` relaxes the two sublattices against each other at every
strain point, because C44 with the atoms clamped is a different (and much
stiffer) number. ``relax_internal_coordinates`` caps itself at 22 steps and
reports ``converged``; ``screen`` used to ignore that flag, so a relaxation that
simply ran out of steps returned its last geometry like any other and C44 came
out of it with nothing to show that the atoms had never reached zero force.

The step count is set by the quality of the force-constant matrix used as the
Newton preconditioner, which is exactly the thing that varies between materials:
AlP and Si reach an acoustic-sum-rule residual around 2e-3 and relax in about
three steps, while GaAs at cutoff 50 Ha came in at 1.17e-1 and needed roughly
eleven. Eleven against a cap of 22 is close enough that hitting it is a live
possibility, not a theoretical one -- and the failure would be silent.

No DFT: ``relax_internal_coordinates`` and ``sensitivity`` are replaced, so what
is exercised is the guard and the arithmetic around it.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3ft.calc import screening  # noqa: E402

_HA_BOHR3_TO_GPA = screening._HA_BOHR3_TO_GPA


class _Cfg:
    """The little of a config that `screen` reads before the strain loop."""

    def __init__(self):
        self.xc = "gga_x_pbe+gga_c_pbe"
        self.d3ft = {"xc_params": None}


def _fake_out(cell_bohr, energy):
    crystal = types.SimpleNamespace(cell_vectors=np.eye(3) * cell_bohr)
    return types.SimpleNamespace(
        total_energy=energy, freq_mask=np.ones((2, 2, 2), dtype=bool),
        crystal=crystal,
    )


def _install(monkey_converged, energies):
    """Replace the two expensive calls; return the recorded strain points."""
    seen = []

    def fake_relax(config, symbols, strain=None, **kw):
        d = float(strain[1, 2])
        seen.append(d)
        return {
            "ground_state": _fake_out(10.0, energies(d)),
            "converged": monkey_converged,
            "max_force": 1.0e-4 if monkey_converged else 7.3e-3,
            "steps": 4 if monkey_converged else 22,
        }

    screening.relax_internal_coordinates = fake_relax
    screening.sensitivity = lambda out, xc_base, nc: np.zeros(
        len(screening.parameter_names(xc_base, nc))
    )
    return seen


def test_unconverged_relaxation_raises():
    """The whole point: it must stop, and say what it stopped on."""
    _orig = screening.relax_internal_coordinates, screening.sensitivity
    try:
        _install(False, lambda d: -1.0)
        try:
            screening.screen(_Cfg(), ["Ga", "As"], mode="shear", delta=0.02,
                             hessian=np.eye(6), verbose=False)
        except RuntimeError as exc:
            msg = str(exc)
            assert "7.300e-03" in msg, msg      # the force it actually reached
            assert "22 steps" in msg, msg       # that it ran out rather than diverged
            assert "1.0e-4" in msg, msg         # the tolerance it was judged against
            assert "hessian" in msg, msg        # where to go and fix it
        else:
            raise AssertionError("an unconverged relaxation was accepted silently")
    finally:
        screening.relax_internal_coordinates, screening.sensitivity = _orig


def test_converged_relaxation_gives_c44():
    """And when it does converge, the Voigt quarter must still be there.

    eps_4 = 2 eps_yz, so C44 is a quarter of the curvature in eps_yz -- not the
    1/(9 V0) of the isotropic branch, which would scale the answer by 9/4. That
    substitution has happened once in this codebase already.
    """
    _orig = screening.relax_internal_coordinates, screening.sensitivity
    try:
        # E(d) = k d^2 / 2 about d = 0, so the second difference is k delta^2.
        k = 3.0
        _install(True, lambda d: 0.5 * k * d * d)
        r = screening.screen(_Cfg(), ["Ga", "As"], mode="shear", delta=0.02,
                             hessian=np.eye(6), verbose=False)
        vol = 10.0 ** 3
        expected = 0.25 * k * _HA_BOHR3_TO_GPA / vol
        assert abs(r["observable"] - expected) < 1e-9 * max(1.0, abs(expected)), \
            (r["observable"], expected)
        assert r["mode"] == "shear"
    finally:
        screening.relax_internal_coordinates, screening.sensitivity = _orig


def test_all_three_strain_points_are_relaxed():
    """Zero strain included: it sets the reference volume and the basis."""
    _orig = screening.relax_internal_coordinates, screening.sensitivity
    try:
        seen = _install(True, lambda d: 0.5 * 3.0 * d * d)
        screening.screen(_Cfg(), ["Ga", "As"], mode="shear", delta=0.02,
                         hessian=np.eye(6), verbose=False)
        assert sorted(seen) == [-0.02, 0.0, 0.02], seen
        assert seen[0] == 0.0, f"zero strain must come first, got {seen}"
    finally:
        screening.relax_internal_coordinates, screening.sensitivity = _orig


if __name__ == "__main__":
    test_unconverged_relaxation_raises()
    test_converged_relaxation_gives_c44()
    test_all_three_strain_points_are_relaxed()
    print("test_22_relaxation_guard: ok")
