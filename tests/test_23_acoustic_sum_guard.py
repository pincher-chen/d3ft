"""Forces that do not sum to zero must stop a relaxation, not steer it.

``sum_I F_I = 0`` follows from translational invariance. A nonzero sum bounds
the error in the forces without saying whether the state or the implementation
is at fault -- but either way, relaxing on them does not find a force minimum,
and ``relax_internal_coordinates`` used to do exactly that and return a geometry
with nothing to show for it. C44 comes out of that geometry.

The numbers this guard is set against, all on a displaced zincblende cell at
uniform settings (SG15 PBE, cutoff 50 Ha, grid 32, k = 6), ``|sum F| / max|F|``
beside the transverse force that C2x symmetry requires to vanish:

    AlP  (Al 11 with 2s2p semicore, P 5)     1.6e-02      7e-04
    AlAs (Al 11, As 5)                       1.0e-02      3e-03
    GaP  (Ga 13 with 3d, P 5)                2.2e-01      0.23
    GaAs (Ga 13 with 3d, As 5)               5.4e-01      1.06

The tolerance of 5% sits above both clean cases by a factor of three and below
both broken ones by a factor of four.

The cause is not known. It is not the force implementation (the first version
of this docstring claimed that) and not the aliased density (the second claimed
that). What is established is that on a d-valence system the force is not a
reproducible quantity: over four random seeds ``F_x`` on the cation lands at
0.46 / 0.81 / 0.52 / -0.24 of the Quantum ESPRESSO reference for GaAs, against
0.9860 / 0.9873 / 0.9880 / 0.9851 for AlP, while the energy is reproducible to
1e-06 Ha in both. Tightening convergence makes it worse: over 3000 / 9000 /
27000 steps the residual falls 3.62e-05 -> 6.62e-06 while the transverse force
grows 9.1e-04 -> 4.4e-03. See d3ft.calc.forces for the full elimination list.

So this guard is doing the only thing that can be done today -- refusing to
relax on forces that cannot be trusted. It is not a diagnostic.

No DFT here: the guard reads a force array, so it can be handed one.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3ft.calc.forces import _check_acoustic_sum  # noqa: E402


def _pair(net_fraction, fmax=1.0e-3):
    """Two forces whose sum is `net_fraction` of the larger one."""
    return np.array([[fmax, 0.0, 0.0],
                     [-fmax * (1.0 - net_fraction), 0.0, 0.0]])


def test_clean_forces_pass():
    """The measured soft-pseudopotential cases must all get through."""
    for ratio in (1.57e-2, 1.02e-2, 2.6e-3, 3.1e-3):
        _check_acoustic_sum(_pair(ratio), ["Al", "P"])


def test_d_valence_ratios_are_refused():
    """Both measured d-valence cases must not get through."""
    for ratio, seen, syms in ((0.544, "54.4%", ["Ga", "As"]),
                              (0.224, "22.4%", ["Ga", "P"])):
        try:
            _check_acoustic_sum(_pair(ratio), syms)
        except RuntimeError as exc:
            assert seen in str(exc), str(exc)
        else:
            raise AssertionError(f"a {ratio:.0%} violation was accepted for {syms}")


def test_message_points_at_the_grid_not_the_finite_difference():
    """The advice must name the real cause and disown the criterion that misled.

    The first version sent the reader to a finite difference of the energy.
    That is what produced the wrong diagnosis in the first place, so the message
    now has to warn against it rather than recommend it.
    """
    try:
        _check_acoustic_sum(_pair(0.544), ["Ga", "As"])
    except RuntimeError as exc:
        msg = str(exc)
        assert "_check_density_grid" in msg, msg        # the actual cause
        assert "variational_residual" in msg, msg       # how to attribute it
        assert "Do not use a finite difference" in msg, msg
    else:
        raise AssertionError("a 54% acoustic sum rule violation was accepted")


def test_threshold_is_where_it_claims():
    """Just under 5% passes, just over does not -- no silent drift in the bound."""
    _check_acoustic_sum(_pair(0.049), ["Ga", "As"])
    try:
        _check_acoustic_sum(_pair(0.051), ["Ga", "As"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("the stated 5% tolerance is not the actual one")


def test_zero_forces_do_not_divide_by_zero():
    """A geometry already at zero force has no ratio to check."""
    _check_acoustic_sum(np.zeros((2, 3)), ["Ga", "As"])


if __name__ == "__main__":
    test_clean_forces_pass()
    test_d_valence_ratios_are_refused()
    test_message_points_at_the_grid_not_the_finite_difference()
    test_threshold_is_where_it_claims()
    test_zero_forces_do_not_divide_by_zero()
    print("test_23_acoustic_sum_guard: ok")
