"""Forces can be unusable on a system whose convergence looks fine.

The variational residual does not separate a system whose forces can be trusted
from one whose forces cannot. Measured on two random seeds at 3000 steps,
cutoff 50 Ha, grid 32, k = 6:

    AlP    residual 1.7e-05   subspace angle 4.05e-05 rad   |dF| 5.49e-06
    GaAs   residual 3.6e-05   subspace angle 1.85e-05 rad   |dF| 1.28e-03

Two things to read off. The residuals are the same order and put the systems in
the *wrong* order. And AlP's two solutions sit **further apart** than GaAs's --
2.2x the mean principal angle -- while its forces still agree 233x better. GaAs
is not the badly converged one; its forces amplify the same difference ~500x
harder, because ``dV_nl/dR`` on a deep Ga 3d projector has that much larger a
norm.

Reading the residual instead is what led to dismissing convergence as the cause
early on, on the system whose forces turned out to be unusable. So the quantity
this module screens on is the ratio, not either half of it.

No DFT, no GPU: the helper takes coefficient and force arrays, so it can be
handed synthetic ones built to a chosen subspace angle.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3ft.calc.forces import force_amplification  # noqa: E402


def _pair(angle, n_kpt=3, n_occ=2, n_g=8, n_band=None, seed=0):
    """Two orthonormal band sets whose subspaces differ by a known angle.

    The second is the first with one occupied direction tilted by ``angle``
    towards an unoccupied one, so every principal angle is 0 except one, which
    is exactly ``angle``.
    """
    rng = np.random.default_rng(seed)
    n_band = n_band or n_occ + 1
    a = np.empty((1, n_kpt, n_band, n_g), dtype=complex)
    b = np.empty_like(a)
    for k in range(n_kpt):
        m = rng.normal(size=(n_band, n_g)) + 1j * rng.normal(size=(n_band, n_g))
        q, _ = np.linalg.qr(m.conj().T)
        q = q.conj().T                       # [n_band, n_g], orthonormal rows
        a[0, k] = q
        tilted = q.copy()
        tilted[0] = np.cos(angle) * q[0] + np.sin(angle) * q[n_occ]
        b[0, k] = tilted
    return a, b


def test_recovers_a_known_subspace_angle():
    """The measured mean angle must be the one that was built in."""
    angle = 3.0e-05
    a, b = _pair(angle, n_kpt=4, n_occ=2)
    out = force_amplification(a, np.zeros((2, 3)), b, np.zeros((2, 3)), n_occ=2)
    # One principal angle is `angle`, the other is 0, so the mean is half.
    assert np.isclose(out["angle_mean"], angle / 2, rtol=1e-6), out
    assert np.isclose(out["angle_max"], angle, rtol=1e-6), out


def test_separates_the_two_measured_systems():
    """The AlP and GaAs measurements must land on opposite sides of any cut.

    This is the point of the whole quantity: the residual orders these two
    backwards, and the amplification separates them by a factor of 500.
    """
    angle_alp, dforce_alp = 4.05e-05, 5.487e-06
    angle_gaas, dforce_gaas = 1.85e-05, 1.279e-03

    a, b = _pair(2 * angle_alp, n_kpt=2, n_occ=2, seed=1)
    alp = force_amplification(
        a, np.zeros((2, 3)), b, np.full((2, 3), dforce_alp), n_occ=2)
    a, b = _pair(2 * angle_gaas, n_kpt=2, n_occ=2, seed=2)
    gaas = force_amplification(
        a, np.zeros((2, 3)), b, np.full((2, 3), dforce_gaas), n_occ=2)

    assert np.isclose(alp["amplification"], 0.135, rtol=0.02), alp
    assert np.isclose(gaas["amplification"], 69.2, rtol=0.02), gaas
    assert gaas["amplification"] / alp["amplification"] > 400, (alp, gaas)
    # And the half that misleads: GaAs's subspaces are the *closer* pair.
    assert gaas["angle_mean"] < alp["angle_mean"], (alp, gaas)


def test_identical_solutions_sit_on_the_resolution_floor():
    """Same state twice measures ~1e-08 rad apart, not 0, and that floor matters.

    ``arccos`` is ill-conditioned at 1: in double precision the best it can do
    is ``sqrt(2 * eps) = 2.1e-08``. So this measure cannot resolve a subspace
    difference below roughly that, which is only one decade under the ~3e-07 a
    1%-accurate GaAs force would need. Pinned so the floor is not mistaken for
    a real angle later.
    """
    a, _ = _pair(1.0e-05)
    out = force_amplification(a, np.zeros((2, 3)), a, np.zeros((2, 3)), n_occ=2)
    assert out["dforce_max"] == 0.0, out
    assert out["angle_max"] < 5e-08, out            # the floor, not a real angle
    assert out["angle_mean"] < 5e-08, out
    # And it is a floor, not zero -- worth knowing before reading a small number
    # off this function as convergence.
    assert out["angle_max"] > 1e-09, out


def test_subset_of_the_occupied_manifold_is_rejected_by_measurement():
    """Comparing part of the occupied space measures gauge, not physics.

    Two runs converged to the *same* subspace still return large angles if only
    some of the occupied bands are compared, because the bands inside the
    manifold have no defined order. This is not a guard in the code -- it is the
    reason `n_occ` must be all of them, and the failure it produces looks like a
    real signal, so it is pinned here.
    """
    rng = np.random.default_rng(3)
    n_occ, n_g = 4, 10
    m = rng.normal(size=(n_occ, n_g)) + 1j * rng.normal(size=(n_occ, n_g))
    q, _ = np.linalg.qr(m.conj().T)
    q = q.conj().T
    # A different basis for the identical subspace.
    u, _ = np.linalg.qr(rng.normal(size=(n_occ, n_occ))
                        + 1j * rng.normal(size=(n_occ, n_occ)))
    a = q[None, None, :, :]
    b = (u @ q)[None, None, :, :]

    full = force_amplification(a, np.zeros((2, 3)), b, np.zeros((2, 3)), n_occ=n_occ)
    # Same subspace: down on the ~2e-08 arccos floor, not below it.
    assert full["angle_max"] < 5e-08, full

    half = force_amplification(a, np.zeros((2, 3)), b, np.zeros((2, 3)),
                               n_occ=n_occ // 2)
    assert half["angle_max"] > 0.1, half            # pure gauge, looks like signal


if __name__ == "__main__":
    test_recovers_a_known_subspace_angle()
    test_separates_the_two_measured_systems()
    test_identical_solutions_sit_on_the_resolution_floor()
    test_subset_of_the_occupied_manifold_is_rejected_by_measurement()
    print("test_24_force_amplification: ok")
