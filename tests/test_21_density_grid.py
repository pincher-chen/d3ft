"""The FFT grid has to span the density, and someone has to say so when it doesn't.

``grid_sizes`` is an independent config entry -- it does not follow
``cutoff_energy``. Wavefunctions live inside |G| <= G_cut, so rho = |psi|^2
reaches 2 G_cut and the grid must span at least that. Nothing enforced it, and
the project's default 24 quietly failed twice on GaAs, measured against Quantum
ESPRESSO 7.5 on byte-identical pseudopotential files:

  * At 24^3 and a = 5.65 Ang the grid stops reaching 2 G_cut above
    cutoff_energy ~ 56 Ha, and the pressure walked away from the converged
    answer as the cutoff rose: +8.27 / +8.90 / +9.50 / +9.88 GPa at
    40 / 50 / 65 / 90 Ha, against QE's +3.3 (itself flat to 0.35 GPa between
    80 and 160 Ry).
  * Even at ratio 2.54, where the density does fit, the GGA integrand
    eps_xc(rho, grad rho) is not band-limited. A semicore charge is sharp
    enough that the real-space quadrature is visibly wrong: GaAs (Ga 3d) came
    out 0.024735 Ha below QE, falling to 3.2e-4 Ha at 32^3. Si and AlP, which
    have no semicore states, move 4e-7 Ha between the two grids -- which is why
    this went unnoticed for so long.

The guard shipped measuring the wrong radius and was therefore inert. It took
|G| at the *corner* of the grid box, but a density component is representable
only if its G lies inside the box, so the bound is the largest sphere the box
*contains* -- the shortest |G| on the Nyquist faces. In a primitive fcc basis
the corner runs 2.28x longer, so at cutoff_energy = 50 Ha it reported

    grid_sizes      24     32     40     48
    corner ratio  2.26   3.04   3.82   4.60     <- guard saw this, stayed quiet
    true ratio    1.00   1.33   1.67   2.00     <- density was aliased

and never fired once, including on the GaAs runs whose 0.0247 Ha offset from QE
prompted writing it. The symptom it missed is a point-group violation: with one
atom displaced along x, C2x is an exact symmetry of the cell, so each force must
lie along x, yet the transverse component reached 106% of the longitudinal one
on GaAs and 23% on GaP (soft-pseudopotential AlP and AlAs: 7e-4 and 3e-3).

Both thresholds are checked here, and so is the corner-vs-inscribed distinction.
No DFT, no GPU: the guard reads only the grid and the mask, so it can be handed
synthetic ones.
"""

import sys
import types
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3ft.calc.ground_state import _check_density_grid  # noqa: E402


def _grid(n, spacing=1.0):
    """A cubic reciprocal grid of n^3 points, |G| running out to n/2 * spacing."""
    ax = (np.arange(n) - n // 2) * spacing
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def _grid_fft_order(n, spacing=1.0):
    """The same box in FFT layout: 0..n/2-1 then -n/2..-1.

    The guard must not care which layout it is handed -- production grids are
    FFT-ordered, the synthetic ones here are centred.
    """
    ax = np.fft.fftfreq(n, d=1.0 / n) * spacing
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def _mask_within(g_vec, g_cut):
    return np.linalg.norm(g_vec, axis=-1) <= g_cut


def _warnings_for(grid_n, g_cut, builder=_grid):
    g_vec = builder(grid_n)
    cfg = types.SimpleNamespace(grid_sizes=grid_n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_density_grid(cfg, g_vec, _mask_within(g_vec, g_cut))
    return [str(w.message) for w in caught]


def test_aliased_grid_warns():
    """Ratio below 2: the density does not fit and the warning must say so.

    A 24^3 box reaches |G| = 12 on its Nyquist face. Against a cutoff sphere of
    9.0 the ratio is 1.33 -- exactly the GaAs / grid 32 / 50 Ha case in
    production, and exactly what the corner-based version waved through.
    """
    msgs = _warnings_for(24, 9.0)
    assert len(msgs) == 1, f"expected exactly one warning, got {msgs}"
    assert "aliased" in msgs[0], msgs[0]
    # It has to name a grid that actually fixes the problem, not just complain.
    suggested = int(msgs[0].split("at least")[1].strip().rstrip("."))
    assert suggested >= 24 * 2.0 / (12.0 / 9.0), suggested


def test_corner_is_not_the_measure():
    """The regression: corner ratio comfortably above 2 while the density aliases.

    24^3, cutoff 9.0. Corner |G| = 12*sqrt(3) = 20.78, so the corner ratio is
    2.31 and the old guard stayed silent. The inscribed radius is 12, ratio
    1.33, and the density is aliased. This case must warn.
    """
    g_vec = _grid(24)
    norms = np.linalg.norm(g_vec, axis=-1)
    corner_ratio = norms.max() / norms[_mask_within(g_vec, 9.0)].max()
    assert corner_ratio > 2.0, corner_ratio       # the old guard's blind spot
    msgs = _warnings_for(24, 9.0)
    assert msgs and "aliased" in msgs[0], (corner_ratio, msgs)


def test_layout_does_not_matter():
    """Centred and FFT-ordered grids describe the same box and must agree."""
    assert _warnings_for(24, 9.0, _grid) == _warnings_for(24, 9.0, _grid_fft_order)
    assert _warnings_for(24, 4.0, _grid) == _warnings_for(24, 4.0, _grid_fft_order)


def test_thin_grid_warns_softly():
    """Ratio in [2, 2.5): the density fits but the GGA integrand is under-resolved.

    Inscribed radius 12 against a cutoff of 5.45 gives 2.20 -- inside the band
    where GaAs was still 0.0247 Ha off QE with the density itself fitting.
    """
    msgs = _warnings_for(24, 5.45)
    assert len(msgs) == 1, f"expected exactly one warning, got {msgs}"
    assert "semicore" in msgs[0], msgs[0]
    assert "aliased" not in msgs[0], msgs[0]


def test_adequate_grid_is_silent():
    """A comfortable grid must not warn, or the guard becomes noise and gets muted."""
    assert _warnings_for(24, 4.0) == []          # inscribed 12 / 4.0 = 3.00


def test_empty_mask_is_silent():
    """No wavefunction basis means nothing to compare against; must not divide by zero."""
    g_vec = _grid(16)
    cfg = types.SimpleNamespace(grid_sizes=16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_density_grid(cfg, g_vec, np.zeros(g_vec.shape[:3], dtype=bool))
    assert [str(w.message) for w in caught] == []


if __name__ == "__main__":
    test_aliased_grid_warns()
    test_corner_is_not_the_measure()
    test_layout_does_not_matter()
    test_thin_grid_warns_softly()
    test_adequate_grid_is_silent()
    test_empty_mask_is_silent()
    print("test_21_density_grid: ok")
