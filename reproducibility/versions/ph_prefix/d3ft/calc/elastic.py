"""Elastic constants as the strain derivative of the stress.

``C_ijkl = d(sigma_ij) / d(eps_kl)``. The stress itself comes from AD of the free
energy, but the elastic constants **cannot** be taken as a second AD derivative of
the same free energy at fixed wavefunctions. At a variational minimum
``dF/d(psi) = 0``, so the first derivative with respect to strain is the partial
one -- that is the Hellmann-Feynman step ``stress_tensor`` relies on. The second
derivative is not::

    d2F/deps2 = partial2F/deps2 + (partial2F / deps dpsi) . dpsi/deps

and ``dpsi/deps`` does not vanish. Differentiating twice through fixed orbitals
gives the clamped-orbital constants, which are systematically too stiff; that
missing term is what DFPT exists to compute.

So this module relaxes the state in each strained cell and finite-differences the
converged stresses. Each stress is exact AD at its own minimum, so only the outer
derivative is numerical. The cost is one ground state per strain point -- five for
a cubic crystal with central differences -- which warm starts from the reference
state make bearable.

**The plane-wave basis is held fixed across the five solves.** The mask is
``|G| <= cutoff`` on the deformed reciprocal lattice, so left alone it changes
size from one strain to the next -- 1243 vectors to 1227 for AlP at
``eps_xx = 0.5%``. The AD stress differentiates at a fixed set of Miller indices,
which is exactly why it carries no Pulay term; an outer finite difference over
cells with different bases would fold that discontinuity back in and would no
longer be differentiating the same quantity. Every deformed solve therefore
inherits the reference mask.

**These are clamped-ion constants.** Atoms are carried affinely with the cell and
their fractional coordinates are held fixed, so the internal displacement that
zincblende allows under shear (the Kleinman parameter ``zeta``) is not relaxed.
For C11 and C12 in zincblende this changes nothing by symmetry; **for C44 it
does**, and the relaxed value is lower. A QE `scf` run at a strained cell with
fixed `ATOMIC_POSITIONS` measures the same clamped-ion quantity, so the two are
comparable as long as neither relaxes.

Voigt bookkeeping, since it is the usual place to lose a factor of two: the Voigt
shear strain is ``eps_4 = 2 eps_yz``, so ``C44 = d(sigma_yz) / d(2 eps_yz)``, half
the tensorial derivative. C11 and C12 need no such factor.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from .stress import _strained, pressure, stress_from_ground_state

# Ha/Bohr^3 -> GPa, the same constant stress.py uses.
_HA_BOHR3_TO_GPA = 29421.02648438959
_BOHR = 0.529177210903


def _write_xyz(path, cell_bohr, positions_bohr, symbols):
    """extXYZ in Angstrom, the format ``config.crystal_file_path_path`` expects."""
    cell = np.asarray(cell_bohr) * _BOHR
    pos = np.asarray(positions_bohr) * _BOHR
    lattice = " ".join(f"{v:.12f}" for v in np.asarray(cell).reshape(-1))
    lines = [
        str(len(symbols)),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"',
    ]
    for sym, r in zip(symbols, pos):
        lines.append(f"{sym} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def ground_state_at_strain(
    config,
    strain,
    symbols,
    init_params: Optional[dict] = None,
    workdir: Optional[str] = None,
    freq_mask=None,
):
    """Relax the electronic state in a cell deformed by ``strain``.

    The cell is mapped by ``(I + strain)`` and the atoms ride along affinely, so
    fractional coordinates are preserved -- see the module docstring on what that
    means for C44.

    Args:
        config: base config; its crystal supplies the undeformed geometry.
        strain: (3, 3) strain tensor.
        symbols: chemical symbols, in the order the crystal stores them.
        init_params: converged parameters of a neighbouring strain, to warm start.
        workdir: where to write the deformed geometry; a temp dir if omitted.
        freq_mask: the reference mask, so every strain shares one basis. Omitting
            it lets the basis change with the cell -- see the module docstring.

    Returns:
        The :class:`GroundStateOutput` for the deformed cell.
    """
    from jrystal.calc.opt_utils import create_crystal

    from .ground_state import calc

    crystal = create_crystal(config)
    cell, positions = _strained(
        np.asarray(crystal.cell_vectors), np.asarray(crystal.positions),
        np.asarray(strain),
    )

    owned = workdir is None
    workdir = tempfile.mkdtemp(prefix="d3ft_elastic_") if owned else workdir
    os.makedirs(workdir, exist_ok=True)
    tag = "_".join(f"{v:+.6f}" for v in np.asarray(strain).reshape(-1))
    xyz = os.path.join(workdir, f"strained{tag}.xyz")
    _write_xyz(xyz, cell, positions, symbols)

    cfg = copy.deepcopy(config)
    cfg.crystal = None
    cfg.crystal_file_path_path = xyz
    return calc(cfg, init_params=init_params, freq_mask=freq_mask)


def _sigma_at(config, strain, symbols, init_params, workdir, freq_mask=None):
    out = ground_state_at_strain(
        config, strain, symbols, init_params=init_params, workdir=workdir,
        freq_mask=freq_mask,
    )
    sigma = np.asarray(stress_from_ground_state(out))
    return sigma, {"pw": out.params_pw, "occ": out.params_occ}, out


def reduce_cubic(s_xx_p, s_xx_m, s_yz_p, s_yz_m, delta, unit=_HA_BOHR3_TO_GPA):
    """Central differences of four stresses into C11, C12, C44 and B.

    Split out from the solves so the QE reference workflow can run the identical
    arithmetic on QE's stresses. The comparison is only informative if the two
    sides differ in the stress and in nothing else -- a second, parallel
    implementation of this reduction would be one more place for them to diverge
    silently.

    Args:
        s_xx_p, s_xx_m: stress at ``eps_xx = +delta`` and ``-delta``.
        s_yz_p, s_yz_m: stress at ``eps_yz = eps_zy = +delta`` and ``-delta``.
        delta: the strain amplitude used.
        unit: multiplier taking the stress unit to GPa. Ha/Bohr^3 by default;
            pass the Ry/Bohr^3 constant for QE.

    Returns:
        ``{"C11", "C12", "C44", "B"}`` in whatever unit ``unit`` produces.
    """
    s_xx_p = np.asarray(s_xx_p)
    s_xx_m = np.asarray(s_xx_m)
    s_yz_p = np.asarray(s_yz_p)
    s_yz_m = np.asarray(s_yz_m)
    scale = unit / (2.0 * delta)

    c11 = float(s_xx_p[0, 0] - s_xx_m[0, 0]) * scale
    # y and z are equivalent under cubic symmetry, so averaging them costs
    # nothing and halves the optimiser noise carried into C12.
    c12 = 0.5 * float(
        (s_xx_p[1, 1] - s_xx_m[1, 1]) + (s_xx_p[2, 2] - s_xx_m[2, 2])
    ) * scale
    # Voigt: eps_4 = 2 eps_yz, hence the factor of 1/2.
    c44 = 0.5 * float(s_yz_p[1, 2] - s_yz_m[1, 2]) * scale
    return {"C11": c11, "C12": c12, "C44": c44, "B": (c11 + 2.0 * c12) / 3.0}


RY_BOHR3_TO_GPA = _HA_BOHR3_TO_GPA / 2.0
"""Ry/Bohr^3 to GPa, for reducing Quantum ESPRESSO stresses."""


def cubic_elastic_constants(
    config,
    symbols,
    delta: float = 5.0e-3,
    warm_start: bool = True,
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """C11, C12 and C44 for a cubic crystal, by central differences of the stress.

    Five ground states: the reference, ``eps_xx = +/- delta``, and
    ``eps_yz = eps_zy = +/- delta``.

    Args:
        config: base config at the reference geometry.
        symbols: chemical symbols in crystal order.
        delta: strain amplitude. The usual advice is to balance optimiser noise
            (which grows as 1/delta) against anharmonicity (which grows as
            delta^2). For AlP the second half of that trade-off turns out not to
            bind: QE over the same deformations gives C11 = 129.3, 129.4, 129.4
            and C44 = 89.8, 89.8, 89.9 GPa at delta = 2e-3, 5e-3, 1e-2 -- flat to
            0.1 GPa, so there is no measurable anharmonicity to trade against.
            Only the noise term matters, and **1e-2 is the better choice**: d3ft
            lands within 0.6% of QE on C11 there against 2.7% at 5e-3 and 13% at
            2e-3.
        warm_start: start each strained solve from the reference state's
            parameters. Cheaper, and it keeps the five solves in the same basin,
            which matters because the seed-to-seed spread on the energy is larger
            than the stress differences being measured.
        workdir: where deformed geometries are written.
        verbose: print each solve as it completes.

    Returns:
        Dict with ``C11``, ``C12``, ``C44`` and the bulk modulus ``B`` in GPa, the
        reference pressure, and the raw stresses.
    """
    symbols = list(symbols)
    ref_sigma, ref_params, ref_out = _sigma_at(
        config, np.zeros((3, 3)), symbols, None, workdir
    )
    mask = ref_out.freq_mask
    seed = ref_params if warm_start else None
    p_ref = pressure(ref_sigma)
    if verbose:
        print(f"  reference: P = {p_ref:+.4f} GPa")

    def strained(matrix):
        sigma, _, _ = _sigma_at(config, matrix, symbols, seed, workdir, mask)
        return sigma

    e_xx = np.zeros((3, 3))
    e_xx[0, 0] = delta
    s_xx_p = strained(e_xx)
    s_xx_m = strained(-e_xx)

    e_yz = np.zeros((3, 3))
    e_yz[1, 2] = e_yz[2, 1] = delta
    s_yz_p = strained(e_yz)
    s_yz_m = strained(-e_yz)

    constants = reduce_cubic(
        s_xx_p, s_xx_m, s_yz_p, s_yz_m, delta, unit=_HA_BOHR3_TO_GPA
    )

    return {
        **constants,
        "pressure_GPa": p_ref,
        "delta": delta,
        "stresses": {
            "ref": ref_sigma, "xx+": s_xx_p, "xx-": s_xx_m,
            "yz+": s_yz_p, "yz-": s_yz_m,
        },
        "ground_state": ref_out,
    }


def bulk_modulus_from_volume(
    config, symbols, delta: float = 5.0e-3, workdir: Optional[str] = None
) -> float:
    """``B = -V dP/dV`` from a pure volumetric strain, in GPa.

    An independent route to the same number as ``(C11 + 2 C12)/3``. The two use
    different deformations and different components of sigma, so agreement is a
    real check on the stress rather than a restatement of it.

    Both are stress routes, so both inherit the stress's assumption that
    ``dF/d(params)`` vanishes. At ``theta_nonlocal = 0`` that assumption holds
    well enough -- stress and relaxed-energy curvature agree on C44 to 0.016% --
    but at ``|theta| = 0.1`` the stress is 2.7% off its own energy surface's
    slope; see :mod:`d3ft.calc.stationary`. For a Hamiltonian parameter away from
    zero, take the modulus from the energy curvature instead.
    """
    symbols = list(symbols)
    iso = np.eye(3) * delta
    ref = ground_state_at_strain(config, np.zeros((3, 3)), symbols, workdir=workdir)
    mask = ref.freq_mask
    s_p, _, _ = _sigma_at(config, iso, symbols, None, workdir, mask)
    s_m, _, _ = _sigma_at(config, -iso, symbols, None, workdir, mask)
    # eps = delta * I gives dV/V = 3 delta to first order.
    return -(pressure(s_p) - pressure(s_m)) / (2.0 * delta * 3.0)


def elastic_report(result: Dict[str, Any], reference: Optional[Dict] = None) -> str:
    """One-line-per-constant summary, optionally against a reference."""
    rows = [("C11", result["C11"]), ("C12", result["C12"]),
            ("C44", result["C44"]), ("B", result["B"])]
    width = 44 if reference else 26
    lines = ["  " + "constant   d3ft (GPa)".ljust(width)]
    if reference:
        lines[0] = "  constant   d3ft (GPa)   reference   difference"
    for name, value in rows:
        if reference and name in reference:
            ref = reference[name]
            delta = 100.0 * (value - ref) / ref if ref else float("nan")
            lines.append(f"  {name:<9}{value:>11.1f}{ref:>12.1f}{delta:>11.1f}%")
        else:
            lines.append(f"  {name:<9}{value:>11.1f}")
    return "\n".join(lines)
