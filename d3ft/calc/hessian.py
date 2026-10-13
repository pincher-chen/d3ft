"""Force constants, phonons, and the ionic response the implicit chain needs.

``Phi_{I alpha, J beta} = -d F_{I alpha} / d R_{J beta}``, by displacing atoms and
finite-differencing the Hellmann-Feynman forces. One object, used twice:

**As physics.** Its eigenvalues are the zone-centre phonon frequencies, an
experimentally measurable quantity, and one that tests the forces far more
sharply than a relaxation does -- a relaxation only needs the forces to point
roughly downhill, while a frequency needs their slope right.

**As machinery.** It is the ``(dF/dR)^{-1}`` in
``dR*/dtheta = -(dF/dR)^{-1} (dF/dtheta)``, the ionic half of the implicit chain
in Phase 2. See PLAN.md: with the ions clamped, AlP's C44 is 46% from experiment
in both d3ft and QE, so any objective built on measured elastic data has to carry
this term.

Why finite differences here, when the forces themselves come from AD: the
Hellmann-Feynman force is a *total* derivative only because the state is
variational. Differentiating it again at fixed state gives the clamped-orbital
force constants, which miss the electronic response -- the same trap as taking a
second AD derivative for the elastic constants (see :mod:`d3ft.calc.elastic`).
Displacing the atoms and re-solving lets the electrons respond, which is what the
frozen-phonon method is for.

Units: force constants in Ha/Bohr^2, frequencies in cm^-1 via the atomic masses.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forces import forces_from_ground_state

_HARTREE_CM = 219474.6313632
"""One Hartree in cm^-1."""

_AMU_TO_ME = 1822.888486209
"""One atomic mass unit in electron masses."""

# omega[cm^-1] = sqrt(k[Ha/Bohr^2] / m[amu]) * this. Derived rather than quoted:
# omega in atomic units is sqrt(k/m) with m in electron masses, so converting the
# mass costs sqrt(_AMU_TO_ME) and converting the frequency costs _HARTREE_CM.
# The value first used here, 3739.4257, is neither this nor its Rydberg
# counterpart (3634.6) -- it put AlP's optical mode at 320 cm^-1 against an
# experimental 439, a 27% error that looked like broken force constants.
_HA_BOHR2_AMU_TO_CM = _HARTREE_CM / _AMU_TO_ME ** 0.5

# Only what the current test systems need; extend as required.
_MASSES = {
    "H": 1.008, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999,
    "Al": 26.9815385, "Si": 28.085, "P": 30.9737619, "S": 32.06,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971,
    "In": 114.818, "Sb": 121.760, "Te": 127.60,
}


def _forces_at(config, positions, symbols, init_params, freq_mask, workdir):
    """Hellmann-Feynman forces with the nuclei at ``positions`` (Bohr, Cartesian)."""
    from .elastic import _write_xyz
    from .ground_state import calc

    from jrystal.calc.opt_utils import create_crystal

    crystal = create_crystal(config)
    cell = np.asarray(crystal.cell_vectors)
    xyz = os.path.join(workdir, f"disp_{abs(hash(positions.tobytes())):016x}.xyz")
    _write_xyz(xyz, cell, positions, symbols)

    cfg = copy.deepcopy(config)
    cfg.crystal = None
    cfg.crystal_file_path_path = xyz
    out = calc(cfg, init_params=init_params, freq_mask=freq_mask)
    return np.asarray(forces_from_ground_state(out)), out


def displacement_tasks(
    ref_positions, displacement: float = 0.01
) -> List[Tuple[str, np.ndarray]]:
    """The ``6N`` displaced geometries the force constants need.

    Split out from :func:`force_constants` so a caller can evaluate them
    however it likes -- in particular across several GPUs at once. They are
    completely independent of one another: each is a ground state at its own
    geometry, sharing only the reference basis and warm-start parameters. On a
    two-atom cell that is 12 solves, which at roughly four minutes each is 48
    minutes in sequence and about eight across eight devices.

    This is the part of the workload that parallelises. The relaxation is not:
    each step needs the geometry the previous step produced. See
    :func:`~d3ft.calc.forces.relax_internal_coordinates` on using the matrix
    built here to cut that sequential chain from 17 steps to 3.

    Args:
        ref_positions: Cartesian positions to expand about, (N, 3) in Bohr.
        displacement: Cartesian displacement in Bohr.

    Returns:
        ``[(label, positions), ...]`` with labels ``"a{atom}x{axis}{+|-}"``, in
        the order :func:`assemble_force_constants` expects.
    """
    ref_positions = np.asarray(ref_positions)
    natom = ref_positions.shape[0]
    tasks = []
    for atom in range(natom):
        for axis in range(3):
            for sign, mark in ((+1, "+"), (-1, "-")):
                moved = ref_positions.copy()
                moved[atom, axis] += sign * displacement
                tasks.append((f"a{atom}x{axis}{mark}", moved))
    return tasks


def assemble_force_constants(
    forces_by_label: Dict[str, np.ndarray],
    natom: int,
    displacement: float,
) -> Dict[str, Any]:
    """Build ``Phi`` from the forces at the displaced geometries.

    The reduction half of :func:`force_constants`, separated so it can be run on
    results gathered from anywhere -- several GPUs, several jobs, a cache.

    Args:
        forces_by_label: ``{label: forces (N, 3)}`` for every label
            :func:`displacement_tasks` produced.
        natom: number of atoms.
        displacement: the amplitude those tasks used.

    Returns:
        Same shape of dict as :func:`force_constants`, minus the ground state.
    """
    phi = np.zeros((3 * natom, 3 * natom))
    for atom in range(natom):
        for axis in range(3):
            col = 3 * atom + axis
            f_p = np.asarray(forces_by_label[f"a{atom}x{axis}+"])
            f_m = np.asarray(forces_by_label[f"a{atom}x{axis}-"])
            phi[:, col] = -(f_p - f_m).reshape(-1) / (2.0 * displacement)

    asym = float(np.abs(phi - phi.T).max())
    phi = 0.5 * (phi + phi.T)
    blocks = phi.reshape(natom, 3, natom, 3)
    asr = float(np.abs(blocks.sum(axis=2)).max())
    for i in range(natom):
        blocks[i, :, i, :] -= blocks[i].sum(axis=1)

    return {
        "phi": phi,
        "asymmetry": asym,
        "residual_asr": asr,
        "displacement": displacement,
    }


def cached_force_constants(cache_path, config, symbols, **kwargs):
    """``force_constants`` with the result kept on disk.

    The matrix costs 6N + 1 ground states -- 13 for a two-atom cell, about
    46 minutes at Gamma 6x6x6 -- and every consumer wants the same one: the
    Newton relaxation, the phonons, and any parameter sweep that re-relaxes at
    several values. Recomputing it per run is the single largest avoidable cost
    in this codebase, and it is what a killed job loses.

    The cache is keyed only by the path, so the caller owns the naming. Anything
    that changes the matrix -- geometry, cutoff, k-mesh, functional,
    pseudopotential, displacement -- has to go in that name.

    Args:
        cache_path: npz to read or write.
        config, symbols, **kwargs: as :func:`force_constants`.

    Returns:
        The same dict, minus ``ground_state`` when loaded from cache.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        print(f"  force constants: loaded {cache_path}")
        return {
            "phi": z["phi"],
            "asymmetry": float(z["asymmetry"]),
            "residual_asr": float(z["residual_asr"]),
            "displacement": float(z["displacement"]),
        }
    result = force_constants(config, symbols, **kwargs)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        # Write through a temporary and rename. Two jobs sharing a matrix is the
        # normal case -- that is the point of the cache -- and a second job that
        # polls for the file would otherwise be able to open one np.savez has
        # only half written. os.replace is atomic within a filesystem, so the
        # path existing means the matrix is complete. The suffix keeps .npz last
        # because np.savez appends it to any name that lacks it.
        tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, phi=result["phi"],
                 asymmetry=result["asymmetry"],
                 residual_asr=result["residual_asr"],
                 displacement=result["displacement"])
        os.replace(tmp, cache_path)
        print(f"  force constants: wrote {cache_path}")
    return result


def force_constants(
    config,
    symbols: Sequence[str],
    displacement: float = 0.01,
    positions: Optional[np.ndarray] = None,
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """``Phi = -dF/dR`` by central differences, in Ha/Bohr^2.

    Costs ``6 N + 1`` ground states for ``N`` atoms with no symmetry reduction --
    13 for a two-atom cell. Every solve shares the reference basis and warm
    starts from the reference state, for the reasons in
    :mod:`d3ft.calc.elastic`: a mask that changes with the geometry would put a
    basis discontinuity into the difference.

    Args:
        config: base config at the geometry to expand about.
        symbols: chemical symbols in crystal order.
        displacement: Cartesian displacement in Bohr. 0.01 is the usual choice;
            it has to be large enough to clear the force noise and small enough
            to stay harmonic.
        positions: expand about these positions instead of the config's, e.g. a
            relaxed geometry. Cartesian, Bohr.
        workdir: where displaced geometries are written.
        verbose: report each displacement.

    Returns:
        Dict with ``phi`` (3N, 3N), the ``residual_asr``, and the reference
        ``forces``.
    """
    from jrystal.calc.opt_utils import create_crystal

    symbols = list(symbols)
    crystal = create_crystal(config)
    ref_positions = (
        np.asarray(crystal.positions) if positions is None else np.asarray(positions)
    )
    natom = ref_positions.shape[0]

    owned = workdir is None
    workdir = tempfile.mkdtemp(prefix="d3ft_hessian_") if owned else workdir
    os.makedirs(workdir, exist_ok=True)

    f_ref, ref_out = _forces_at(config, ref_positions, symbols, None, None, workdir)
    mask = ref_out.freq_mask
    seed = {"pw": ref_out.params_pw, "occ": ref_out.params_occ}
    if verbose:
        print(f"  reference: max|F| = {np.abs(f_ref).max():.3e} Ha/Bohr")

    phi = np.zeros((3 * natom, 3 * natom))
    for atom in range(natom):
        for axis in range(3):
            col = 3 * atom + axis
            plus = ref_positions.copy()
            plus[atom, axis] += displacement
            minus = ref_positions.copy()
            minus[atom, axis] -= displacement
            f_p, _ = _forces_at(config, plus, symbols, seed, mask, workdir)
            f_m, _ = _forces_at(config, minus, symbols, seed, mask, workdir)
            # Phi = -dF/dR, and the column is the response of every force to this
            # one displacement.
            phi[:, col] = -(f_p - f_m).reshape(-1) / (2.0 * displacement)
            if verbose:
                print(f"    atom {atom} axis {axis}: "
                      f"|dF| = {np.abs(f_p - f_m).max():.3e} Ha/Bohr")

    # Symmetrise: Phi is a second derivative of one energy, so it must be, and the
    # asymmetry that survives is a direct read on the finite-difference error.
    asym = float(np.abs(phi - phi.T).max())
    phi = 0.5 * (phi + phi.T)

    # Acoustic sum rule: translating everything costs nothing, so every 3x3 block
    # row must sum to zero. Imposing it is standard; the residual before imposing
    # is the honest error bar.
    blocks = phi.reshape(natom, 3, natom, 3)
    asr = float(np.abs(blocks.sum(axis=2)).max())
    for i in range(natom):
        blocks[i, :, i, :] -= blocks[i].sum(axis=1)

    return {
        "phi": phi,
        "asymmetry": asym,
        "residual_asr": asr,
        "forces_ref": f_ref,
        "displacement": displacement,
        "ground_state": ref_out,
    }


def phonon_frequencies(phi, symbols: Sequence[str]) -> np.ndarray:
    """Zone-centre frequencies in cm^-1, sorted; negative means imaginary.

    Args:
        phi: force constants, (3N, 3N), Ha/Bohr^2.
        symbols: chemical symbols in the same order.

    Returns:
        3N frequencies. The three acoustic modes should sit at zero; how close
        they come is the quality measure on the whole calculation.
    """
    masses = np.array([_MASSES[s] for s in symbols])
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses, 3))
    dynamical = np.asarray(phi) * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    eig = np.linalg.eigvalsh(0.5 * (dynamical + dynamical.T))
    return np.sign(eig) * np.sqrt(np.abs(eig)) * _HA_BOHR2_AMU_TO_CM


def phonon_report(result: Dict[str, Any], symbols: Sequence[str]) -> str:
    """Frequencies with the two diagnostics that qualify them."""
    freqs = phonon_frequencies(result["phi"], symbols)
    acoustic = np.sort(np.abs(freqs))[:3]
    lines = [
        f"  frequencies (cm^-1): " + "  ".join(f"{f:8.1f}" for f in freqs),
        f"  acoustic modes at   : " + "  ".join(f"{f:8.1f}" for f in acoustic)
        + "   (should be 0)",
        f"  Phi asymmetry       : {result['asymmetry']:.3e} Ha/Bohr^2",
        f"  ASR residual        : {result['residual_asr']:.3e} Ha/Bohr^2"
        " (before imposing)",
    ]
    return "\n".join(lines)
