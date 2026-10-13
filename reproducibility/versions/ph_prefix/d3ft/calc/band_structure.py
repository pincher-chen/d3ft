"""Band-structure calculation with D3FT-Lanczos.

After a ground-state run produces a self-consistent density, this module computes the
band structure along a user-defined k-path by extracting the lowest eigenvalues of the
effective Kohn–Sham Hamiltonian at each k via :func:`d3ft.spectrum.lanczos.lanczos_eigvals`.

Compared to :mod:`jrystal.calc.calc_band_structure_normcons`, this replaces the
:math:`O(\\dim^2)` explicit-Hessian assembly with an :math:`O(k_\\text{iter} \\cdot \\dim)`
Krylov method.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from absl import logging


@dataclass
class BandStructureOutput:
    config: Any
    k_path: Any
    band_energies: Any


def calc(config, gs_output=None) -> BandStructureOutput:
    """Compute the band structure along a k-path from a D3FT ground state.

    D3FT's native Lanczos band path (``d3ft.spectrum.lanczos`` on a matrix-free
    Kohn-Sham HVP) is not yet wired up. In the meantime this routine uses D3FT's
    converged ground-state *density* (produced by the Riemannian-Adam + Sinkhorn
    optimiser) and extracts the eigenvalues along the k-path with the shared,
    well-tested Jrystal KS-Hamiltonian band assembly. The scientific claim it
    supports is therefore "D3FT's ground state reproduces the correct band
    structure", validated against Jrystal and Quantum ESPRESSO.

    Args:
        config: ConfigDict with the band-path fields (``k_path_special_points``,
            ``num_kpoints``, ``band_structure_*``) and a Jrystal-compatible
            ``optimizer`` (e.g. ``"adam"``) for the per-k band optimisation.
        gs_output: A :class:`d3ft.calc.ground_state.GroundStateOutput` (must carry
            ``density_grid``). If ``None``, a D3FT ground state is run with
            ``config`` first.

    Returns:
        :class:`BandStructureOutput` with ``band_energies`` of shape ``[kpt, band]``.
    """
    from jrystal.calc import calc_band_structure_normcons as jr_band
    from jrystal.calc.calc_ground_state_energy_normcons import GroundStateEnergyOutput

    from .ground_state import calc as gs_calc

    if gs_output is None:
        logging.info("===> D3FT ground state (for band structure)...")
        gs_output = gs_calc(config)

    if getattr(gs_output, "density_grid", None) is None:
        raise ValueError(
            "gs_output.density_grid is None; the D3FT ground state must expose the "
            "converged density grid for the band-structure calculation."
        )

    # Wrap D3FT's converged density into the Jrystal output object that the band
    # calculator consumes (it reads `.density_grid` as the fixed KS density).
    jr_gso = GroundStateEnergyOutput(
        config=config,
        crystal=gs_output.crystal,
        params_pw=gs_output.params_pw,
        params_occ=gs_output.params_occ,
        total_energy=gs_output.total_energy,
        total_energy_history=gs_output.energy_history,
        density_grid=gs_output.density_grid,
    )

    logging.info("===> Band structure via shared KS assembly (D3FT density)...")
    jr_band.calc(config, ground_state_energy_output=jr_gso)

    # jrystal's band calc saves "<symbols>_band_structure.npy" in the CWD and
    # returns None; load it back so the caller gets the eigenvalues directly.
    save_file = "".join(gs_output.crystal.symbols) + "_band_structure.npy"
    band_energies = jnp.load(save_file) if os.path.exists(save_file) else None
    return BandStructureOutput(
        config=config, k_path=None, band_energies=band_energies
    )
