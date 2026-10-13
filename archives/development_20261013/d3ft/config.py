"""Configuration handling for D3FT.

D3FT reuses :class:`jrystal.config.JrystalConfigDict` and adds a single ``d3ft`` sub-key
containing module-specific configuration. The :func:`load_config` helper merges a YAML
file with the defaults and returns a fully-populated ``ml_collections.ConfigDict``.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from ml_collections import ConfigDict


def _default_d3ft_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.optim = ConfigDict()
    cfg.optim.enabled = True
    cfg.optim.preconditioner = "kinetic"
    cfg.optim.retraction = "qr"
    cfg.optim.mu = 1.0
    # "atomic": diagonalise H_KS at the superposition of atomic valence densities
    # (PP_RHOATOM) and start from its lowest eigenvectors. "random": random
    # Stiefel point. Atomic is the default because it is deterministic, not
    # because it finds the best minimum -- it does not; see the measurements in
    # ground_state.calc and PLAN.md blocker 5.4.
    cfg.optim.init = "atomic"
    # Adam at a constant rate does not settle: it orbits the minimum at a radius
    # set by the step size, which is why the seed-to-seed energy spread, the
    # acoustic sum rule and the C11 / C12 elastic constants all carried noise
    # that more epochs did not remove. Three seeds under "cosine" land on the
    # same AlP energy to six decimals, against a 0.994 mRy spread under
    # "constant"; set "constant" to reproduce results published before 2026-09-03.
    cfg.optim.lr_schedule = "cosine"
    cfg.optim.lr_final = 1.0e-4

    cfg.theta_nonlocal = 0.0
    # Scalar multiplier on the nonlocal (KB) D matrix, baked into the SCF solve.
    # A stand-in learnable Hamiltonian parameter for d3ft.calc.implicit; zero
    # reproduces the unperturbed pseudopotential exactly and is what every other
    # measurement in this project uses.

    cfg.kpoints = ConfigDict()
    # "gamma": Gamma-centred, closed under the crystal point group at every n.
    # "mp": ASE's half-shifted Monkhorst-Pack mesh, which for an fcc primitive
    # cell breaks 36 of the 48 cubic operations and puts a spurious off-diagonal
    # term into the stress at 90% of the diagonal. See d3ft.grid.kpoints.
    cfg.kpoints.mesh = "gamma"

    cfg.parallel = ConfigDict()
    # Shard the wavefunction coefficients over the k-point axis, QE's "pool"
    # level. This is the only level that shortens the internal-coordinate
    # relaxation, whose steps are strictly sequential; batching independent
    # geometries cannot touch it. Requires d3ft.parallel.plain_fft_patch().
    cfg.parallel.k_pool = False

    cfg.occupation = ConfigDict()
    cfg.occupation.enabled = True
    cfg.occupation.method = "sinkhorn"
    cfg.occupation.epsilon_init = 1.0
    cfg.occupation.epsilon_final = 1.0e-3
    cfg.occupation.epsilon_schedule = "log"
    cfg.occupation.sinkhorn_iters = 50
    # Separate step size for the band-energy parameters. They enter only through
    # ``band_energies / epsilon``, so their natural scale is epsilon, not the
    # scale of the plane-wave coefficients -- sharing one learning rate steps
    # them by ~lr regardless of epsilon, which at epsilon = 1e-3 is ten times the
    # whole scale on which the occupations vary. None keeps the shared value.
    cfg.occupation.learning_rate = None

    cfg.spectrum = ConfigDict()
    cfg.spectrum.enabled = True
    cfg.spectrum.n_lanczos_iters = 64
    cfg.spectrum.n_probes = 8
    cfg.spectrum.implicit_grad = True

    cfg.free_energy = ConfigDict()
    cfg.free_energy.enabled = False
    cfg.free_energy.temp_init = 100.0

    return cfg


def load_config(path: str | Path) -> ConfigDict:
    """Load a D3FT config file and merge with the defaults.

    Merges (in order):
    1. Jrystal's ``default_config`` dict.
    2. D3FT's default ``d3ft`` sub-config.
    3. The user YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        A :class:`ml_collections.ConfigDict` containing both Jrystal and D3FT keys.
    """
    from jrystal.config import default_config as jr_defaults

    merged: dict = dict(jr_defaults)
    merged["d3ft"] = {
        "optim": {
            "enabled": True,
            "preconditioner": "kinetic",
            "retraction": "qr",
            "mu": 1.0,
            "init": "atomic",
            # See _default_d3ft_config. "constant" keeps the original behaviour.
            "lr_schedule": "cosine",
            "lr_final": 1.0e-4,
        },
        "theta_nonlocal": 0.0,
        "kpoints": {
            # See d3ft.grid.kpoints. "mp" reproduces Jrystal's sampling and is
            # kept for comparison; it is not symmetric under the point group.
            "mesh": "gamma",
        },
        "parallel": {
            # See _default_d3ft_config. Needs plain_fft_patch() around the call.
            "k_pool": False,
        },
        "occupation": {
            "enabled": True,
            "method": "sinkhorn",
            "epsilon_init": 1.0,
            "epsilon_final": 1.0e-3,
            "epsilon_schedule": "log",
            "sinkhorn_iters": 50,
            # See _default_d3ft_config: the band energies live on the scale of
            # epsilon, not of the plane-wave coefficients. None shares the
            # global learning rate.
            "learning_rate": None,
        },
        "spectrum": {
            "enabled": True,
            "n_lanczos_iters": 64,
            "n_probes": 8,
            "implicit_grad": True,
        },
        "free_energy": {
            # When enabled, the optimisation target becomes the Mermin free energy
            # F = E - T*S (entropy from Fermi-Dirac occupations) with T annealed from
            # temp_init down to `smearing`, matching Jrystal's native pipeline. The energy
            # history and convergence check still track the total energy E. Default off.
            "enabled": False,
            "temp_init": 100.0,
        },
    }

    with open(path) as f:
        user = yaml.safe_load(f) or {}

    _deep_update(merged, user)

    cfg = ConfigDict(merged)
    if cfg.band_structure_empty_bands is None:
        cfg.band_structure_empty_bands = cfg.empty_bands
    return cfg


def _deep_update(target: dict, source: dict) -> None:
    # Merge happens while `target` is still a plain dict (before ConfigDict construction),
    # so recurse on dict, not ConfigDict. Otherwise a partially-specified nested block in
    # the YAML (e.g. `d3ft` without `free_energy`) would wholesale-replace the defaults and
    # drop keys the user did not mention.
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v
