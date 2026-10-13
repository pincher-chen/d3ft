#!/usr/bin/env python3
"""Level 4 regression: the KS Hamiltonian must be invariant under rigid translation.

Shifting every atom by the same vector d is a change of origin: the eigenvalues of
T + V_loc + V_nl cannot move. This catches the whole class of bugs where the local
and nonlocal parts of the pseudopotential are placed at inconsistent positions,
without needing any external reference data, for any system.

It is the system-independent guard for the Kleinman-Bylander conjugation bug: using
beta instead of conj(beta) in the inner projection turns exp(-i(G+k).R) into
exp(+i(G+k).R), so V_nl sits at -R while V_loc sits at +R. Under a rigid shift d the
local part transforms as U(d) V U(d)^dag and the broken nonlocal part as
U(d)^dag V U(d); the sum is then not a single unitary conjugation and the spectrum
moves. An atom at the origin hides this (+R and -R coincide), which is why elemental
cells looked healthy while binary cells did not.

Deliberately run at a small cutoff -- translation invariance is exact at any basis
size, so a cheap basis makes this a fast unit test.

Run:
  python tests/test_08_translation_invariance.py
"""
import os
import sys
from pathlib import Path

import numpy as np

CUTOFF = 8          # Ha; small on purpose, see docstring
GRID = 24
A0 = 5.467          # Angstrom
SHIFT_FRAC = np.array([0.137, 0.041, 0.283])   # arbitrary, not a symmetry op
TOL_EV = 1e-6


def _root():
    """Directory holding the curated validation data package.

    Order: the D3FT_ALP_ROOT override, then this repository's ``validation/``,
    then the cluster-style ``<root>/software/d3ft/tests/`` layout.
    """
    env = os.environ.get("D3FT_ALP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    repo_validation = here.parents[1] / "validation"
    if repo_validation.is_dir():
        return repo_validation
    return here.parents[3]


def main():
    root = _root()
    pp_dir = root / "pseudopotentials" / "alp"
    base_yaml = root / "config" / "config_base.yaml"
    if not (pp_dir / "Al.upf").exists() or not base_yaml.exists():
        print(f"[SKIP] curated AlP data package not found under {root}")
        return 0

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from d3ft.config import load_config
    from d3ft.spectrum.ks_matvec import make_ks_hvp
    from jrystal.calc.opt_utils import (
        create_crystal, create_freq_mask, create_grids,
        create_pseudopotential, set_env_params,
    )
    from d3ft import pseudopotential as d3pp

    HA2EV = 27.211386245988
    tmp = root / "results" / "tmp" / "alp_diagnostics"
    tmp.mkdir(parents=True, exist_ok=True)
    xyz = tmp / "alp_translation.xyz"
    h, p = A0 / 2.0, A0 / 4.0
    xyz.write_text(
        "2\n"
        f'Lattice="0 {h:.10f} {h:.10f} {h:.10f} 0 {h:.10f} {h:.10f} {h:.10f} 0" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Al 0 0 0\n"
        f"P {p:.10f} {p:.10f} {p:.10f}\n"
    )

    cfg = load_config(str(base_yaml))
    cfg.crystal = None
    cfg.crystal_file_path_path = str(xyz)
    cfg.spin = 0
    cfg.k_grid_sizes = 1
    cfg.cutoff_energy = CUTOFF
    cfg.grid_sizes = GRID
    cfg.xc = "lda_x"
    cfg.empty_bands = 4
    cfg.seed = 0
    cfg.verbose = False
    cfg.use_pseudopotential = True
    cfg.pseudopotential_type = "nc"
    cfg.pseudopotential_file_dir = str(pp_dir)
    cfg.d3ft.optim.enabled = True
    cfg.d3ft.optim.preconditioner = "kinetic"
    cfg.d3ft.occupation.enabled = True

    set_env_params(cfg)
    cr = create_crystal(cfg)
    g_vec, _, _ = create_grids(cfg)
    fm = create_freq_mask(cfg)
    pp = create_pseudopotential(cfg)
    vol = float(cr.vol)

    cell = np.asarray(cr.cell_vectors, dtype=float)
    shift = SHIFT_FRAC @ cell
    pos0 = np.asarray(cr.positions, dtype=float)

    # A uniform density: Hartree/XC are then translation invariant by construction,
    # so the test isolates the relative placement of V_loc and V_nl.
    grid_shape = tuple(np.asarray(g_vec).shape[:-1])
    n_elec = float(np.sum(np.asarray(pp.valence_charges, dtype=float)))
    rho = jnp.full((1, *grid_shape), n_elec / vol)
    k_cart = jnp.zeros((3,))

    print(f"[RUN] cutoff={CUTOFF} Ha grid={GRID}^3  shift(frac)={SHIFT_FRAC.tolist()}")
    spectra = []
    for name, pos in (("unshifted", pos0), ("shifted", pos0 + shift)):
        pos_j = jnp.asarray(pos)
        pot_loc = d3pp.potential_local_reciprocal(
            pos_j, g_vec, pp.r_grid, pp.local_potential_grid,
            pp.local_potential_charge, vol,
        )
        gk_vec = jnp.expand_dims(k_cart[None, :], (1, 2, 3)) \
            + jnp.expand_dims(g_vec, 0)
        beta_gk = d3pp.beta_grid(
            pp.r_grid, pp.nonlocal_beta_grid, pp.nonlocal_angular_momentum,
            gk_vec, d3pp.kmax_for(gk_vec),
        )
        pot_nl = d3pp.projectors_reciprocal(
            pos_j, gk_vec, pp.nonlocal_angular_momentum, beta_gk,
        )
        nl_d = d3pp.concat_d_matrix(pp.nonlocal_d_matrix)
        pn_k = jax.tree.map(lambda x: x[0], pot_nl)
        hop, n_g = make_ks_hvp(
            rho, pot_loc, pn_k, g_vec, k_cart, vol, fm,
            xc=cfg.xc, nl_d_matrix=nl_d,
        )
        mat = jax.vmap(hop, in_axes=1, out_axes=1)(jnp.eye(n_g, dtype=jnp.complex128))
        w = np.asarray(
            jnp.linalg.eigvalsh(0.5 * (mat + jnp.conj(mat).T))
        ).real * HA2EV
        spectra.append(w)
        print(f"  {name:10s} n_g={n_g}  lowest6={np.round(w[:6], 6)}")

    d = np.abs(spectra[0] - spectra[1])
    print(f"  max |Δeigenvalue| = {d.max():.3e} eV   (tol {TOL_EV:.0e})")
    if d.max() > TOL_EV:
        i = int(np.argmax(d))
        print(f"  worst at band {i}: {spectra[0][i]:.6f} vs {spectra[1][i]:.6f} eV")
        print("  V_loc and V_nl are not placed at the same atomic positions")
        print("[FAILED] test_08_translation_invariance")
        return 1
    print("[OK]  spectrum is invariant under rigid translation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
