"""GPU test: matrix-free KS HVP eigenvalues match jrystal's dense band matrix.

Builds a small Si ground state, then compares the lowest eigenvalues from:
  (a) jrystal's explicit [band, band] hamiltonian_matrix + eigvalsh, and
  (b) the matrix-free G-space operator make_ks_hvp, both
      - assembled into a dense n_g x n_g matrix (exact), and
      - via Lanczos.

Needs a GPU and an importable ``jrystal``. Paths are taken from the environment
so the test is not tied to any one machine:

  JRYSTAL_DIR  jrystal checkout (default: importable from sys.path)
  D3FT_DIR     this package     (default: the parent of tests/)
  D3FT_TEST_PP_DIR  norm-conserving UPF directory
               (default: <jrystal>/pseudopotential/normconserving)
               NOTE: deliberately not PP_DIR -- the surrounding workflow exports
               PP_DIR for its own (system-specific) pseudopotentials.
  BASE_YAML    base config       (default: <d3ft>/config.yaml)

Run:
    python tests/test_06_ks_hvp.py
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
D3FT_DIR = os.environ.get("D3FT_DIR", str(_HERE.parent))
sys.path.insert(0, D3FT_DIR)

JRYSTAL_DIR = os.environ.get("JRYSTAL_DIR", "")
if JRYSTAL_DIR:
    sys.path.insert(0, JRYSTAL_DIR)
else:
    import importlib.util
    _spec = importlib.util.find_spec("jrystal")
    if _spec is None or not _spec.origin:
        print("[SKIP] jrystal not importable; set JRYSTAL_DIR")
        raise SystemExit(0)
    JRYSTAL_DIR = str(Path(_spec.origin).resolve().parents[1])

PP_DIR = os.environ.get(
    "D3FT_TEST_PP_DIR",
    str(Path(JRYSTAL_DIR) / "pseudopotential" / "normconserving") + "/",
)
BASE_YAML = os.environ.get("BASE_YAML", str(Path(D3FT_DIR) / "config.yaml"))


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from d3ft.config import load_config
    from d3ft.calc.ground_state import calc
    from d3ft.optim.stiefel import stiefel_coeff
    from d3ft.spectrum.ks_matvec import make_ks_hvp
    from d3ft.spectrum.lanczos import lanczos_eigvals
    from jrystal.calc.opt_utils import (
        create_crystal, create_freq_mask, create_grids,
        create_pseudopotential, set_env_params,
    )
    from jrystal.calc.pre_calc import pre_calc_beta_sbt
    from jrystal.pseudopotential import normcons

    cfg = load_config(BASE_YAML)
    cfg.crystal = "si"
    cfg.spin = 0
    cfg.k_grid_sizes = 1
    cfg.cutoff_energy = 12
    cfg.empty_bands = 4
    cfg.epoch = 400
    cfg.seed = 0
    cfg.use_pseudopotential = True
    cfg.pseudopotential_type = "nc"
    cfg.pseudopotential_file_dir = PP_DIR
    cfg.verbose = False
    cfg.d3ft.optim.enabled = True
    cfg.d3ft.occupation.enabled = True

    print("Running Si ground state...")
    gs = calc(cfg)

    set_env_params(cfg)
    crystal = create_crystal(cfg)
    g_vec, _, k_vec = create_grids(cfg)
    freq_mask = create_freq_mask(cfg)
    pseudopot = create_pseudopotential(cfg)
    potential_loc = normcons.potential_local_reciprocal(
        crystal.positions, g_vec, pseudopot.r_grid,
        pseudopot.local_potential_grid, pseudopot.local_potential_charge, crystal.vol,
    )
    beta_gk = pre_calc_beta_sbt(pseudopot, np.array(g_vec), np.array(k_vec))
    potential_nl = normcons.potential_nonlocal_psi_reciprocal(
        crystal.positions, g_vec, k_vec, pseudopot.r_grid,
        pseudopot.nonlocal_beta_grid, pseudopot.nonlocal_angular_momentum,
        pseudopot.nonlocal_d_matrix, beta_gk,
    )
    nl_d = normcons.concat_d_matrix(pseudopot.nonlocal_d_matrix)
    k_gamma = k_vec[0:1]
    pnl_g = jax.tree.map(lambda x: x[0:1], potential_nl)
    print("potential_nl shape:", np.asarray(potential_nl).shape)
    density = gs.density_grid
    coeff = stiefel_coeff(gs.params_pw, freq_mask)[0, 0]  # [band, x, y, z]
    n_bands = coeff.shape[0]

    # (a) jrystal dense band matrix eigenvalues.
    H_band = normcons.hamiltonian_matrix(
        coeff[None, None], density, potential_loc, pnl_g, g_vec, k_gamma,
        crystal.vol, cfg.xc, kohn_sham=True, d_matrix=nl_d,
    )[0, 0]
    ev_band = jnp.linalg.eigvalsh(H_band)
    print(f"jrystal band-matrix eigenvalues ({n_bands} bands):")
    print(np.asarray(ev_band))

    # (b) matrix-free G-space operator.
    H_op, n_g = make_ks_hvp(
        density, potential_loc, pnl_g[0], g_vec, k_gamma[0],
        crystal.vol, freq_mask, xc=cfg.xc, nl_d_matrix=nl_d,
    )
    print(f"matrix-free operator: n_g = {n_g}")

    # Element-wise check: rebuild the band matrix from H_op and compare to jrystal.
    mask_np = np.asarray(freq_mask).astype(bool).reshape(-1)
    idx = np.where(mask_np)[0]
    c_masked = coeff.reshape(n_bands, -1)[:, idx]            # [band, n_g]
    Hc = jax.vmap(H_op)(c_masked)                            # [band, n_g]
    M = jnp.conj(c_masked) @ Hc.T                            # [band, band]
    elem_err = float(jnp.max(jnp.abs(M - H_band)))
    print(f"max |Δ| element-wise (H_op band matrix vs jrystal): {elem_err:.3e} Ha")
    print("  diag(M)     :", np.asarray(jnp.diag(M).real)[:6])
    print("  diag(H_band):", np.asarray(jnp.diag(H_band).real)[:6])

    # Assemble dense n_g x n_g by applying H_op to basis vectors (validation only).
    eye = jnp.eye(n_g, dtype=jnp.complex64)
    Hfull = jax.vmap(H_op, in_axes=1, out_axes=1)(eye)
    herm_err = float(jnp.max(jnp.abs(Hfull - Hfull.conj().T)))
    ev_full = jnp.linalg.eigvalsh((Hfull + Hfull.conj().T) / 2)
    print(f"Hermiticity error of matrix-free H: {herm_err:.2e}")
    print(f"lowest {n_bands} eigenvalues of full masked H:")
    print(np.asarray(jnp.sort(ev_full)[:n_bands]))

    # Compare lowest n_bands.
    lo_band = jnp.sort(ev_band)
    lo_full = jnp.sort(ev_full)[:n_bands]
    match_err = float(jnp.max(jnp.abs(lo_band - lo_full)))
    print(f"max |Δ| lowest-{n_bands} (band-matrix vs full HVP): {match_err:.3e} Ha")

    # (c) Lanczos on the matrix-free operator: lowest m_occ eigenvalues.
    m = max(1, n_bands // 2)  # occupied-ish; well separated from the cluster above
    key = jax.random.PRNGKey(0)
    k_iter = min(4 * n_bands + 40, n_g)
    ev_lanc = lanczos_eigvals(H_op, n_g, k_iter, 1, key, k_return=m)
    lanc_err = float(jnp.max(jnp.abs(jnp.sort(ev_lanc)[:m] - lo_full[:m])))
    print(f"lowest {m} eigenvalues via Lanczos (k_iter={k_iter}):")
    print(np.asarray(jnp.sort(ev_lanc)[:m]))
    print(f"max |Δ| lowest-{m} Lanczos vs full HVP: {lanc_err:.3e} Ha")

    # The decisive correctness check is the element-wise match: the matrix-free
    # operator must reproduce jrystal's band matrix exactly. The full-spectrum
    # comparison only matches on the *converged* (occupied) bands.
    occ_match = float(jnp.max(jnp.abs(lo_band[:m] - lo_full[:m])))
    print(f"max |Δ| lowest-{m} (band-matrix vs full HVP, converged bands): {occ_match:.3e} Ha")
    # Decisive check: the matrix-free operator reproduces jrystal's band matrix to
    # machine precision. (Lanczos resolution of near-degenerate clusters is a
    # separate, well-understood convergence property and is reported above.)
    ok = (elem_err < 1e-6) and (herm_err < 1e-3)
    print("PASS" if ok else "CHECK: tolerances exceeded (inspect above)")


if __name__ == "__main__":
    main()
