#!/usr/bin/env python3
"""Level 4: the stress tensor is the strain derivative of the free energy.

Checks the machinery, and separately reports whether the state it was evaluated
at justifies calling the result a stress.

  sigma_ab = (1/Omega) dF/d(eps_ab)

Two things have to hold and only one of them is this module's responsibility:

1. **The derivative is right.** ``jax.grad`` of the free energy with respect to
   strain must match central differences of that same free energy. This is what
   the test asserts.

2. **The state is stationary.** The Hellmann-Feynman step assumes
   ``dF/d(params) = 0``, so that the total derivative with respect to strain
   collapses to the partial one. At a non-converged point the number returned is
   the partial derivative at an arbitrary point, not a stress -- and the
   finite-difference check above cannot see that, because it validates the
   derivative rather than the state. The residual is therefore printed and
   compared against the scale of the stress, but not asserted on: how tightly
   D3FT's direct optimisation converges is a property of the optimiser, not of
   this code path.

Run:
  python tests/test_10_stress.py
"""
import os
import sys
from pathlib import Path

import numpy as np

A0 = 5.467          # Angstrom
CUTOFF = 10         # Ha; small on purpose, the identity is cutoff-independent
K_GRID = 2
EPOCH = 200
TOL_FD = 1e-5       # autodiff vs central differences of F, relative


def _root():
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
        print(f"[SKIP] AlP data package not found under {root}")
        return 0

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from d3ft.calc import stress as st
    from d3ft.calc.ground_state import calc
    from d3ft.config import load_config

    tmp = root / "results" / "tmp" / "alp_diagnostics"
    tmp.mkdir(parents=True, exist_ok=True)
    xyz = tmp / "alp_stress.xyz"
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
    cfg.k_grid_sizes = K_GRID
    cfg.cutoff_energy = CUTOFF
    cfg.xc = "lda_x"
    cfg.empty_bands = 4
    cfg.epoch = EPOCH
    cfg.seed = 0
    cfg.verbose = False
    cfg.use_pseudopotential = True
    cfg.pseudopotential_type = "nc"
    cfg.pseudopotential_file_dir = str(pp_dir)
    cfg.d3ft.optim.enabled = True
    cfg.d3ft.optim.preconditioner = "kinetic"
    cfg.d3ft.occupation.enabled = True

    print(f"[RUN] AlP  cutoff={CUTOFF} Ha  k={K_GRID}^3  epoch={EPOCH}")
    out = calc(cfg)

    kwargs = dict(
        params_pw=out.params_pw, params_occ=out.params_occ, config=out.config,
        crystal=out.crystal, freq_mask=out.freq_mask, make_coeff=out.make_coeff,
        get_occupation=out.get_occupation, pseudopot=out.pseudopot,
        beta_tables=out.beta_tables, kmax=out.beta_kmax, grid_kmax=out.grid_kmax,
        k_frac=out.k_frac, temperature=out.temperature, eps=out.final_eps,
    )
    sigma = np.asarray(st.stress_tensor(**kwargs))
    vol = float(abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))

    failures = []

    n_bad = int(np.sum(~np.isfinite(sigma)))
    print(f"  non-finite entries      = {n_bad}/{sigma.size}")
    if n_bad:
        failures.append(f"{n_bad} non-finite stress entries")

    asym = float(np.abs(sigma - sigma.T).max())
    print(f"  asymmetry               = {asym:.3e}")
    if asym > 1e-12:
        failures.append(f"stress not symmetric ({asym:.3e})")

    def f_of(e):
        return float(st.free_energy_at_strain(jnp.asarray(e), **kwargs))

    fd = np.zeros((3, 3))
    eps = 1e-5
    for a in range(3):
        for b in range(3):
            ep, em = np.zeros((3, 3)), np.zeros((3, 3))
            ep[a, b] += eps
            em[a, b] -= eps
            fd[a, b] = (f_of(ep) - f_of(em)) / (2 * eps) / vol
    fd = 0.5 * (fd + fd.T)
    rel = float(np.abs(sigma - fd).max() / max(np.abs(fd).max(), 1e-30))
    print(f"  autodiff vs central-diff = {rel:.3e}  (tol {TOL_FD:.0e})")
    if not (rel < TOL_FD):
        failures.append(f"stress differs from finite differences by {rel:.3e}")

    # Reported, not asserted: see the module docstring.
    resid = st.variational_residual(**kwargs)
    scale = float(np.abs(sigma).max()) * vol
    print(f"  pressure                 = {st.pressure(sigma):+.4f} GPa")
    print(f"  variational residual     = {resid:.3e}  vs |sigma|*vol = {scale:.3e}")
    if scale > 0 and resid > 0.05 * scale:
        print("  NOTE: the run is not at a stationary point, so this number is a "
              "strain derivative rather than a physical stress. The identity "
              "checked above still holds; converge the ground state before using "
              "the value.")

    for f in failures:
        print(f"  - {f}")
    if failures:
        print("[FAILED] test_10_stress")
        return 1
    print("[OK]  stress is the strain derivative of the free energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
