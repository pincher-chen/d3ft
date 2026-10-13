#!/usr/bin/env python3
"""Level 4: gradients of the nonlocal pseudopotential energy must be usable.

Direct differentiation is the reason d3ft exists, so the projector path has to be
differentiable end to end. Three things are checked, in increasing difficulty:

1. d/d(atomic positions) -- flows through the structure factor exp(-i(G+k).R),
   which has always been traced. Verified against central differences.
2. d/d(cell) must be FINITE. It used to be NaN for every cell entry: reciprocal
   grids contain G + k = 0 and points on the z axis, where |x| and atan2(y, x)
   have undefined derivatives. Repairing those with jnp.where on the *result*
   does not help, because the NaN is already in the backward pass and
   0 * NaN = NaN poisons the sum even when the cotangent is zero.
3. d/d(cell) of the projector table itself, which requires beta_l(|G+k|) to be
   evaluated inside the JAX graph (jrystal.pseudopotential.beta.beta_sbt_*)
   rather than precomputed with NumPy/SciPy.

The composite cell gradient is only checked loosely. arccos is clipped away from
+-1 to keep its derivative finite, which puts a kink at grid points lying on the
polar axis; central differences straddle that kink and stop converging below
about 1e-4. The polar-axis singularity is intrinsic to spherical coordinates, so
this is a tolerance, not a defect to chase.

Run:
  python tests/test_09_pseudopotential_gradients.py
"""
import os
import sys
from pathlib import Path

import numpy as np

A0 = 5.467          # Angstrom
GRID = 12           # small on purpose; differentiability is size-independent
TOL_POS = 1e-6      # autodiff vs central differences, relative
TOL_TABLE = 1e-7    # beta / spherical-harmonic sub-paths, relative
TOL_CELL = 5e-2     # composite, limited by the polar-axis kink (see docstring)


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
    if not (pp_dir / "Al.upf").exists():
        print(f"[SKIP] AlP pseudopotentials not found under {root}")
        return 0

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jrystal._src.crystal import Crystal
    from jrystal._src.grid import g_vectors
    from jrystal.pseudopotential.dataclass import NormConservingPseudopotential

    from d3ft import pseudopotential as d3pp
    from d3ft.pseudopotential import beta_sbt_reciprocal, beta_sbt_table

    tmp = root / "results" / "tmp" / "alp_diagnostics"
    tmp.mkdir(parents=True, exist_ok=True)
    xyz = tmp / "alp_gradients.xyz"
    h, p = A0 / 2.0, A0 / 4.0
    xyz.write_text(
        "2\n"
        f'Lattice="0 {h:.10f} {h:.10f} {h:.10f} 0 {h:.10f} {h:.10f} {h:.10f} 0" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Al 0 0 0\n"
        f"P {p:.10f} {p:.10f} {p:.10f}\n"
    )

    cr = Crystal.create_from_file(str(xyz))
    pp = NormConservingPseudopotential.create(cr, str(pp_dir))
    cell0 = np.asarray(cr.cell_vectors, dtype=float)
    pos0 = np.asarray(cr.positions, dtype=float)
    k_cart = jnp.zeros((1, 3))
    nl_d = d3pp.concat_d_matrix(pp.nonlocal_d_matrix)

    gv0 = np.asarray(g_vectors(cell0, [GRID] * 3))
    kmax = float(np.max(np.sqrt(np.sum(gv0 ** 2, -1)))) * 1.6
    table = [
        beta_sbt_table(r, b, l, kmax)
        for r, b, l in zip(pp.r_grid, pp.nonlocal_beta_grid,
                           pp.nonlocal_angular_momentum)
    ]

    rng = np.random.default_rng(0)
    psi = rng.normal(size=gv0.shape[:-1]) + 1j * rng.normal(size=gv0.shape[:-1])
    psi = jnp.asarray(psi / np.linalg.norm(psi))

    def gk_of(cell):
        g = g_vectors(cell, [GRID] * 3)
        return jnp.expand_dims(k_cart, (1, 2, 3)) + jnp.expand_dims(g, 0)

    def beta_of(cell):
        gk = gk_of(cell)
        return [beta_sbt_reciprocal(jnp.asarray(k), jnp.asarray(bk), gk)
                for k, bk in table]

    def energy(positions, cell):
        g = g_vectors(cell, [GRID] * 3)
        vol = jnp.abs(jnp.linalg.det(cell))
        pot = d3pp.projectors_reciprocal(
            positions, gk_of(cell), pp.nonlocal_angular_momentum, beta_of(cell),
        )
        nl = jax.tree.map(lambda x: x[0], pot)
        f = jnp.einsum("xyz,bmxyz->bm", psi, jnp.conj(nl))
        return jnp.real(jnp.einsum("bm,bc,cm->", jnp.conj(f), nl_d, f)) / vol

    def central(f, x, n, eps):
        out = []
        for i in range(n):
            xp, xm = x.copy(), x.copy()
            xp.flat[i] += eps
            xm.flat[i] -= eps
            out.append((f(xp) - f(xm)) / (2 * eps))
        return np.array(out)

    def rel(a, b):
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

    failures = []

    # 1. positions
    g_ad = np.asarray(jax.grad(lambda P: energy(P, jnp.asarray(cell0)))(
        jnp.asarray(pos0))).ravel()[:6]
    g_fd = central(lambda P: float(energy(jnp.asarray(P), jnp.asarray(cell0))),
                   pos0, 6, 1e-5)
    e = rel(g_ad, g_fd)
    print(f"  d/d(positions)      rel.err = {e:.3e}  (tol {TOL_POS:.0e})")
    if not (e < TOL_POS):
        failures.append(f"position gradient off by {e:.3e}")

    # 2. beta table sub-path, tight
    f_beta = lambda C: sum(jnp.sum(b ** 2) for b in beta_of(C))
    b_ad = np.asarray(jax.grad(f_beta)(jnp.asarray(cell0))).ravel()[:6]
    b_fd = central(lambda C: float(f_beta(jnp.asarray(C))), cell0, 6, 1e-4)
    e = rel(b_ad, b_fd)
    print(f"  d/d(cell) of beta   rel.err = {e:.3e}  (tol {TOL_TABLE:.0e})")
    if not (e < TOL_TABLE):
        failures.append(f"beta table cell gradient off by {e:.3e}")

    # 3. composite cell gradient: finite first, then loosely accurate
    c_ad = np.asarray(jax.grad(lambda C: energy(jnp.asarray(pos0), C))(
        jnp.asarray(cell0)))
    n_nan = int(np.sum(~np.isfinite(c_ad)))
    print(f"  d/d(cell) non-finite entries = {n_nan}/{c_ad.size}")
    if n_nan:
        failures.append(
            f"{n_nan} non-finite cell-gradient entries; a NaN-producing "
            "derivative (|x| or atan2 at the origin / polar axis) is back in the graph"
        )
    else:
        c_fd = central(
            lambda C: float(energy(jnp.asarray(pos0), jnp.asarray(C))),
            cell0, 6, 1e-4)
        e = rel(c_ad.ravel()[:6], c_fd)
        print(f"  d/d(cell) composite rel.err = {e:.3e}  (tol {TOL_CELL:.0e})")
        if not (e < TOL_CELL):
            failures.append(f"composite cell gradient off by {e:.3e}")

    for f in failures:
        print(f"  - {f}")
    if failures:
        print("[FAILED] test_09_pseudopotential_gradients")
        return 1
    print("[OK]  pseudopotential gradients are finite and match finite differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
