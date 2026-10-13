#!/usr/bin/env python3
"""Level 1: the occupation must hold exactly the electrons it was given.

`sinkhorn_occupation` ends with a bisection for a global potential shift `s`
solving `sum_n min(exp(base_n + s), capacity) = N_e`. The bracket for that search
used to be a fixed `[-60, 60]`, while `base` lives on the scale of
`band_energies / epsilon` -- a quantity the optimiser is free to grow, with spans
of ~1e4 measured in practice. When the root falls outside the bracket the search
returns its upper end and the result is silently short of charge.

That failure is quiet and expensive. Fewer electrons means a lower energy, so on
AlP the runs that lost the most charge looked like the best minima:

    scheme     seed   E (Ha)      sum(occ)   integral rho
    Sinkhorn   0      -9.336224   7.50000    7.50000
    Sinkhorn   1      -9.313522   7.75000    7.75000
    Sinkhorn   3      -9.367471   7.50000    7.50000
    native     0..3   -9.2848..   8.00000    8.00000

The 1.5 eV spread across "local minima" was largely this, not a property of the
energy landscape.

The bracket is now derived from the data so the root is enclosed by construction.
This test pins that down at the unit level, across the regimes that matter:
large `band_energies`, small `epsilon`, and degenerate levels.

Run:
  python tests/test_12_occupation_charge.py
"""
import numpy as np

TOL = 1e-9


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from d3ft.occupation.sinkhorn import sinkhorn_occupation

    kpt, band = 8, 8
    n_elec = 8.0
    capacity = 2.0 / kpt
    rng = np.random.default_rng(0)

    cases = []
    for scale in (0.1, 1.0, 5.0, 20.0, 100.0):
        for eps in (1.0, 0.117, 1e-3):
            cases.append((f"random scale={scale} eps={eps:g}",
                          rng.normal(size=(1, kpt, band)) * scale, eps))

    # A three-fold degenerate valence top, as in zincblende at Gamma.
    degenerate = np.tile(
        np.array([-6.0, -2.0, -2.0, -2.0, 3.0, 3.0, 3.0, 5.0]), (kpt, 1))[None]
    for eps in (0.5, 0.117, 1e-2, 1e-3):
        cases.append((f"3-fold degenerate eps={eps:g}", degenerate, eps))

    # The regime that first exposed the fixed bracket: base spanning ~1e4.
    cases.append(("wide span (base ~ 1e4)",
                  rng.normal(size=(1, kpt, band)) * 5.0, 1e-3))

    failures = []
    worst = 0.0
    for label, be, eps in cases:
        f = np.asarray(sinkhorn_occupation(
            jnp.asarray(be), n_elec, eps, spin_restricted=True, n_iter=50))
        total = float(f.sum())
        err = abs(total - n_elec)
        worst = max(worst, err)
        over = float(np.max(f)) - capacity
        if err > TOL:
            failures.append(f"{label}: sum(occ) = {total:.6f}, off by {total - n_elec:+.3e}")
        if over > TOL:
            failures.append(f"{label}: occupation exceeds capacity by {over:.3e}")
        if np.any(f < -TOL):
            failures.append(f"{label}: negative occupation {float(np.min(f)):.3e}")

    print(f"  {len(cases)} cases, worst |sum(occ) - N_e| = {worst:.3e}  (tol {TOL:.0e})")

    # ---- gradients must stay finite however large band_energies/epsilon grows.
    #
    # This is what made Sinkhorn unusable for production: the forward value was
    # always right, so nothing looked wrong until Adam poisoned every band-energy
    # parameter. `total_mass` computed `min(exp(base + s), capacity)`; once
    # `base + s` passed 709 the exp overflowed to inf, the clip hid it in the
    # forward pass, and the reverse pass multiplied the zero cotangent that
    # `minimum` sends down the unselected branch by that inf. Since `base` scales
    # as `band_energies / epsilon` and the band energies grow with optimisation,
    # *any* epsilon reached it eventually -- which is why the onset looked like a
    # floor near eps = 2e-2 at 3000 steps and moved with the step count.
    def occupied_energy(band_energies, eps):
        occ = sinkhorn_occupation(
            band_energies, n_elec, eps, spin_restricted=True, n_iter=50)
        return jnp.sum(occ * band_energies)

    print("  gradient finiteness at large band_energies/epsilon:")
    grad_fn = jax.grad(occupied_energy)
    for scale in (1.0, 1e2, 1e3, 1e4, 1e6):
        for eps in (1e-1, 3e-2, 1e-3):
            be = jnp.asarray(rng.normal(size=(1, kpt, band)) * scale)
            span = float(jnp.max(jnp.abs(be)) / eps)
            grad = np.asarray(grad_fn(be, eps))
            occ = np.asarray(sinkhorn_occupation(
                be, n_elec, eps, spin_restricted=True, n_iter=50))
            if not np.all(np.isfinite(grad)):
                failures.append(
                    f"non-finite gradient at scale={scale:g} eps={eps:g} "
                    f"(max|base| ~ {span:.3g})")
            if abs(float(occ.sum()) - n_elec) > TOL:
                failures.append(
                    f"charge lost at scale={scale:g} eps={eps:g}: "
                    f"sum(occ) = {float(occ.sum()):.6f}")
    print(f"    15 combinations up to max|base| ~ 2.5e9: "
          f"{'all finite' if not failures else 'FAILURES'}")

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_12_occupation_charge")
        return 1
    print("[OK]  occupations conserve charge, respect capacity, and differentiate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
