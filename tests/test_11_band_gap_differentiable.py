#!/usr/bin/env python3
"""Level 2: the band gap used in a gradient must be the smoothed one.

`E_gap = min(conduction) - max(valence)` is not differentiable at a band
crossing. Autodiff still returns a number there -- whichever branch the tie-break
selected -- with nothing to signal that the value is meaningless. This is not an
edge case for inverse design: the gap is typically extremal exactly where the
band ordering changes, so the optimum tends to sit *on* the crossing.

Checked here:

1. `soft_band_gap` converges to the hard gap as `beta` grows, and its bias stays
   inside `softening_bias_bound` (log-sum-exp overestimates a max and
   underestimates a min, so the soft gap is low by at most
   `(log N_occ + log N_empty)/beta`).
2. Its own gradient agrees with central differences everywhere, including at a
   crossing.
3. At a crossing the *hard* gap's autodiff gradient and its finite difference
   disagree, while the soft one is continuous. This is the regression guard: if
   someone reintroduces `jnp.max`/`jnp.min` into a differentiated path, this
   fails.

Needs no DFT: the eigenvalue crossing is constructed directly, which keeps the
test fast and makes the failure mode unambiguous.

Run:
  python tests/test_11_band_gap_differentiable.py
"""
import numpy as np

BETAS = (10.0, 30.0, 100.0, 300.0)
T_CROSS = 0.15          # where the two conduction levels below cross
TOL_GRAD = 1e-7         # autodiff vs central differences of the soft gap
TOL_LIMIT = 1e-6        # soft gap vs hard gap at the largest beta


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from d3ft.calc import bandgap as bg

    # Four filled levels and four empty ones; two of the empty levels cross at
    # t = T_CROSS, which swaps which one is the CBM.
    occ = jnp.array([1., 1., 1., 1., 0., 0., 0., 0.])

    def eigenvalues(t):
        valence = jnp.array([-6.0, -2.0, -1.9, -1.85])
        conduction = jnp.array([1.0 + 3.0 * t, 1.6 - 1.0 * t, 3.0, 4.2])
        return jnp.concatenate([valence, conduction])

    failures = []

    # ---- 1. limit and bias bound -------------------------------------------
    hard = float(bg.hard_band_gap(eigenvalues(0.0), occ))
    print(f"  hard gap = {hard:.6f}")
    prev_bias = np.inf
    for beta in BETAS:
        soft = float(bg.soft_band_gap(eigenvalues(0.0), occ, beta))
        bias = hard - soft
        bound = bg.softening_bias_bound(occ, beta)
        ok = (0.0 <= bias <= bound + 1e-12) and bias <= prev_bias + 1e-12
        print(f"  beta={beta:6.0f}  soft={soft:.6f}  bias={bias:+.3e}  bound={bound:.3e}")
        if not ok:
            failures.append(
                f"beta={beta}: bias {bias:.3e} outside [0, {bound:.3e}] or not decreasing"
            )
        prev_bias = bias
    if abs(hard - float(bg.soft_band_gap(eigenvalues(0.0), occ, BETAS[-1]))) > TOL_LIMIT:
        failures.append(f"soft gap has not converged to the hard gap at beta={BETAS[-1]}")

    # ---- 2. gradient self-consistency --------------------------------------
    print("  autodiff vs central differences of the soft gap:")
    for beta in (10.0, 50.0):
        f = lambda t: bg.soft_band_gap(eigenvalues(t), occ, beta)
        for t0 in (0.0, T_CROSS, 0.4):
            ad = float(jax.grad(f)(t0))
            h = 1e-6
            fd = (float(f(t0 + h)) - float(f(t0 - h))) / (2 * h)
            rel = abs(ad - fd) / max(abs(fd), 1e-30)
            print(f"    beta={beta:5.0f} t={t0:.2f}  AD={ad:+.8f}  FD={fd:+.8f}  rel={rel:.2e}")
            if not (rel < TOL_GRAD):
                failures.append(f"soft gradient off by {rel:.2e} at beta={beta}, t={t0}")

    # ---- 3. the hazard being guarded against -------------------------------
    fh = lambda t: bg.hard_band_gap(eigenvalues(t), occ)
    ad_at_cross = float(jax.grad(fh)(T_CROSS))
    h = 1e-6
    fd_at_cross = (float(fh(T_CROSS + h)) - float(fh(T_CROSS - h))) / (2 * h)
    print(f"  at the crossing, hard gap: AD={ad_at_cross:+.5f}  FD={fd_at_cross:+.5f}")
    if abs(ad_at_cross - fd_at_cross) < 1e-6:
        failures.append(
            "the hard gap's autodiff gradient agrees with finite differences at the "
            "crossing, so this test is no longer exercising the hazard it guards "
            "(the constructed crossing has probably drifted)"
        )

    fs = lambda t: bg.soft_band_gap(eigenvalues(t), occ, 50.0)
    around = [float(jax.grad(fs)(T_CROSS + d)) for d in (-1e-3, 0.0, 1e-3)]
    jump = max(abs(around[1] - around[0]), abs(around[2] - around[1]))
    print(f"  soft gradient across the crossing: {[round(g, 5) for g in around]}"
          f"  max step {jump:.3e}")
    if jump > 0.5:
        failures.append(f"soft gradient jumps by {jump:.3e} across the crossing")

    for f in failures:
        print(f"  - {f}")
    if failures:
        print("[FAILED] test_11_band_gap_differentiable")
        return 1
    print("[OK]  the soft band gap is differentiable through a crossing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
