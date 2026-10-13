"""The tunable exchange-correlation path must equal the one it replaces.

``xc_energy_tunable`` re-expresses :func:`jrystal._src.energy.xc_energy` composed
with :func:`jrystal._src.xc.xc_density`, restricted to the unpolarised
``kohn_sham=False`` GGA branch, so that libxc's parameter struct can be passed in
rather than built inside. That is the only way to reach PBE's ``kappa`` and
``mu`` with ``jax.grad``, and reaching them turned out to matter: ``mu`` is the
strongest single knob on C44 in this system, ahead of every pseudopotential
channel.

The risk is obvious. A re-implementation that quietly disagrees with the original
poisons every energy computed through it, and the disagreement would look like
physics. So the first check pins them against each other bit-for-bit on a
synthetic density, and it runs without DFT so it runs on every commit rather than
only when someone remembers.

The rest of the file guards the reason the module exists: that an override
actually reaches the functional, and that the derivative with respect to it is
real rather than a zero returned by a silently ignored argument -- the failure
mode that would make a screen report "this parameter does nothing".
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jrystal._src.energy import xc_energy  # noqa: E402
from jrystal._src.grid import g_vectors  # noqa: E402

from d3ft.calc.xc_tunable import (  # noqa: E402
    default_xc_params,
    xc_energy_tunable,
)

XC = "gga_x_pbe+gga_c_pbe"
N = 8


def _problem():
    """A smooth positive density on a small cubic grid, and its reciprocal grid."""
    a = 10.0
    cell = jnp.eye(3) * a
    g_vec = g_vectors(cell, np.array([N, N, N]))
    x = jnp.linspace(0.0, 2 * jnp.pi, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    # Positive and varying, so the gradient term is exercised; the offset keeps
    # it away from the functional's density threshold.
    rho = 0.3 + 0.1 * (jnp.sin(X) * jnp.cos(Y) + jnp.sin(Z))
    return rho[None, ...], g_vec, float(a ** 3)


def check_reproduces_the_stock_energy(failures):
    rho, g_vec, vol = _problem()
    stock = float(xc_energy(rho, g_vec, vol, XC, kohn_sham=False))
    base = default_xc_params(XC)
    tuned = float(xc_energy_tunable(rho, g_vec, vol, XC, xc_params=base))
    none = float(xc_energy_tunable(rho, g_vec, vol, XC, xc_params=None))
    print(f"  stock={stock:.15e}")
    print(f"  tunable(defaults)={tuned:.15e}  diff={abs(tuned-stock):.2e}")
    print(f"  tunable(None)     ={none:.15e}  diff={abs(none-stock):.2e}")
    for label, got in (("defaults", tuned), ("None", none)):
        rel = abs(got - stock) / abs(stock)
        if rel > 1e-12:
            failures.append(
                f"tunable path with {label} differs from jrystal by {rel:.2e} "
                "relative; every energy through it would be wrong")


def check_reproduces_the_stock_energy_with_nlcc(failures):
    """The same equality with a pseudo-core density added.

    The plain check runs without NLCC, but every real calculation here has one,
    and the two paths add the core density in separate pieces of code. A control
    run reproduced the stock bulk modulus to 0.05 GPa rather than exactly, which
    is inside the 0.3 GPa noise but is not the zero that a bit-identical path
    should give; this is the first place to look if that ever grows.
    """
    rho, g_vec, vol = _problem()
    core = 0.05 * (1.0 + 0.2 * jnp.cos(
        jnp.linspace(0.0, 2 * jnp.pi, N, endpoint=False)))[:, None, None] \
        * jnp.ones((N, N, N))
    stock = float(xc_energy(rho, g_vec, vol, XC, kohn_sham=False,
                            core_density_grid=core))
    tuned = float(xc_energy_tunable(rho, g_vec, vol, XC, core_density_grid=core,
                                    xc_params=default_xc_params(XC)))
    rel = abs(tuned - stock) / abs(stock)
    print(f"  with NLCC: stock={stock:.12e} tunable={tuned:.12e} rel={rel:.2e}")
    if rel > 1e-12:
        failures.append(
            f"the NLCC path differs from jrystal by {rel:.2e} relative")


def check_defaults_are_the_published_pbe_constants(failures):
    base = default_xc_params(XC)
    print(f"  {base}")
    want = {"gga_x_pbe": {"kappa": 0.804, "mu": 0.2195149727645171},
            "gga_c_pbe": {"beta": 0.06672455060314922, "gamma": 0.031090690869655}}
    for func, params in want.items():
        for k, v in params.items():
            got = base.get(func, {}).get(k)
            if got is None:
                failures.append(f"{func}.{k} missing from default_xc_params")
            elif abs(got - v) > 1e-9:
                failures.append(f"{func}.{k} = {got}, expected the PBE value {v}")


def check_an_override_actually_moves_the_energy(failures):
    """A silently ignored override would make every screen report 'no coupling'."""
    rho, g_vec, vol = _problem()
    base = default_xc_params(XC)
    e0 = float(xc_energy_tunable(rho, g_vec, vol, XC, xc_params=base))
    for func, key in (("gga_x_pbe", "kappa"), ("gga_x_pbe", "mu"),
                      ("gga_c_pbe", "beta"), ("gga_c_pbe", "gamma")):
        bumped = {f: dict(p) for f, p in base.items()}
        bumped[func][key] = base[func][key] * 1.2
        e1 = float(xc_energy_tunable(rho, g_vec, vol, XC, xc_params=bumped))
        rel = abs(e1 - e0) / abs(e0)
        print(f"  {func}.{key} +20% -> dE/E = {rel:.3e}")
        if rel < 1e-9:
            failures.append(
                f"raising {func}.{key} by 20% changed the energy by {rel:.1e}; "
                "the override is not reaching the functional")


def check_gradient_matches_a_finite_difference(failures):
    """The screen reads these derivatives; a wrong one is worse than none."""
    rho, g_vec, vol = _problem()
    base = default_xc_params(XC)
    keys = [("gga_x_pbe", "kappa"), ("gga_x_pbe", "mu"),
            ("gga_c_pbe", "beta"), ("gga_c_pbe", "gamma")]

    def energy(v):
        xp = {f: dict(p) for f, p in base.items()}
        for i, (f, k) in enumerate(keys):
            xp[f][k] = base[f][k] + v[i]
        return xc_energy_tunable(rho, g_vec, vol, XC, xc_params=xp)

    grad = np.asarray(jax.grad(energy)(jnp.zeros(len(keys))))
    h = 1e-6
    e0 = abs(float(energy(jnp.zeros(len(keys)))))
    for i, (f, k) in enumerate(keys):
        plus = np.zeros(len(keys)); plus[i] = h
        fd = (float(energy(jnp.asarray(plus))) -
              float(energy(jnp.asarray(-plus)))) / (2 * h)
        # The finite difference is the less accurate side here, and by how much
        # depends on the derivative: cancelling two energies of size |E| at
        # spacing h leaves a roundoff floor of about eps |E| / h, which for
        # kappa -- whose derivative is four orders below mu's -- is 2e-4
        # relative. A fixed 1e-6 tolerance would fail the code for the
        # difference's own noise. Compare against that floor instead.
        floor = 2.3e-16 * e0 / h
        tol = max(1e-6 * abs(fd), 3.0 * floor)
        bad = abs(grad[i] - fd) > tol
        print(f"  d/d({f.split('_')[-1]}:{k}): AD={grad[i]:+.6e} "
              f"FD={fd:+.6e} |diff|={abs(grad[i]-fd):.2e} tol={tol:.2e}"
              f"{'  <-- FAIL' if bad else ''}")
        if bad:
            failures.append(
                f"AD and FD disagree on {f}.{k} by {abs(grad[i]-fd):.2e}, "
                f"beyond the difference's own {tol:.2e} floor")


def check_unsupported_cases_are_refused(failures):
    """Silence on a case this does not cover would be the dangerous answer."""
    rho, g_vec, vol = _problem()
    for label, call in (
        ("spin-polarised",
         lambda: xc_energy_tunable(jnp.concatenate([rho, rho]), g_vec, vol, XC)),
        ("LDA",
         lambda: xc_energy_tunable(rho, g_vec, vol, "lda_x")),
    ):
        try:
            call()
        except ValueError as exc:
            print(f"  {label}: refused ({str(exc)[:60]}...)")
        else:
            failures.append(f"{label} was accepted; it is not covered by this path")


def main():
    failures = []
    for check in (
        check_reproduces_the_stock_energy,
        check_reproduces_the_stock_energy_with_nlcc,
        check_defaults_are_the_published_pbe_constants,
        check_an_override_actually_moves_the_energy,
        check_gradient_matches_a_finite_difference,
        check_unsupported_cases_are_refused,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_20_xc_tunable")
        return 1
    print("[OK]  the tunable XC path equals the stock one, and its parameters "
          "are reachable and differentiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
