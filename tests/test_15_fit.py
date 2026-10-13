"""The C44 fit's arithmetic, without DFT.

The expensive half of `d3ft.calc.fit` is a Newton loop over ground states. The
half that is easy to get wrong is the reduction from two sheared directions to
C44, the central difference in theta, and the sign of the Newton step -- all
checkable in milliseconds against a synthetic solid whose answer is known.

The gradient here is a finite difference over theta with each point re-relaxed.
The cheap alternative, the implicit chain, is withdrawn: it omits the electronic
response and comes out 362% wrong with the opposite sign. See
`d3ft/calc/implicit.py`.
"""

import numpy as np

from d3ft.calc.fit import c44_and_gradient, fit_theta_to_c44


class AttrDict(dict):
    """dict whose keys are also attributes, readable and writable.

    The fit does ``cfg.d3ft.theta_nonlocal = ...``, which a plain dict cannot
    take, so the stand-in has to support attribute assignment as ml_collections'
    ConfigDict does.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __deepcopy__(self, memo):
        out = AttrDict()
        for k, v in self.items():
            out[k] = AttrDict(v) if isinstance(v, dict) else v
        return out


def ToyConfig():
    return AttrDict(d3ft=AttrDict(theta_nonlocal=0.0))


def make_fake_solver(c44_at_zero, slope, monkey):
    """Patch the relaxation and stress with an exactly linear response in theta.

    C44(theta) = c44_at_zero + slope * theta, expressed through the sigma_yz that
    the real code reduces, so the reduction and the central difference are both
    exercised rather than bypassed.
    """
    from d3ft.calc import fit as fit_mod
    from d3ft.calc.elastic import _HA_BOHR3_TO_GPA

    delta = 1.0e-2
    scale = _HA_BOHR3_TO_GPA / (2.0 * delta)
    state = {"theta": 0.0}

    def fake_relax(cfg, symbols, strain=None, **kwargs):
        state["theta"] = float(cfg["d3ft"]["theta_nonlocal"])
        state["sign"] = 1.0 if strain[1, 2] > 0 else -1.0
        return {"ground_state": None, "steps": 2, "converged": True,
                "max_force": 1e-5}

    def fake_stress(out, **kwargs):
        sig = np.zeros((3, 3))
        val = state["sign"] * (c44_at_zero + slope * state["theta"]) / scale
        sig[1, 2] = sig[2, 1] = val
        return sig

    monkey.append((fit_mod, "relax_internal_coordinates",
                   fit_mod.relax_internal_coordinates))
    monkey.append((fit_mod, "stress_from_ground_state",
                   fit_mod.stress_from_ground_state))
    fit_mod.relax_internal_coordinates = fake_relax
    fit_mod.stress_from_ground_state = fake_stress


def restore(monkey):
    for mod, name, orig in monkey:
        setattr(mod, name, orig)


def check_reduction(failures):
    """C44 and dC44/dtheta come back as the synthetic solid defines them."""
    monkey = []
    make_fake_solver(87.7, -140.0, monkey)
    try:
        r = c44_and_gradient(ToyConfig(), ["Al", "P"], theta=0.0, verbose=False)
        for name, got, want in (("C44", r["C44"], 87.7),
                                ("dC44_dtheta", r["dC44_dtheta"], -140.0)):
            if abs(got - want) > 1e-6:
                failures.append(f"{name}: got {got:.6f}, want {want:.6f}")
        print(f"  C44 = {r['C44']:.4f}, dC44/dtheta = {r['dC44_dtheta']:.4f} "
              f"(central difference recovers the exact slope)")
    finally:
        restore(monkey)


def check_newton_reaches_target(failures):
    """An exactly linear response should be solved in one step."""
    monkey = []
    make_fake_solver(87.7, -140.0, monkey)
    try:
        r = fit_theta_to_c44(ToyConfig(), ["Al", "P"], target=61.5,
                             theta0=0.0, max_steps=5, tol=1e-6, verbose=False)
        want = (61.5 - 87.7) / -140.0
        if not r["converged"]:
            failures.append("Newton did not converge on a linear response")
        if abs(r["theta"] - want) > 1e-6:
            failures.append(f"theta {r['theta']:.6f}, want {want:.6f}")
        if r["steps"] > 2:
            failures.append(f"took {r['steps']} steps on a linear response")
        print(f"  converged in {r['steps']} step(s) to theta = {r['theta']:.6f} "
              f"(exact {want:.6f})")
    finally:
        restore(monkey)


def main():
    print("[test_15] C44 fit: reduction, central difference in theta, Newton step")
    failures = []
    for check in (check_reduction, check_newton_reaches_target):
        check(failures)
    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_15_fit")
        return 1
    print("[OK]  reduction, central difference and Newton step are exact on a linear response")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
