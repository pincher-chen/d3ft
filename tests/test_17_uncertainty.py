"""A finite difference must come with the error bar of its own inputs.

Three retractions in this project came from the same omission, so this guards
the arithmetic rather than any physics. The case that matters most is the last
check here: a ``dF/d(theta)`` reference whose two independent measurements were
5.8374e-06 and 1.1055e-05. That number was used as ground truth for a day. Fed
through this module it is rejected outright -- the spread is 62% of the mean, so
nothing built on top of it can be resolved.

Pure numpy: no SCF, no GPU, runs in well under a second.
"""

import numpy as np

from d3ft.utils.uncertainty import (
    central_difference,
    differentiate,
    format_report,
    noise_level,
    richardson,
    tolerance_plateau,
)


def _wiggle(x):
    """Reproducible pseudo-noise, uncorrelated between abscissae.

    Seeded from the float's bit pattern rather than ``hash``, which Python salts
    per process -- a test that fails one run in ten teaches nothing.

    All 64 bits go into the seed. Truncating to the low 31 discards the sign bit,
    which makes ``+h`` and ``-h`` draw the same number; the noise then cancels
    exactly in a central difference and the test passes while measuring nothing.
    """
    bits = int(np.frombuffer(np.float64(x).tobytes(), dtype=np.uint64)[0])
    return float(np.random.default_rng(bits).standard_normal())


def check_richardson_is_exact_on_a_cubic(failures):
    """``D(h) = f' + 2h^2`` exactly for this f, so extrapolation must be exact."""
    f = lambda x: 2.0 * x ** 3 - 3.0 * x ** 2 + x
    exact = lambda x: 6.0 * x ** 2 - 6.0 * x + 1.0
    x0 = 0.35
    d1 = central_difference(f, x0, 0.2)
    d2 = central_difference(f, x0, 0.1)
    got = richardson(d1, d2, ratio=2.0)
    err = abs(got - exact(x0))
    print(f"  cubic: D(0.2)={d1:+.9f} D(0.1)={d2:+.9f} -> {got:+.12f} "
          f"(exact {exact(x0):+.12f}, err {err:.1e})")
    if err > 1e-12:
        failures.append(f"Richardson not exact on a cubic: error {err:.2e}")
    # And the un-extrapolated values really are off by the predicted 2h^2.
    for h, d in ((0.2, d1), (0.1, d2)):
        predicted = exact(x0) + 2.0 * h ** 2
        if abs(d - predicted) > 1e-12:
            failures.append(f"D({h}) = {d} does not match f' + 2h^2 = {predicted}")


def check_clean_sweep_is_accepted(failures):
    """A noiseless function over a sane step range must pass and be accurate."""
    x0 = 0.7
    r = differentiate(np.sin, x0, steps=[1e-1, 5e-2, 2.5e-2])
    err = abs(r["value"] - np.cos(x0))
    print(f"  sin at {x0}: value={r['value']:.12f} exact={np.cos(x0):.12f} "
          f"err={err:.1e} reliable={r['reliable']}")
    if not r["reliable"]:
        failures.append(f"clean sweep rejected: {r['reason']}")
    if err > 1e-8:
        failures.append(f"Richardson on sin is only accurate to {err:.2e}")
    if r["evaluations"] != 6:
        failures.append(f"expected 6 evaluations, cache gave {r['evaluations']}")


def check_cache_prevents_repeat_work(failures):
    """The base point is shared, and each call is a ground state in real use."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.sin(x)

    cache = {}
    central_difference(f, 0.4, 0.1, cache=cache)
    central_difference(f, 0.4, 0.1, cache=cache)
    print(f"  cache: two identical central differences cost {calls['n']} evaluations")
    if calls["n"] != 2:
        failures.append(f"cache did not deduplicate: {calls['n']} calls, expected 2")


def check_noise_is_detected(failures):
    """Noise far below the function value still destroys a small-step sweep."""
    amp = 1.0e-6

    def noisy(x):
        return np.sin(x) + amp * _wiggle(x)

    r = differentiate(noisy, 0.7, steps=[1e-3, 1e-4, 1e-5], sigma=amp)
    print(f"  noisy: reliable={r['reliable']}  reason={r['reason'] or '-'}")
    for h in sorted(r["estimates"], reverse=True):
        print(f"    h={h:<8g} D={r['estimates'][h]:+.6f}")
    if r["reliable"]:
        failures.append("a sweep dominated by 1e-6 noise was accepted as reliable")


def check_endpoint_optimum_is_flagged(failures):
    """If the best step is at the edge, the sweep did not bracket anything."""
    r = differentiate(np.sin, 0.7, steps=[1e-1, 5e-2], sigma=1e-12)
    print(f"  endpoint: best_step={r['best_step']:g} reliable={r['reliable']} "
          f"reason={r['reason'] or '-'}")
    if r["reliable"]:
        failures.append("an unbracketed optimum was accepted as reliable")


def check_noise_level_statistics(failures):
    v = [1.0, 1.2, 0.8, 1.1]
    s = noise_level(v)
    print(f"  noise_level: mean={s['mean']:.4f} sigma={s['sigma']:.4f} "
          f"spread={s['spread']:.4f} n={s['n']}")
    if abs(s["mean"] - 1.025) > 1e-12:
        failures.append(f"mean wrong: {s['mean']}")
    if abs(s["spread"] - 0.4) > 1e-12:
        failures.append(f"spread wrong: {s['spread']}")
    if abs(s["sigma"] - float(np.std(v, ddof=1))) > 1e-12:
        failures.append("sigma is not the sample standard deviation")
    try:
        noise_level([1.0])
        failures.append("noise_level accepted a single value")
    except ValueError:
        pass


def check_tolerance_plateau(failures):
    settled = tolerance_plateau({1e-4: 61.9, 1e-6: 61.42, 1e-8: 61.417})
    moving = tolerance_plateau({1e-4: 61.9, 1e-6: 61.42, 1e-8: 60.1})
    print(f"  tolerance: settled drift={settled['drift']:.2e} "
          f"converged={settled['converged']}; "
          f"moving drift={moving['drift']:.2e} converged={moving['converged']}")
    if not settled["converged"]:
        failures.append("a settled tolerance sweep was called unconverged")
    if moving["converged"]:
        failures.append("a drifting tolerance sweep was called converged")
    if abs(settled["value"] - 61.417) > 1e-12:
        failures.append("tolerance_plateau did not report the tightest value")


def check_the_retracted_reference_is_rejected(failures):
    """The real incident: two runs of one AlP configuration, 1.9x apart.

    Nothing downstream of a number with this spread can be resolved, and the
    module has to say so rather than average them into a false precision.
    """
    runs = [5.8374e-06, 1.1055e-05]
    s = noise_level(runs)
    rel = s["spread"] / s["mean"]
    print(f"  retracted reference: mean={s['mean']:.4e} sigma={s['sigma']:.4e} "
          f"spread/mean={rel:.1%}")
    if rel < 0.5:
        failures.append(f"expected the historical spread to exceed 50%, got {rel:.1%}")

    # With sigma that large, a sweep over the steps actually used is refused.
    # The historical signal was 0.8% of its own baseline. Reproduce that ratio:
    # a slope of 1e-5 measured through noise of 3.7e-6 per evaluation puts the
    # noise on D(h) an order of magnitude above the slope at every step used.
    def fake(x):
        return 1.0e-5 * x + s["sigma"] * _wiggle(x)

    r = differentiate(fake, 0.0, steps=[0.02, 0.01, 0.005], sigma=s["sigma"])
    worst = s["sigma"] / (0.005 * np.sqrt(2.0))
    print(f"    predicted noise on D at h=0.005: {worst:.2e} vs slope 1.00e-05 "
          f"-> {worst / 1e-5:.0f}x the quantity being measured")
    if worst < 1e-5:
        failures.append("historical configuration no longer reproduces the incident")
    print(f"    sweep at the original steps: reliable={r['reliable']} "
          f"({r['reason'] or '-'})")
    if r["reliable"]:
        failures.append("the historical configuration was accepted as reliable")
    print("    " + format_report(r, name="dF/dtheta", unit="Ha").replace("\n", "\n    "))


def main():
    failures = []
    for check in (
        check_richardson_is_exact_on_a_cubic,
        check_clean_sweep_is_accepted,
        check_cache_prevents_repeat_work,
        check_noise_is_detected,
        check_endpoint_optimum_is_flagged,
        check_noise_level_statistics,
        check_tolerance_plateau,
        check_the_retracted_reference_is_rejected,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_17_uncertainty")
        return 1
    print("[OK]  finite differences carry their own error bar, and refuse when they cannot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
