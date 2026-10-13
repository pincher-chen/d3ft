"""Error bars for numbers that come out of finite differences.

Three results in this codebase were reported, acted on, and then retracted
because the number quoted carried no error bar of its own:

* A ``dF/d(theta)`` reference at step 0.02 was used as ground truth. Two runs of
  the same AlP configuration gave 5.8374e-06 and 1.1055e-05 -- a factor of 1.9 --
  because the difference being resolved was 0.8% of the stress baseline.
* ``dC44/d(theta)`` from the stationary route was quoted as "34% from the finite
  difference", which invited the reading that a coefficient was wrong. The
  estimator's own scatter turned out to be of order the answer itself, so the
  34% was a draw, not a bias.
* A shear stress was called "30x above noise" after being compared against the
  mean of the diagonal components rather than their spread.

The common shape is quoting a difference without knowing what the inputs to that
difference are worth. This module makes that arithmetic the default: give it the
step sizes and a noise level, and it returns a value with an interval, or
refuses to return one when the sweep does not support it.

The model throughout is the standard one for a central difference::

    D(h) = f' + C h^2 + O(h^4)          truncation, falls as h^2
    noise on D(h) ~ sigma / (h sqrt 2)  solver noise, grows as 1/h

so total error is U-shaped in ``h`` and there is a best step. A sweep that does
not show that shape is telling you the model does not hold -- higher-order terms,
a discontinuity, or noise so large it swamps the whole range -- and the honest
response is to report that rather than to pick a number off the curve.

See :mod:`d3ft.calc.stationary` for the calculation this was extracted from, and
``workflows/`` for the driver scripts that sweep a solver tolerance the same way.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "noise_level",
    "central_difference",
    "richardson",
    "differentiate",
    "tolerance_plateau",
    "format_report",
]


def noise_level(values: Sequence[float]) -> Dict[str, float]:
    """Spread of repeated evaluations of the same quantity.

    The caller does the repeating, because only the caller knows what to vary:
    for a deterministic solver it is the initialisation seed, for an iterative
    one it may be the starting guess. Passing the numbers in keeps this module
    free of any assumption about where the randomness lives.

    Args:
        values: at least two evaluations of nominally the same number.

    Returns:
        ``sigma`` (sample standard deviation), ``mean``, ``spread``
        (max - min) and ``n``. ``sigma`` is what :func:`differentiate` wants.

    Raises:
        ValueError: fewer than two values.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("need at least two repeats to estimate a spread")
    return {
        "sigma": float(v.std(ddof=1)),
        "mean": float(v.mean()),
        "spread": float(v.max() - v.min()),
        "n": int(v.size),
    }


def central_difference(f: Callable[[float], float], x0: float, h: float,
                       cache: Optional[Dict[float, float]] = None) -> float:
    """``(f(x0+h) - f(x0-h)) / (2h)``, reusing cached evaluations.

    Every call to ``f`` here is a ground state or a relaxation, so the cache is
    not a micro-optimisation: sweeping four step sizes without it costs eight
    solves, and several of the points coincide between sweeps.

    Args:
        f: the function, called with a single float.
        x0: base point.
        h: step, must be positive.
        cache: optional dict keyed by the abscissa; updated in place.

    Returns:
        The central difference.
    """
    if h <= 0:
        raise ValueError(f"step must be positive, got {h}")
    cache = {} if cache is None else cache

    def ev(x: float) -> float:
        if x not in cache:
            cache[x] = float(f(x))
        return cache[x]

    return (ev(x0 + h) - ev(x0 - h)) / (2.0 * h)


def richardson(d_coarse: float, d_fine: float, ratio: float,
               order: int = 2) -> float:
    """Extrapolate two central differences to zero step.

    With ``D(h) = f' + C h^order``, two steps in the ratio ``ratio = h_coarse /
    h_fine`` eliminate ``C``.

    Args:
        d_coarse: ``D`` at the larger step.
        d_fine: ``D`` at the smaller step.
        ratio: larger step divided by smaller, greater than one.
        order: leading error power; 2 for a central difference.

    Returns:
        The extrapolated derivative.
    """
    if ratio <= 1.0:
        raise ValueError(f"ratio must exceed 1, got {ratio}")
    r = float(ratio) ** order
    return (r * d_fine - d_coarse) / (r - 1.0)


def differentiate(
    f: Callable[[float], float],
    x0: float,
    steps: Sequence[float],
    sigma: Optional[float] = None,
    order: int = 2,
    consistency: float = 3.0,
) -> Dict[str, object]:
    """A derivative with an interval, or a refusal.

    Evaluates a central difference at every step in ``steps``, fits the
    ``C h^order`` truncation from consecutive pairs, and combines it with the
    ``sigma / (h sqrt 2)`` noise term to pick the step where the two balance.

    The refusal matters more than the estimate. When the truncation coefficients
    implied by different pairs disagree by more than ``consistency``, the
    ``D(h) = f' + C h^order`` picture does not describe the data, and any single
    number read off it is a draw from something wider than its apparent
    precision. That is exactly the situation that produced the retracted 1.9x
    reference, and there the sweep would have said so.

    Args:
        f: function of one variable; each call may be expensive.
        x0: point at which the derivative is wanted.
        steps: at least two positive steps. Order does not matter; they are
            sorted internally, largest first.
        sigma: standard deviation of a single ``f`` evaluation, from
            :func:`noise_level`. Without it the noise term cannot be predicted
            and only the truncation-consistency check is available.
        order: leading error power of the difference formula.
        consistency: how far the pairwise truncation coefficients may spread,
            as a ratio, before the result is marked unreliable.

    Returns:
        Dict with ``estimates`` (step to ``D(h)``), ``richardson`` (from the two
        finest steps), ``truncation_coefficients``, ``best_step``, ``value``,
        ``error``, ``reliable`` (bool) and ``reason`` (why not, when not).
        ``value`` is still filled in when ``reliable`` is False so a caller can
        look at it, but it must not be quoted without the flag.

    Raises:
        ValueError: fewer than two steps, or a non-positive one.
    """
    hs = sorted((float(h) for h in steps), reverse=True)
    if len(hs) < 2:
        raise ValueError("need at least two step sizes to say anything")
    if hs[-1] <= 0:
        raise ValueError("steps must be positive")

    cache: Dict[float, float] = {}
    d = {h: central_difference(f, x0, h, cache=cache) for h in hs}

    # C from each consecutive pair: D(h1) - D(h2) = C (h1^order - h2^order).
    coeffs = []
    for h1, h2 in zip(hs, hs[1:]):
        denom = h1 ** order - h2 ** order
        coeffs.append((d[h1] - d[h2]) / denom if denom else np.nan)
    finite = [c for c in coeffs if np.isfinite(c) and c != 0.0]

    reliable, reason = True, ""
    if len(finite) >= 2:
        mag = np.abs(finite)
        if mag.max() / mag.min() > consistency:
            reliable = False
            reason = (f"truncation coefficient varies by "
                      f"{mag.max() / mag.min():.1f}x across the sweep; "
                      f"the h^{order} model does not hold here")
        elif min(finite) * max(finite) < 0:
            reliable = False
            reason = "truncation coefficient changes sign across the sweep"

    c_trunc = float(np.median(np.abs(finite))) if finite else float("nan")
    extrap = richardson(d[hs[-2]], d[hs[-1]], hs[-2] / hs[-1], order=order)

    # Balance C h^order against sigma / (h sqrt 2) over the steps actually run.
    best_h, best_err = hs[-1], float("nan")
    if sigma is not None and np.isfinite(c_trunc):
        def total(h: float) -> float:
            return c_trunc * h ** order + sigma / (h * np.sqrt(2.0))
        best_h = min(hs, key=total)
        best_err = total(best_h)
        if best_h == hs[0] or best_h == hs[-1]:
            # The minimum sits on the boundary, so the sweep has not bracketed
            # it and the true best step is outside the range that was run.
            reliable = False
            reason = reason or (
                f"error is still falling at the edge of the sweep "
                f"(best step {best_h:g} is an endpoint); widen it")

    value = d[best_h] if sigma is not None and np.isfinite(best_err) else extrap
    return {
        "estimates": d,
        "richardson": extrap,
        "truncation_coefficients": coeffs,
        "truncation_coefficient": c_trunc,
        "best_step": best_h,
        "value": float(value),
        "error": float(best_err),
        "sigma": sigma,
        "reliable": bool(reliable),
        "reason": reason,
        "evaluations": len(cache),
    }


def tolerance_plateau(values_by_tol: Dict[float, float],
                      rel_tol: float = 1.0e-3) -> Dict[str, object]:
    """Has a quantity stopped moving as the solver tolerance is tightened?

    The counterpart of the step sweep for the other knob. ad-dfpt runs its
    elastic constants at several SCF tolerances and puts the tolerance in the
    output filename; this is the same check, reduced to the question a caller
    actually has.

    Args:
        values_by_tol: solver tolerance to the resulting number.
        rel_tol: how close the two tightest values must be, relatively.

    Returns:
        Dict with ``converged`` (bool), ``drift`` (relative change between the
        two tightest tolerances), ``value`` (at the tightest) and ``ordered``
        (the pairs, tightest last).

    Raises:
        ValueError: fewer than two tolerances.
    """
    if len(values_by_tol) < 2:
        raise ValueError("need at least two tolerances")
    ordered = sorted(values_by_tol.items(), key=lambda kv: -kv[0])
    tightest, second = ordered[-1][1], ordered[-2][1]
    drift = abs(tightest - second) / abs(tightest) if tightest else float("inf")
    return {
        "converged": bool(drift <= rel_tol),
        "drift": float(drift),
        "value": float(tightest),
        "ordered": ordered,
    }


def format_report(result: Dict[str, object], name: str = "derivative",
                  unit: str = "") -> str:
    """Render :func:`differentiate` for a log, refusal included.

    Args:
        result: what :func:`differentiate` returned.
        name: what the number is.
        unit: appended to the value.

    Returns:
        A multi-line string. The first line is the headline; when the sweep was
        judged unreliable it says so instead of quoting a bare number.
    """
    est = result["estimates"]
    lines = []
    suffix = f" {unit}" if unit else ""
    if result["reliable"]:
        err = result["error"]
        if np.isfinite(err):
            lines.append(f"{name} = {result['value']:+.6g} +/- {err:.2g}{suffix}"
                         f"  (step {result['best_step']:g})")
        else:
            lines.append(f"{name} = {result['value']:+.6g}{suffix}"
                         f"  (Richardson; no noise estimate supplied)")
    else:
        lines.append(f"{name}: SWEEP UNRELIABLE -- {result['reason']}")
        lines.append(f"  best available value {result['value']:+.6g}{suffix}, "
                     f"not to be quoted without a wider sweep")
    for h in sorted(est, reverse=True):
        lines.append(f"    h = {h:<10.3g} D = {est[h]:+.9e}")
    lines.append(f"    Richardson = {result['richardson']:+.9e}"
                 f"   ({result['evaluations']} evaluations)")
    return "\n".join(lines)
