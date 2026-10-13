"""Declared production FD gates; resolution budgets are not statistical bounds."""
import math

OBSERVABLES = {'a0_A', 'B_GPa', 'C11_minus_C12_GPa', 'C44_GPa'}


def component(observable, predicted, actual, baseline_a0, rtol):
    if observable not in OBSERVABLES:
        raise ValueError('Unknown FD observable')
    if not all(math.isfinite(x) for x in (predicted, actual, baseline_a0, rtol)):
        raise ValueError('Nonfinite FD input')
    if baseline_a0 <= 0 or rtol < 0:
        raise ValueError('Invalid lattice constant or tolerance')
    floor = 1e-7 if observable == 'a0_A' else 1e-3
    budget = 1e-4 * baseline_a0 if observable == 'a0_A' else .1
    error = abs(predicted - actual)
    agreement = error <= rtol * abs(actual) + floor
    resolved = abs(actual) > budget
    return dict(observable=observable, predicted_difference=predicted,
                actual_difference=actual, relative_error=error / max(abs(actual), floor),
                passed=agreement, agreement_passed=agreement,
                numerical_resolution_budget=budget, signal_to_budget=abs(actual) / budget,
                signal_resolved=resolved, joint_gate_passed=agreement and resolved)


def aggregate(rows):
    keys = [r['observable'] for r in rows]
    complete = len(keys) == 4 and set(keys) == OBSERVABLES
    return dict(all_four_observables_present=complete,
                agreement_passed=bool(rows) and all(r['agreement_passed'] for r in rows),
                joint_gate_passed=complete and all(r['joint_gate_passed'] for r in rows))
