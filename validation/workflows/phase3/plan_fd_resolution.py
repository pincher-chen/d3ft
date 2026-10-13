"""Rank coordinate FD experiments by frozen-Jacobian signal, not agreement.

Uses only the frozen training rows. Missing observables are explicitly retained
as missing; linear predictions are not convergence or FD acceptance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path


def plan(data):
    observables = ('a0_A', 'B_GPa', 'C11_minus_C12_GPa', 'C44_GPa')
    results = []
    for material in sorted({r['material'] for r in data['rows']}):
        rows = [r for r in data['rows'] if r['material'] == material]
        for index, name in enumerate(data['parameter_names']):
            components = []
            for row in rows:
                baseline = row['reference'] * (1 + row['residual'])
                budget = abs(baseline) * 1e-4 if row['observable'] == 'a0_A' else .1
                # gradient is d(observable/reference)/dp; alpha is scaled.
                slope = abs(row['gradient'][index] * row['reference'])
                signal_per_alpha = 2 * data['parameter_scales'][index] * slope
                components.append(dict(observable=row['observable'], budget=budget,
                    difference_per_scaled_alpha=signal_per_alpha,
                    alpha_at_budget=budget / signal_per_alpha if signal_per_alpha else None))
            missing = sorted(set(observables) - {r['observable'] for r in rows})
            thresholds = [r['alpha_at_budget'] for r in components]
            required = max(thresholds) if not missing and all(x is not None for x in thresholds) else None
            results.append(dict(material=material, parameter=name, missing_observables=missing,
                components=components, alpha_at_all_four_budgets=required,
                twofold_signal_target_alpha=2 * required if required is not None else None))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('jacobian', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.jacobian.read_text())
    rows = plan(data)
    ranked = {name: sorted([r for r in rows if r['parameter'] == name
                           and r['alpha_at_all_four_budgets'] is not None],
                          key=lambda r: r['alpha_at_all_four_budgets'])
              for name in data['parameter_names']}
    report = dict(scope='Linear experimental-design predictions only; no FD acceptance. '
                  'Twofold signal is a planning margin, not a changed acceptance gate. '
                  'Large alpha requires multiple-step truncation checks and physical parameter checks.',
                  source=str(args.jacobian.resolve()),
                  source_sha256=hashlib.sha256(args.jacobian.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, ranked_complete_materials=ranked)
    with args.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    for name, candidates in ranked.items():
        print(name, [(r['material'], round(r['twofold_signal_target_alpha'], 5))
                     for r in candidates[:3]])


if __name__ == '__main__':
    main()
