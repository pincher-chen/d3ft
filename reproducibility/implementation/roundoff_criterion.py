"""User-declared supplementary diagnostic; never changes solver acceptance."""
import math

DECISIONS = ('accepted', 'reductions', 'trial_count', 'rate', 'next_rate', 'streak')


def margin(record):
    return record['F_trial'] - record['F_before'] - 1e-4 * record['rate'] * record['slope']


def compare(reference, candidate):
    mr, mc = margin(reference), margin(candidate)
    finite = all(math.isfinite(d[k]) for d in (reference, candidate)
                 for k in ('F_before', 'F_trial', 'rate', 'slope', 'rounding'))
    sensitive = finite and (abs(mr) <= 10 * reference['rounding']
                           or abs(mc) <= 10 * candidate['rounding'])
    same = all(reference[k] == candidate[k] for k in DECISIONS)
    return dict(passed=finite and (same or sensitive), decisions_identical=same,
                roundoff_sensitive=sensitive, reference_margin_Ha=mr,
                candidate_margin_Ha=mc,
                reference_10rho_Ha=10 * reference['rounding'],
                candidate_10rho_Ha=10 * candidate['rounding'])
