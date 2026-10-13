"""Conditional weighted-loss intervals, not certified DFT error bounds.

An interval is rigorous only conditional on supplied observable errors actually
bounding the numerical error. A convergence-pair tolerance alone is not such a
proof. Reference uncertainty is not added here: residual weights remain fixed.
"""
import math


def loss_interval(residuals, errors):
    """Bounds on sum(r_i**2) for independent |delta r_i| <= errors_i."""
    if len(residuals) != len(errors) or not residuals:
        raise ValueError('Require matching nonempty residual/error vectors')
    if any(not math.isfinite(x) for x in list(residuals)+list(errors)):
        raise ValueError('Nonfinite residual/error')
    if any(e < 0 for e in errors):
        raise ValueError('Negative error bound')
    return (math.fsum(max(abs(r)-e, 0.)**2 for r,e in zip(residuals,errors)),
            math.fsum((abs(r)+e)**2 for r,e in zip(residuals,errors)))


def reduction_interval(before, after, before_errors, after_errors):
    if len(before) != len(after):
        raise ValueError('Baseline/candidate dimensions differ')
    lo0, hi0 = loss_interval(before, before_errors)
    lo1, hi1 = loss_interval(after, after_errors)
    lower, upper = lo0-hi1, hi0-lo1
    return {'loss_before_interval': [lo0,hi0], 'loss_after_interval':[lo1,hi1],
            'reduction_interval':[lower,upper],
            'conditional_decision': ('resolved_decrease' if lower>0 else
                                     'resolved_increase' if upper<0 else 'unresolved')}
