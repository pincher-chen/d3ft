"""Fail-closed explicit-schema acceptance; no inference from missing fields."""
import math

DEFAULTS=dict(charge_error=1e-10,orbital_orthonormality_max=1e-10,
    orbital_outside_max_Ha=1e-6,orbital_commutator_max_Ha=1e-6,
    occupation_max_error=1e-6,maximum_ritz_residual_Ha=1e-8,
    spectral_update_density_L1_per_electron=1e-6,
    spectral_update_free_energy_change_abs_Ha=1e-8)


def check(metrics,thresholds=None):
    limits=dict(DEFAULTS if thresholds is None else thresholds)
    missing=[name for name in limits if metrics.get(name) is None]
    tests={name:math.isfinite(float(metrics[name])) and abs(float(metrics[name]))<=limit
           for name,limit in limits.items() if name not in missing}
    return dict(complete=not missing,passed=not missing and all(tests.values()),
        missing=missing,tests=tests,thresholds=limits,
        scope='Registered numerical gates; not ionic equilibrium, global minimum, or experimental accuracy.')
