"""Raw force-sum check with an explicit, bounded absolute force scale.

An optional floor is tied to the ionic stopping tolerance, never to a measured
force error. With floor=1e-6 and relative_tolerance=.05, the absolute net-force
budget near convergence is 5e-8 Ha/Bohr. Forces are never projected or corrected.
This conservation check does not certify individual-force quadrature accuracy.
"""
import numpy as np

IONIC_FORCE_TOLERANCE = 1e-6


def check_force_sum(forces, floor=0., relative_tolerance=.05):
    if not np.isfinite(floor) or not 0 <= floor <= IONIC_FORCE_TOLERANCE:
        raise ValueError('Force scale floor must be between 0 and the ionic tolerance 1e-6')
    force = np.asarray(forces, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or not np.isfinite(force).all():
        raise ValueError('Expected finite atomic forces with shape (natoms,3)')
    largest = float(np.abs(force).max())
    net = float(np.abs(force.sum(axis=0)).max())
    limit = relative_tolerance * max(largest, floor)
    result = dict(max_atomic_force=largest, max_net_force=net, net_force_limit=limit,
                  relative_asr=net/largest if largest else 0., force_scale_floor=floor,
                  ionic_force_tolerance=IONIC_FORCE_TOLERANCE,
                  passes_net_force=net <= limit,
                  passes_ionic_force=largest < IONIC_FORCE_TOLERANCE)
    if net > limit:
        raise ValueError(f'Raw acoustic sum rule failed: net {net:.6g} > budget {limit:.6g} Ha/Bohr')
    return result
