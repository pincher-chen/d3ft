"""Independent target/constraint checks before the physical six-strain run."""
import json
from pathlib import Path
import numpy as np
from six_strain_design import proposal

origin = np.array([.003, -.002, .001, .002, -.001, .0015])
gradient = np.array([-8.46, -7.65, -9.65, -.094, .032, -.075])
gap, target = 4.41, 4.51
candidate = proposal(origin, origin, gradient, gap, target, .01, .03)
assert abs(gap + gradient @ (candidate-origin) - target) < 1e-12
assert np.count_nonzero(abs(candidate-origin) > 1e-8) == 6
# This is a minimum-norm update: an independently constructed null-space
# displacement preserves the linear target but increases the displacement.
null = np.array([1., -.5, .3, .2, -.1, .7])
null -= gradient * (gradient @ null) / (gradient @ gradient)
alternative = candidate-origin + .001*null
assert abs(gradient @ alternative - .1) < 1e-12
assert np.linalg.norm(alternative) > np.linalg.norm(candidate-origin)
# Independent nonlinear objective, with known symmetric curvature.
curvature = np.diag([3., -2., 4., 1., 2., -1.])
theta = origin.copy()
errors = [.1]
for iteration in range(3):
    displacement = theta-origin
    actual = gap + gradient @ displacement + .5*displacement @ curvature @ displacement
    derivative = gradient + curvature @ displacement
    theta = proposal(theta, origin, derivative, actual, target, .01, .03)
    displacement = theta-origin
    actual = gap + gradient @ displacement + .5*displacement @ curvature @ displacement
    errors.append(abs(actual-target))
    assert errors[-1] <= errors[-2] + 1e-14
    if errors[-1] < 1e-10:
        break
assert errors[-1] < 1e-10
# At/near the total boundary, projections must retain both radii.
rng = np.random.default_rng(91)
checks = 0
for _ in range(100):
    position = rng.normal(size=6)
    position *= .03 / np.linalg.norm(position)
    theta = origin + position
    trial = proposal(theta, origin, rng.normal(size=6), 0., 10., .01, .03)
    assert np.linalg.norm(trial-origin) <= .03+1e-14
    assert np.linalg.norm(trial-theta) <= .01+1e-14
    checks += 1
for bad in [np.zeros(6), np.full(6, np.nan)]:
    try:
        proposal(origin, origin, bad, gap, target, .01, .03)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('Invalid gradient accepted')
print(json.dumps(dict(passed=True, linear_target_reached=True, six_components_updated=True,
    nonlinear_errors_eV=errors, boundary_checks=checks, invalid_gradients_rejected=True,
    scope='Independent finite optimizer checks; not a physical DFT design result.'), indent=2))
