"""Compare forward and reverse AD on the identical stationary-energy closure.

Only the sensitivity function's local JAX binding is substituted. Nested
energy routines and the production module keep their original globals. This
does not validate an electronic-response mixed derivative or an active d channel.
"""
import types
import numpy as np
import jax
import jax.numpy as jnp


def forward_gradient(fun, argnums=0):
    """Sequential scalar jacfwd columns avoid seven simultaneous tangent states."""
    def calculate(*args):
        vector = args[argnums]
        if vector.ndim != 1:
            raise ValueError('Expected a one-dimensional parameter vector')
        def column(index):
            direction = jax.nn.one_hot(index, vector.size, dtype=vector.dtype)
            def along(step):
                changed = list(args)
                changed[argnums] = vector + step * direction
                return fun(*changed)
            return jax.jacfwd(along)(jnp.zeros((), dtype=vector.dtype))
        return jax.lax.map(column, jnp.arange(vector.size))
    return calculate


class ForwardJax:
    def __getattr__(self, name):
        return forward_gradient if name == 'grad' else getattr(jax, name)


def compare(out, xc_base, reverse, n_channels=3, sensitivity_fn=None):
    if sensitivity_fn is None:
        from d3ft.calc.screening import sensitivity as sensitivity_fn
    sensitivity = sensitivity_fn
    bindings = dict(sensitivity.__globals__, jax=ForwardJax())
    forward_fn = types.FunctionType(sensitivity.__code__, bindings,
                                    'sensitivity_forward', sensitivity.__defaults__,
                                    sensitivity.__closure__)
    forward = np.asarray(forward_fn(out, xc_base, n_channels))
    reverse = np.asarray(reverse)
    if forward.shape != reverse.shape or not np.isfinite(forward).all():
        raise ValueError('Invalid forward sensitivity')
    if not np.isfinite(reverse).all():
        raise ValueError('Invalid reverse sensitivity')
    error = np.abs(forward - reverse)
    tolerance = 1e-9 + 1e-8 * np.maximum(np.abs(forward), np.abs(reverse))
    result = dict(forward=forward.tolist(), reverse=reverse.tolist(),
                  absolute_errors=error.tolist(), atol=1e-9, rtol=1e-8,
                  passed=bool(np.all(error <= tolerance)),
                  forward_algorithm='sequential scalar jacfwd columns via lax.map',
                  scope='First stationary total-energy parameter derivative, same wavefunction and closure')
    return result
