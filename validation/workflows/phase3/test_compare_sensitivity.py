"""CPU check of transform substitution; production DFT evidence is separate."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import compare_sensitivity as audit

jax.config.update('jax_enable_x64', True)


def toy_sensitivity(out, vector, n_channels=3):
    return np.asarray(jax.grad(lambda p: jnp.sum(jnp.sin(p)*out+p*p))(vector))


class ForwardComparisonTests(unittest.TestCase):
    def test_local_substitution_and_mismatch(self):
        original_grad = jax.grad
        vector = jnp.arange(7, dtype=float)/10
        expected = 2*np.cos(vector)+2*vector
        self.assertTrue(audit.compare(2., vector, expected, sensitivity_fn=toy_sensitivity)['passed'])
        self.assertFalse(audit.compare(2., vector, expected+.1, sensitivity_fn=toy_sensitivity)['passed'])
        self.assertIs(jax.grad, original_grad)

    def test_jitted_nonzero_argnum(self):
        function = lambda p, o, v: jnp.sum(p*v*v)+o
        vector = jnp.arange(7, dtype=float)/10
        derivative = jax.jit(audit.forward_gradient(function, argnums=2))
        np.testing.assert_allclose(derivative(3., 2., vector), 6*np.asarray(vector), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
