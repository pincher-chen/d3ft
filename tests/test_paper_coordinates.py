"""Charge, Pauli, gauge, implicit-JVP and corrected Rayleigh regression tests."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64',True)
from d3ft.occupation.fd import fermi_dirac_occupation as fd
from d3ft.optim.metric_direction import rotation_weight
class Coordinates(unittest.TestCase):
 def test_charge_pauli_gauge(self):
  for nk in [1,8,27]:
   z=jax.random.normal(jax.random.PRNGKey(nk),(1,nk,12));f=fd(z,8,.1);g=fd(z+2.,8,.1)
   self.assertLess(abs(float(f.sum())-8),1e-12);self.assertTrue(bool(jnp.all((f>=0)&(f<=2/nk))))
   np.testing.assert_allclose(f,g,atol=3e-15,rtol=1e-13)
 def test_charge_derivative(self):
  z=jnp.linspace(-.2,.3,24).reshape(1,2,12);v=jnp.sin(z*3)
  f,df=jax.jvp(lambda x:fd(x,8.,.1),(z,),(v,));h=1e-5
  num=(fd(z+h*v,8.,.1)-fd(z-h*v,8.,.1))/(2*h)
  self.assertLess(abs(float(df.sum())),1e-12);np.testing.assert_allclose(df,num,atol=2e-9,rtol=2e-8)
 def test_capacity_boundary_rejected(self):
  with self.assertRaises(ValueError):fd(jnp.zeros((1,1,4)),8,.1)
 def test_tiny_positive_rayleigh(self):
  f=jnp.array([[[1.,1e-20]]]);e=jnp.array([.2,.8]);xg=jnp.diag(2*f[0,0]*e)[None,None]
  q=rotation_weight(f,xg);np.testing.assert_allclose(q,q.swapaxes(-1,-2),rtol=0,atol=0)
  cap=2.;expected=float((f[0,0,0]+f[0,0,1])*(2*cap-f[0,0,0]-f[0,0,1])/(2*cap)*jnp.sqrt(.6**2+.001**2))
  self.assertAlmostEqual(float(q[0,0,0,1]),expected,places=14)
 def test_float_dtype(self):
  self.assertEqual(np.asarray([1.,2.],dtype=float).dtype,np.dtype('float64'))
if __name__=='__main__':unittest.main()
