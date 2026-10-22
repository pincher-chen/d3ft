import unittest,numpy as np,jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from compact_k_stream import CompactKStream
from d3ft.optim.stiefel import stiefel_coeff
from jrystal._src import pw,energy
from d3ft import pseudopotential as pp
from d3ft.parallel import plain_fft_patch
class CompactTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(42);self.mask=rng.random((6,6,6))>.3;ng=self.mask.sum();self.route=CompactKStream(self.mask)
  q=rng.normal(size=(1,8,ng,4))+1j*rng.normal(size=(1,8,ng,4));self.q=jnp.asarray(np.linalg.qr(q)[0]);self.occ=jnp.full((1,8,4),2/8)
  self.g=jnp.asarray(rng.normal(size=(6,6,6,3)));self.k=jnp.asarray(rng.normal(size=(8,3)))
  self.proj=jnp.asarray(rng.normal(size=(8,3,2,ng))+1j*rng.normal(size=(8,3,2,ng)));self.D=jnp.asarray([[1,.2,.1],[.2,2,0],[.1,0,3]])
 def test_density_charge_and_coupled_gradient(self):
  with plain_fft_patch():
   def dense(q):return pw.density_grid(stiefel_coeff(q,self.mask),100.,self.occ)
   def stream(q):return self.route.density(q,100.,self.occ)
   np.testing.assert_allclose(dense(self.q),stream(self.q),atol=1e-14)
   self.assertAlmostEqual(float(stream(self.q).sum()*100/216),8,places=12)
   # Nonlinear common-density energy tests coupling and reverse scan transpose.
   f=lambda fun:lambda q:jnp.sum(fun(q)**2)
   np.testing.assert_allclose(jax.grad(f(dense))(self.q),jax.jit(jax.grad(f(stream)))(self.q),atol=1e-13,rtol=1e-12)
 def test_kinetic_nonlocal_values_and_derivatives(self):
  def dense(q,v):
   c=stiefel_coeff(q,self.mask)
   return energy.kinetic(self.g,self.k,c,self.occ)+pp.energy_nonlocal(c,self.proj,v,self.occ,self.D,freq_mask=self.mask)
  def stream(q,v):return self.route.kinetic(self.g,self.k,q,self.occ)+self.route.nonlocal_energy(q,self.proj,v,self.occ,self.D,self.mask)
  np.testing.assert_allclose(dense(self.q,100.),stream(self.q,100.),atol=2e-13)
  for a,b in zip(jax.grad(dense,argnums=(0,1))(self.q,100.),jax.grad(stream,argnums=(0,1))(self.q,100.)):np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-12)
 def test_volume_and_geometry_gradients(self):
  with plain_fft_patch():
   def objective(scale,compact):
    v=100*jnp.exp(scale);c=self.q if compact else stiefel_coeff(self.q,self.mask)
    density=self.route.density(c,v,self.occ) if compact else pw.density_grid(c,v,self.occ)
    kinetic=self.route.kinetic(self.g*jnp.exp(-scale/3),self.k*jnp.exp(-scale/3),c,self.occ) if compact else energy.kinetic(self.g*jnp.exp(-scale/3),self.k*jnp.exp(-scale/3),c,self.occ)
    return kinetic+jnp.sum(density**2)*v/216
   np.testing.assert_allclose(jax.grad(lambda s:objective(s,True))(0.),jax.grad(lambda s:objective(s,False))(0.),atol=1e-13)
if __name__=='__main__':unittest.main()
