"""Single-device compact PW route: expand only one four-band k chunk.

Scoped opt-in execution-order change for restricted, fixed-occupation Stiefel
states. The ordinary DFT functional, optimizer, charge, force/stress and residual
programs remain in use. Unlike FFT-only streaming, no [Nk,band,FFT] array exists.
"""
from contextlib import contextmanager
import importlib
import numpy as np
import jax
import jax.numpy as jnp

class CompactKStream:
 def __init__(self,mask,block=4):
  self.mask=np.asarray(mask);self.shape=tuple(self.mask.shape);self.n=int(np.prod(self.shape))
  self.idx=jnp.asarray(np.flatnonzero(self.mask.reshape(-1)));self.block=block
 def coefficients(self,p,mask):
  if p.ndim!=4 or p.shape[0]!=1 or p.shape[2]!=len(self.idx):raise ValueError('Restricted compact Stiefel coefficients required')
  if not np.array_equal(np.asarray(mask),self.mask):raise ValueError('Compact route Miller mismatch')
  return p
 def density(self,p,volume,occupation=None):
  if occupation is None:raise ValueError('Compact streaming requires explicit weighted occupations')
  if p.ndim!=4 or p.shape[0]!=1 or p.shape[2]!=len(self.idx):raise ValueError('Not compact restricted orbitals')
  nk,ng,nb=p.shape[1:];block=int(np.gcd(nb,self.block))
  q=p[0].reshape(nk,ng,nb//block,block).transpose(0,2,1,3).reshape(-1,ng,block)
  weights=occupation[0].reshape(-1,block)
  def one(rho,pair):
   c,w=pair
   full=jnp.zeros((self.n,block),dtype=c.dtype).at[self.idx,:].set(c)
   psi=jnp.fft.ifftn(full.reshape(self.shape+(block,)),axes=(0,1,2))*self.n/jnp.sqrt(volume)
   return rho+jnp.sum(jnp.real(psi*jnp.conj(psi))*w[None,None,None,:],axis=-1),None
  rho,_=jax.lax.scan(jax.checkpoint(one),jnp.zeros(self.shape,dtype=p.real.dtype),(q,weights))
  return rho[None,...]
 def kinetic(self,g,k,p,occupation=None):
  if occupation is None:raise ValueError('Weighted occupation required')
  selected=g.reshape(-1,3)[self.idx]
  gk2=jnp.sum((selected[None,:,:]+k[:,None,:])**2,axis=-1)
  return .5*jnp.sum(jnp.real(p*jnp.conj(p))*gk2[None,:,:,None]*occupation[:,:,None,:])
 def nonlocal_energy(self,p,projectors,vol,occupation,d_matrix,freq_mask=None):
  if projectors.ndim!=4 or projectors.shape[-1]!=p.shape[2]:raise ValueError('Compact KB projectors on same Miller set required')
  overlaps=jnp.einsum('skgi,kbmg->skbmi',p,jnp.conj(projectors))
  return jnp.real(jnp.einsum('skami,ab,skbmi,ski->',jnp.conj(overlaps),d_matrix,overlaps,occupation))/vol

@contextmanager
def compact_k_stream_patch(mask,block=4):
 from jrystal._src import pw,energy
 from d3ft import pseudopotential as pp
 ground=importlib.import_module('d3ft.calc.ground_state')
 route=CompactKStream(mask,block)
 saved=(ground.stiefel_coeff,pw.density_grid,energy.kinetic,pp.energy_nonlocal)
 ground.stiefel_coeff=route.coefficients;pw.density_grid=route.density
 energy.kinetic=route.kinetic;pp.energy_nonlocal=route.nonlocal_energy
 try:yield route
 finally:ground.stiefel_coeff,pw.density_grid,energy.kinetic,pp.energy_nonlocal=saved
