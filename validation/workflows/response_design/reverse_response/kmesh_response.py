"""Coupled uniform k-mesh tangent/adjoint, one common density and HXC kernel.

Initial implementation is streamed on ONE GPU; does not claim distributed k2
production scalability. Fixed integer occupation is 2/Nk, never multiplied by
wk again. The projected Hessian is self-adjoint in unweighted block real inner
product because all wk=1/Nk; forward correction/pullback include 4/Nk.
"""
import os
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from geometry_layout import GeometryLayout
from mesh_target import select_mesh_target
from projected_solver import pcg,real_inner
class KMeshResponse:
 def __init__(self,out,orbitals,values,theta0,block_size=4,*,include_coordinates=False,displacement_unit='bohr',objective='hard',tau_Ha=None):
  from jrystal._src.grid import g_vectors
  from d3ft import pseudopotential as pp
  from d3ft.calc.xc_tunable import xc_energy_tunable
  self.out=out;self.shape=tuple(out.freq_mask.shape);self.n=int(np.prod(self.shape))
  self.idx=jnp.asarray(np.where(np.asarray(out.freq_mask).reshape(-1))[0]);self.nk=len(out.k_frac)
  if self.nk<1 or out.config.spin!=0 or not out.config.spin_restricted or out.core_density_grid is not None:
   raise ValueError('Restricted non-NLCC uniform insulating mesh required')
  if out.temperature!=0 or out.config.d3ft.occupation.enabled:raise ValueError('Fixed zero-temperature occupation required')
  ne=float(np.sum(out.pseudopot.valence_charges));self.n_occupied=int(round(ne/2))
  if abs(2*self.n_occupied-ne)>1e-10:raise ValueError('Closed-shell integer electron count required')
  occ=np.asarray(out.get_occupation(out.params_occ,out.final_eps))
  if occ.shape[-1]!=self.n_occupied or np.max(abs(occ-2/self.nk))>1e-12:raise ValueError('Occupation must contain 2/Nk exactly once')
  if os.environ.get('HNL_CONJ_FIX','1')!='1':raise ValueError('Validated KB conjugation required')
  orbitals=np.asarray(orbitals);values=np.asarray(values)
  if orbitals.ndim!=3 or orbitals.shape[:2]!=(self.nk,len(self.idx)) or values.shape!=(self.nk,orbitals.shape[2]):raise ValueError('Mismatched k spectrum/basis')
  weights,self.target_info=select_mesh_target(values,self.n_occupied,objective,tau_Ha)
  self.target_weights=self.put_k(weights);self.q=self.put_k(orbitals[:,:,:self.n_occupied]);self.edges=self.put_k(orbitals)
  self.values=self.put_k(values[:,:self.n_occupied]);self.w=1/self.nk;self.block=int(np.gcd(block_size,self.n_occupied))
  if block_size<1:raise ValueError('Positive block size required')
  self.cell=jnp.asarray(out.crystal.cell_vectors);self.positions=jnp.asarray(out.crystal.positions);self.k_frac=self.put_k(np.asarray(out.k_frac))
  self.layout=GeometryLayout(len(out.crystal.positions),theta0,include_coordinates=include_coordinates,displacement_unit=displacement_unit)
  self.n_variables=self.layout.nvariables;self.zero=jnp.zeros(self.n_variables)
  self.volume=jnp.abs(jnp.linalg.det(self.cell));self.g=g_vectors(self.cell,np.asarray(self.shape));self.nl=jnp.asarray(out.nl_d_matrix)
  xc_params=out.config.d3ft.get('xc_params',None);factor=out.config.d3ft.get('xc_grid_factor',1)
  def hxc(rho,g,volume):
   rec=jnp.fft.fftn(rho);g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
   vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*rec/g2).at[0,0,0].set(0.)))
   def ex(r):return xc_energy_tunable(r[None,...],g,volume,out.config.xc,xc_params=xc_params,grid_factor=factor)
   return vh+jax.grad(ex)(rho)*self.n/volume
  self.hxc=hxc
  self.rho=jax.jit(lambda:self.density(self.q,self.volume))();jax.block_until_ready(self.rho)
  def base(delta):
   cell,positions=self.layout.deform(self.cell,self.positions,delta)
   volume=jnp.abs(jnp.linalg.det(cell));g=g_vectors(cell,np.asarray(self.shape));reciprocal=2*jnp.pi*jnp.linalg.inv(cell).T
   kcart=self.k_frac@reciprocal
   local=pp.potential_local_reciprocal(positions,g,out.pseudopot.r_grid,out.pseudopot.local_potential_grid,out.pseudopot.local_potential_charge,volume,kmax=out.grid_kmax)
   gk=pp.gk_vectors(g,kcart,out.freq_mask)
   beta=pp.beta_grid(out.pseudopot.r_grid,out.pseudopot.nonlocal_beta_grid,out.pseudopot.nonlocal_angular_momentum,gk,out.beta_kmax,tables=out.beta_tables)
   proj=pp.projectors_reciprocal(positions,gk,out.pseudopot.nonlocal_angular_momentum,beta)
   return jnp.sum(gk*gk,axis=-1)/2,proj,jnp.real(jnp.fft.ifftn(local)),volume,g
  self.geometry_base=base
  def geometry(delta):
   diagonal,proj,local,volume,g=base(delta);rho=self.rho*self.volume/volume
   return diagonal,proj,local+hxc(rho,g,volume),volume
  self.geometry=geometry
  def project(x):
   overlap=jnp.einsum('kga,kgb->kab',jnp.conj(self.q),x)
   return x-jnp.einsum('kga,kab->kgb',self.q,overlap)
  self.project=jax.jit(project);self.apply_local=jax.jit(self.local_action)
  print('KMESH_STAGE geometry_potential',flush=True)
  self.diagonal,self.projectors,self.veff,self.volume0=self.compile_geometry(geometry)(self.zero);jax.block_until_ready(self.veff)
  self.ham=jax.jit(lambda x:self.ham_action(x,self.diagonal,self.projectors,self.veff,self.volume0))
  self.kernel=jax.jit(lambda dr:jax.jvp(lambda r:hxc(r,self.g,self.volume),(self.rho,),(dr,))[1])
  self.delta_rho=jax.jit(self.density_tangent)
  self.hessian=jax.jit(lambda x:self.project(self.ham(x)-x*self.values[:,None,:]+self.local_action(self.q,self.kernel(self.density_tangent(x)))))
  self.precondition=jax.jit(lambda x:self.project(self.project(x)/(.5+self.diagonal[:,:,None]-self.values[:,None,:])))
  self.edge_density=jax.jit(lambda:self.weighted_density(self.edges,self.target_weights,self.volume))()
  self.explicit_gap=jax.jit(lambda d:jnp.sum(self.target_weights*jnp.real(jnp.sum(jnp.conj(self.edges)*self.ham_geometry(d,self.edges),axis=1))))
  self.external=jax.jit(lambda d:jax.jvp(lambda z:self.ham_geometry(z,self.q),(self.zero,),(d,))[1])
  self.edge_g=jnp.sum(jnp.abs(self.edges)**2*self.target_weights[:,None,:],axis=-1)
  self.edge_r=self.edge_density*self.volume
  def scalar_geometry(delta,weight_g,weight_r,lam):
   diagonal,proj,local,volume,g=base(delta);kinetic=jnp.sum(weight_g*diagonal)
   rho=self.rho*self.volume/volume;g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
   vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*jnp.fft.fftn(rho)/g2).at[0,0,0].set(0.)))
   locality=jnp.sum(weight_r*(local+vh))/self.n
   edge=jnp.einsum('kbmg,kgi->kbmi',jnp.conj(proj),self.edges)
   occ=jnp.einsum('kbmg,kgi->kbmi',jnp.conj(proj),self.q)
   adj=jnp.einsum('kbmg,kgi->kbmi',jnp.conj(proj),lam)
   target_nl=jnp.real(jnp.einsum('kami,ab,kbmi,ki->',jnp.conj(edge),self.nl,edge,self.target_weights))
   adjoint_nl=jnp.real(jnp.einsum('kami,ab,kbmi->',jnp.conj(adj),self.nl,occ))
   def ex(r):return xc_energy_tunable(r[None,...],g,volume,out.config.xc,xc_params=xc_params,grid_factor=factor)
   xc_direction=jax.jvp(ex,(rho,),(weight_r/volume,))[1]
   return kinetic+locality+(target_nl-4*self.w*adjoint_nl)/volume+xc_direction
  self.scalar_geometry=jax.jit(scalar_geometry);self.scalar_gradient=jax.jit(jax.grad(scalar_geometry,argnums=0))
  self.bare=lambda d:self.scalar_gradient(d,self.edge_g,self.edge_r,jnp.zeros_like(self.q))
 def put_k(self,x):return jnp.asarray(x)
 def compile_geometry(self,geometry):return jax.jit(geometry)
 def wave(self,x):
  full=jnp.zeros((self.n,x.shape[1]),dtype=x.dtype).at[self.idx,:].set(x)
  return jnp.fft.ifftn(full.reshape(self.shape+(x.shape[1],)),axes=(0,1,2))
 def chunks(self,x):
  pad=(-x.shape[-1])%self.block;x=jnp.pad(x,((0,0),(0,0),(0,pad)))
  return jnp.transpose(x.reshape(self.nk,x.shape[1],-1,self.block),(0,2,1,3)).reshape(-1,x.shape[1],self.block)
 def weighted_density(self,x,weights,volume):
  pad=(-x.shape[-1])%self.block;weights=jnp.pad(weights,((0,0),(0,pad))).reshape(-1,self.block)
  def one(r,pair):
   orb,w=pair;return r+jnp.sum(jnp.abs(self.wave(orb))**2*w[None,None,None,:],axis=-1),None
  total,_=jax.lax.scan(one,jnp.zeros(self.shape),(self.chunks(x),weights))
  return total*self.n**2/volume
 def density(self,x,volume):return self.weighted_density(x,jnp.full((self.nk,x.shape[-1]),2*self.w),volume)
 def density_tangent(self,x):
  def one(r,pair):
   q,dx=pair;return r+jnp.sum(jnp.real(jnp.conj(self.wave(q))*self.wave(dx)),axis=-1),None
  total,_=jax.lax.scan(one,jnp.zeros(self.shape),(self.chunks(self.q),self.chunks(x)))
  return total*4*self.w*self.n**2/self.volume
 def local_action(self,x,potential):
  def one(b):
   data=jnp.fft.fftn(potential[...,None]*self.wave(b),axes=(0,1,2))
   return data.reshape(self.n,self.block)[self.idx,:]
  result=jax.lax.map(one,self.chunks(x)).reshape(self.nk,-1,x.shape[1],self.block)
  return jnp.transpose(result,(0,2,1,3)).reshape(self.nk,x.shape[1],-1)[:,:,:x.shape[-1]]
 def ham_action(self,x,diagonal,projectors,potential,volume):
  inner=jnp.einsum('kbmg,kgi->kbmi',jnp.conj(projectors),x)
  coupled=jnp.einsum('ab,kbmi->kami',self.nl,inner)
  nonlocal_term=jnp.einsum('kbmg,kbmi->kgi',projectors,coupled)/volume
  return diagonal[:,:,None]*x+nonlocal_term+self.local_action(x,potential)
 def ham_geometry(self,d,x):return self.ham_action(x,*self.geometry(d))
 def reverse_gradient(self,lam):
  weight_g=self.edge_g-4*self.w*jnp.real(jnp.sum(jnp.conj(lam)*self.q,axis=-1))
  weight_r=self.edge_r-self.delta_rho(lam)*self.volume
  return self.scalar_gradient(self.zero,weight_g,weight_r,lam)
 def solve(self,b,callback):return pcg(self.hessian,b,self.project,self.precondition,callback=callback)
