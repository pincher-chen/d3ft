"""K-sharded coupled adjoint, with explicit local FFTs and density psums.

Each device owns k blocks and complete reciprocal/real grids. Only densities
and scalar contractions are collective. HXC is evaluated on the same summed
density on every device; independent per-k response is never substituted.
Uniform Nk divisibility is mandatory. GPU acceptance is separate from CPU QA.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as S
from jax.experimental.shard_map import shard_map
from kmesh_response import KMeshResponse


class ShardedKMeshResponse(KMeshResponse):
 def __init__(self,*args,devices=None,**kwargs):
  devices=list(jax.devices() if devices is None else devices)
  if len(devices)<2:raise ValueError('At least two devices required; use KMeshResponse for one')
  self.mesh=Mesh(np.asarray(devices),('k',))
  nk=len(args[0].k_frac) if args else len(kwargs['out'].k_frac)
  if nk%len(devices):raise ValueError('Uniform k blocks must divide device count')
  super().__init__(*args,**kwargs)
  # The large persistent arrays must never be replicated on GPU0.
  for name in ['q','edges','values','target_weights','k_frac','diagonal','projectors']:
   x=getattr(self,name)
   if x.sharding.spec[0]!='k':raise RuntimeError('Unsharded k array: '+name)

 def put_k(self,x):
  x=np.asarray(x)
  return jax.device_put(x,NamedSharding(self.mesh,S('k',*([None]*(x.ndim-1)))))

 def compile_geometry(self,geometry):
  replicated=NamedSharding(self.mesh,S())
  return jax.jit(geometry,out_shardings=(NamedSharding(self.mesh,S('k',None)),
    NamedSharding(self.mesh,S('k',None,None,None)),replicated,replicated))

 def _local_chunks(self,x):
  nk=x.shape[0];pad=(-x.shape[-1])%self.block
  x=jnp.pad(x,((0,0),(0,0),(0,pad)))
  return jnp.transpose(x.reshape(nk,x.shape[1],-1,self.block),(0,2,1,3)).reshape(-1,x.shape[1],self.block)

 def weighted_density(self,x,weights,volume):
  def local(orbitals,w,v):
   pad=(-orbitals.shape[-1])%self.block
   w=jnp.pad(w,((0,0),(0,pad))).reshape(-1,self.block)
   def one(r,pair):
    orb,ww=pair
    return r+jnp.sum(abs(self.wave(orb))**2*ww[None,None,None,:],axis=-1),None
   rho,_=jax.lax.scan(one,jnp.zeros(self.shape),(self._local_chunks(orbitals),w))
   return jax.lax.psum(rho,'k')*self.n**2/v
  return shard_map(local,mesh=self.mesh,in_specs=(S('k',None,None),S('k',None),S()),out_specs=S(),check_rep=False)(x,weights,volume)

 def density_tangent(self,x):
  def local(q,dx):
   def one(r,pair):
    qq,dd=pair
    return r+jnp.sum(jnp.real(jnp.conj(self.wave(qq))*self.wave(dd)),axis=-1),None
   rho,_=jax.lax.scan(one,jnp.zeros(self.shape),(self._local_chunks(q),self._local_chunks(dx)))
   return jax.lax.psum(rho,'k')*4*self.w*self.n**2/self.volume
  return shard_map(local,mesh=self.mesh,in_specs=(S('k',None,None),S('k',None,None)),out_specs=S(),check_rep=False)(self.q,x)

 def local_action(self,x,potential):
  def local(orbitals,v):
   def one(b):
    f=jnp.fft.fftn(v[...,None]*self.wave(b),axes=(0,1,2))
    return f.reshape(self.n,self.block)[self.idx,:]
   nk=orbitals.shape[0]
   result=jax.lax.map(one,self._local_chunks(orbitals)).reshape(nk,-1,orbitals.shape[1],self.block)
   return jnp.transpose(result,(0,2,1,3)).reshape(nk,orbitals.shape[1],-1)[:,:,:orbitals.shape[-1]]
  return shard_map(local,mesh=self.mesh,in_specs=(S('k',None,None),S()),out_specs=S('k',None,None),check_rep=False)(x,potential)
