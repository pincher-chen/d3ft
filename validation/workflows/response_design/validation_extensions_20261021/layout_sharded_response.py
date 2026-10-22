"""Explicit layouts only: unchanged coupled KMeshResponse expressions and PCG."""
import contextlib
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as S
from jax.experimental.shard_map import shard_map
from sharded_kmesh_response import ShardedKMeshResponse
import projected_solver as solver

class ExplicitLayoutResponse(ShardedKMeshResponse):
 @property
 def rep(self):return NamedSharding(self.mesh,S())
 @property
 def vec(self):return NamedSharding(self.mesh,S('k',None,None))
 @property
 def grid_k(self):return NamedSharding(self.mesh,S('k',None))
 def _jit_constant(self,f):return jax.jit(f,out_shardings=self.rep)
 def _jit_vector(self,f):return jax.jit(f,in_shardings=(self.vec,),out_shardings=self.vec)
 def _jit_local(self,f):return jax.jit(f,in_shardings=(self.vec,self.rep),out_shardings=self.vec)
 def _jit_replicated(self,f):return jax.jit(f,in_shardings=(self.rep,),out_shardings=self.rep)
 def _jit_density_tangent(self,f):return jax.jit(f,in_shardings=(self.vec,),out_shardings=self.rep)
 def _jit_external(self,f):return jax.jit(f,in_shardings=(self.rep,),out_shardings=self.vec)
 def _jit_geometry_scalar(self,f):return jax.jit(f,in_shardings=(self.rep,self.grid_k,self.rep,self.vec),out_shardings=self.rep)
 def compile_geometry(self,f):return jax.jit(f,in_shardings=(self.rep,),out_shardings=(self.grid_k,NamedSharding(self.mesh,S('k',None,None,None)),self.rep,self.rep))
 def setup_inner(self):
  def local(a,b):return jax.lax.psum(jnp.real(jnp.sum(jnp.conj(a)*b)),'k')
  self._inner=jax.jit(shard_map(local,mesh=self.mesh,in_specs=(S('k',None,None),S('k',None,None)),out_specs=S(),check_rep=False),in_shardings=(self.vec,self.vec),out_shardings=self.rep)
  # Lazy compilation must preserve a probe installed BEFORE the first solve.
  if not hasattr(self,'inner_probe'):self.inner_probe=None
 def inner(self,a,b):
  if not hasattr(self,'_inner'):self.setup_inner()
  value=self._inner(a,b)
  if self.inner_probe is not None:self.inner_probe(np.asarray(jax.device_get(a)),np.asarray(jax.device_get(b)),float(value))
  return value
 def solve(self,b,callback):
  # The original PCG and all tolerances are unchanged. Only its contraction
  # dispatch uses explicit per-k layouts and one global real sum.
  previous=solver.real_inner;solver.real_inner=self.inner
  try:return super().solve(b,callback)
  finally:solver.real_inner=previous
