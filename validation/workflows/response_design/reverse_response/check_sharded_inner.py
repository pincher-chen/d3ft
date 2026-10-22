"""Multi-device CPU regression: complex contractions and projected PCG."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as S
from projected_solver import real_inner,pcg
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!='cpu' or len(jax.devices())<2:raise ValueError('Multiple CPU devices required')
mesh=Mesh(np.asarray(jax.devices()),('k',));sh=NamedSharding(mesh,S('k',None,None));rep=NamedSharding(mesh,S())
rng=np.random.default_rng(20261015);shape=(8,19,5);x=rng.normal(size=shape)+1j*rng.normal(size=shape);y=rng.normal(size=shape)+1j*rng.normal(size=shape)
xs=jax.device_put(x,sh);ys=jax.device_put(y,sh);errors={}
errors['complex_contraction']=abs(float(real_inner(xs,ys))-float(np.vdot(x,y).real))
# Exercise asynchronous jit -> elementwise CG expressions -> contraction,
# including a collective density-like scalar coupling between k blocks.
diag=jnp.asarray(np.linspace(1,3,np.prod(shape)).reshape(shape));u=jnp.asarray(np.ones(shape)/np.sqrt(np.prod(shape)))
def action(v):return diag*v+.2*u*jnp.real(jnp.sum(jnp.conj(u)*v))
serial_action=jax.jit(action)
sharded_action=jax.jit(action,in_shardings=(sh,),out_shardings=sh)
mixed=sharded_action(xs)+.37*ys
errors['async_mixed_contraction']=abs(float(real_inner(xs,mixed))-float(np.real(np.sum(np.conj(x)*np.asarray(mixed)))))
errors['traced_contraction']=abs(float(jax.jit(lambda v:real_inner(v,v),in_shardings=(sh,),out_shardings=rep)(xs))-float(np.vdot(x,x).real))
project=lambda v:v.at[:,:,0].set(0)
precondition=lambda v:project(v/diag)
serial,sc=pcg(serial_action,jnp.asarray(y),project,precondition)
parallel,pc=pcg(sharded_action,ys,jax.jit(project,in_shardings=(sh,),out_shardings=sh),precondition)
errors['PCG_solution']=float(np.max(abs(np.asarray(serial)-np.asarray(parallel))))
errors['true_residual']=float(abs(pc['residual']-sc['residual']))
passed=all(v<1e-10 for v in errors.values()) and sc['converged'] and pc['converged']
(a.output/'acceptance.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),passed=passed,devices=len(jax.devices()),errors=errors,serial_record=sc,sharded_record=pc,scope='CPU arithmetic/async mixed expressions/projected PCG equivalence; GPU Mg response still requires a new native run'),indent=2)+'\n')
print(json.dumps(dict(passed=passed,errors=errors)),flush=True)
if not passed:raise ValueError('Sharded contraction regression failed')
