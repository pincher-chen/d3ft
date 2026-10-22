"""Exactly two GPU-sharded PCG updates versus an independent NumPy recurrence."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as S
from projected_solver import real_inner
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!='gpu' or len(jax.devices())!=2:raise ValueError('Exactly two GPUs required')
mesh=Mesh(np.asarray(jax.devices()),('k',));layout=NamedSharding(mesh,S('k',None,None))
rng=np.random.default_rng(20261015);shape=(8,4096,8)
rhs=rng.normal(size=shape)+1j*rng.normal(size=shape);diag=np.linspace(1,4,np.prod(shape)).reshape(shape);u=np.ones(shape)/np.sqrt(np.prod(shape))
@jax.jit
def action(v):return jnp.asarray(diag)*v+.2*jnp.asarray(u)*jnp.real(jnp.sum(jnp.conj(jnp.asarray(u))*v))
def cpu_action(v):return diag*v+.2*u*np.real(np.sum(np.conj(u)*v))
project=lambda v:v.at[:,:,0].set(0)
def cp(v):v=v.copy();v[:,:,0]=0;return v
b=jax.device_put(rhs,layout);x=jnp.zeros_like(b);r=project(b);z=project(r/jnp.asarray(diag));p=z;rz=float(real_inner(r,z))
cx=np.zeros_like(rhs);cr=cp(rhs);cz=cp(cr/diag);cpdir=cz;crz=np.vdot(cr,cz).real;checks=[]
for i in range(2):
 apv=action(p);curvature=float(real_inner(p,apv));alpha=rz/curvature;x=project(x+alpha*p);r=project(r-alpha*apv)
 cap=cpu_action(cpdir);cc=np.vdot(cpdir,cap).real;ca=crz/cc;cx=cp(cx+ca*cpdir);cr=cp(cr-ca*cap)
 z=project(r/jnp.asarray(diag));new=float(real_inner(r,z));p=project(z+(new/rz)*p);rz=new
 cz=cp(cr/diag);cnew=np.vdot(cr,cz).real;cpdir=cp(cz+(cnew/crz)*cpdir);crz=cnew
 checks.append(dict(iteration=i+1,solution_max_error=float(np.max(abs(np.asarray(x)-cx))),residual_max_error=float(np.max(abs(np.asarray(r)-cr))),curvature_relative_error=float(abs(curvature-cc)/abs(cc))))
passed=all(c['solution_max_error']<1e-10 and c['residual_max_error']<1e-10 and c['curvature_relative_error']<1e-11 for c in checks)
result=dict(time=datetime.datetime.now().isoformat(),passed=passed,devices=2,backend=jax.default_backend(),checks=checks,steps=2,scope='Real two-GPU complex sharded asynchronous operator and PCG recurrence vs independent serial NumPy; synthetic coupled SPD, not Mg physical convergence',jobid=__import__('os').environ.get('SLURM_JOB_ID'),step=__import__('os').environ.get('SLURM_STEP_ID'))
(a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if not passed:raise ValueError('Two-step GPU/CPU recurrence gate failed')
