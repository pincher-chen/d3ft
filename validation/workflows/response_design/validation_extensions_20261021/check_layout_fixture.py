"""Identical64k fixture: independent CPU reference -> actual eight-GPU gate."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.sharding import Mesh
from kmesh_response_original import KMeshResponse as OriginalResponse
from layout_sharded_response import ExplicitLayoutResponse
from projected_solver import real_inner
ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reference',type=Path);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!=a.backend or len(jax.devices())!=8:raise ValueError('Eight devices on explicitly requested backend required')
rng=np.random.default_rng(20261015);nk=64;ng=27;nb=2;shape=(3,3,3);n=27
q=np.zeros((nk,ng,nb),complex)
for k in range(nk):q[k,0,0]=np.exp(.2j*k);q[k,1,1]=np.exp(-.3j*k)
edges=np.zeros((nk,ng,4),complex);edges[:,:,:2]=q;edges[:,2,2]=edges[:,3,2]=edges[:,2,3]=1/np.sqrt(2);edges[:,3,3]=-1/np.sqrt(2)
weights=np.zeros((nk,4));weights[0,1]=-1;weights[7,2]=1;diag=np.tile(np.arange(ng,dtype=float),(nk,1))+np.arange(nk)[:,None]*.01;values=diag[:,:nb]
def make(cls):
 r=cls.__new__(cls)
 if cls is ExplicitLayoutResponse:r.mesh=Mesh(np.asarray(jax.devices()),('k',))
 r.shape=shape;r.n=n;r.idx=jnp.arange(ng);r.nk=nk;r.block=2;r.w=1/nk;r.volume=jnp.asarray(125.);r.q=r.put_k(q);r.edges=r.put_k(edges);r.target_weights=r.put_k(weights);r.diagonal=r.put_k(diag);r.values=r.put_k(values);r.projectors=r.put_k(np.zeros((nk,1,1,ng),complex));r.nl=jnp.ones((1,1));r.veff=jnp.zeros(shape)
 def project(x):return x-jnp.einsum('kga,kab->kgb',r.q,jnp.einsum('kga,kgb->kab',jnp.conj(r.q),x))
 kernel=lambda dr:.3*dr+.1*jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dr)/(1+jnp.arange(n).reshape(shape))))
 if cls is ExplicitLayoutResponse:
  r.project=r._jit_vector(project);r.apply_local=r._jit_local(r.local_action);r.kernel=r._jit_replicated(kernel);r.ham=r._jit_vector(lambda x:r.ham_action(x,r.diagonal,r.projectors,r.veff,r.volume));r.hessian=r._jit_vector(lambda x:r.project(r.ham(x)-x*r.values[:,None,:]+r.local_action(r.q,r.kernel(r.density_tangent(x)))))
  r.precondition=r._jit_vector(lambda x:r.project(x/(.5+r.diagonal[:,:,None]-r.values[:,None,:])));r.setup_inner()
 else:
  r.project=jax.jit(project);r.apply_local=jax.jit(r.local_action);r.kernel=jax.jit(kernel);r.ham=jax.jit(lambda x:r.ham_action(x,r.diagonal,r.projectors,r.veff,r.volume));r.hessian=jax.jit(lambda x:r.project(r.ham(x)-x*r.values[:,None,:]+r.local_action(r.q,r.kernel(r.density_tangent(x)))));r.precondition=jax.jit(lambda x:r.project(x/(.5+r.diagonal[:,:,None]-r.values[:,None,:])))
 return r
x=rng.normal(size=q.shape)+1j*rng.normal(size=q.shape);directions=[rng.normal(size=q.shape)+1j*rng.normal(size=q.shape) for _ in range(12)]
def evaluate(r):
 x1=r.project(r.put_k(x));edge=r.weighted_density(r.edges,r.target_weights,r.volume);b=r.project(r.apply_local(r.q,r.kernel(edge)));lam,rec=r.solve(b,None)
 out={'density':np.asarray(r.density(r.q,r.volume)),'edge_density':np.asarray(edge),'density_tangent':np.asarray(r.density_tangent(x1)),'Hessian':np.asarray(r.hessian(x1)),'rhs':np.asarray(b),'lambda':np.asarray(lam)};rev=[];fw=[]
 for d in directions:
  e=r.project(r.put_k(d));u,_=r.solve(-e,None);inner=r.inner if isinstance(r,ExplicitLayoutResponse) else real_inner;rev.append(float(-4*r.w*inner(lam,e)));fw.append(float(4*r.w*inner(b,u)))
 out['reverse12']=np.asarray(rev);out['forward12']=np.asarray(fw);return out,rec
errors={};compared_inner=[]
if a.backend=='cpu':
 reference,serial_rec=evaluate(make(OriginalResponse));np.savez(a.output/'reference.npz',**reference)
else:
 if a.reference is None:raise ValueError('Independent CPU arrays required')
 reference=dict(np.load(a.reference));serial_rec=None
split=make(ExplicitLayoutResponse)
def probe(aa,bb,value):
 ref=float(np.vdot(aa,bb).real);err=abs(ref-value);tol=1e-9+1e-11*abs(ref);compared_inner.append({'error':err,'tolerance':tol})
 if err>tol:raise ValueError('Real device/NumPy contraction mismatch')
split.inner_probe=probe;actual,rec=evaluate(split)
for key in actual:
 error=float(np.max(abs(actual[key]-reference[key])));errors[key]=error
 if error>2e-10:raise ValueError('CPU/device mismatch '+key+': '+str(error))
if np.max(abs(actual['forward12']-actual['reverse12']))>2e-10:raise ValueError('Forward/reverse fixture mismatch')
if not compared_inner or any(len(x.addressable_shards)!=8 or any(sh.data.shape[0]!=8 for sh in x.addressable_shards) for x in [split.q,split.edges,split.projectors]):raise ValueError('Contraction recording/8local_k ownership missing')
result={'time':datetime.datetime.now().astimezone().isoformat(),'passed':True,'backend':a.backend,'devices':8,'nk':64,'k_per_device':8,'errors':errors,'inner_comparison_count':len(compared_inner),'max_inner_absolute_error':max(q['error'] for q in compared_inner),'PCG':rec,'original_serial_PCG':serial_rec,'scope':'Small synthetic64k true device layouts/common-density Hessian/nonzero PCG vsindependentCPU; not Mgphysical64k certificate'};(a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','backend','nk','errors','inner_comparison_count']}),flush=True)
