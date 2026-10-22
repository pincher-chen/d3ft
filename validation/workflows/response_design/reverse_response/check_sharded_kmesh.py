"""CPU eight-device QA of the actual sharded FFT/Hessian/PCG implementation.

Small synthetic insulating blocks isolate the collective implementation, not
MgSiN2 physics. All comparisons use the unchanged serial KMeshResponse methods.
"""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.sharding import Mesh
from kmesh_response import KMeshResponse
from sharded_kmesh_response import ShardedKMeshResponse
from projected_solver import real_inner

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
if jax.default_backend()!='cpu' or len(jax.devices())!=8:raise ValueError('Force CPU with eight virtual devices')
a.output.mkdir(exist_ok=False)
rng=np.random.default_rng(20261015);nk=8;ng=27;nb=2;shape=(3,3,3);n=int(np.prod(shape))
q=np.zeros((nk,ng,nb),complex)
for k in range(nk):
 q[k,0,0]=np.exp(.2j*k);q[k,1,1]=np.exp(-.3j*k)
edges=np.zeros((nk,ng,4),complex);edges[:,:,:2]=q
edges[:,2,2]=1/np.sqrt(2);edges[:,3,2]=1/np.sqrt(2)
edges[:,2,3]=1/np.sqrt(2);edges[:,3,3]=-1/np.sqrt(2)
weights=np.zeros((nk,4));weights[0,1]=-1;weights[7,2]=1
diag=np.tile(np.arange(ng,dtype=float),(nk,1));diag+=np.arange(nk)[:,None]*.01
values=diag[:,:nb]

def make(cls):
 r=cls.__new__(cls)
 if cls is ShardedKMeshResponse:r.mesh=Mesh(np.array(jax.devices()),('k',))
 r.shape=shape;r.n=n;r.idx=jnp.arange(ng);r.nk=nk;r.block=2;r.w=1/nk;r.volume=jnp.asarray(125.)
 r.q=r.put_k(q);r.edges=r.put_k(edges);r.target_weights=r.put_k(weights);r.diagonal=r.put_k(diag);r.values=r.put_k(values)
 r.projectors=r.put_k(np.zeros((nk,1,1,ng),complex));r.nl=jnp.ones((1,1));r.veff=jnp.zeros(shape)
 def project(x):return x-jnp.einsum('kga,kab->kgb',r.q,jnp.einsum('kga,kgb->kab',jnp.conj(r.q),x))
 r.project=jax.jit(project);r.apply_local=jax.jit(r.local_action)
 r.kernel=jax.jit(lambda dr:.3*dr+.1*jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dr)/(1+jnp.arange(n).reshape(shape)))))
 r.ham=jax.jit(lambda x:r.ham_action(x,r.diagonal,r.projectors,r.veff,r.volume))
 r.hessian=jax.jit(lambda x:r.project(r.ham(x)-x*r.values[:,None,:]+r.local_action(r.q,r.kernel(r.density_tangent(x)))))
 r.precondition=jax.jit(lambda x:r.project(x/(.5+r.diagonal[:,:,None]-r.values[:,None,:])))
 return r

serial=make(KMeshResponse);split=make(ShardedKMeshResponse)
x=serial.project(jnp.asarray(rng.normal(size=q.shape)+1j*rng.normal(size=q.shape)));sx=split.put_k(np.asarray(x))
errs={}
def compare(name,left,right):
 error=float(np.max(abs(np.asarray(left)-np.asarray(right))));errs[name]=error
 if error>2e-10:raise ValueError(name+' difference '+str(error))
compare('density',serial.density(serial.q,serial.volume),split.density(split.q,split.volume))
compare('weighted_edge_density',serial.weighted_density(serial.edges,serial.target_weights,serial.volume),split.weighted_density(split.edges,split.target_weights,split.volume))
compare('common_density_tangent',serial.density_tangent(x),split.density_tangent(sx))
compare('local_action',serial.local_action(x,serial.kernel(serial.density_tangent(x))),split.local_action(sx,split.kernel(split.density_tangent(sx))))
compare('coupled_Hessian',serial.hessian(x),split.hessian(sx))
edge=serial.weighted_density(serial.edges,serial.target_weights,serial.volume)
b=serial.project(serial.apply_local(serial.q,serial.kernel(edge)))
sb=split.project(split.apply_local(split.q,split.kernel(split.weighted_density(split.edges,split.target_weights,split.volume))))
compare('adjoint_rhs',b,sb)
if float(jnp.linalg.norm(b))<1e-9:raise ValueError('Trivial zero RHS does not test adjoint collectives')
lam,rec=serial.solve(b,None);slam,srec=split.solve(sb,None);compare('adjoint_solution',lam,slam)
if min(rec['iterations'],srec['iterations'])<1:raise ValueError('Nontrivial PCG coverage required')
# Twelve geometry directions in complex orbital space; adjoint and tangent
# corrections use the original 4/Nk convention, never an extra block weight.
full=[];sharded=[];tangents=[]
for axis in range(12):
 e=serial.project(jnp.asarray(rng.normal(size=q.shape)+1j*rng.normal(size=q.shape)))
 full.append(float(-4*serial.w*real_inner(lam,e)))
 sharded.append(float(-4*split.w*real_inner(slam,split.put_k(np.asarray(e)))))
 u,ur=split.solve(-split.put_k(np.asarray(e)),None)
 tangents.append(float(4*split.w*real_inner(sb,u)))
compare('full_12_gradient',full,sharded);compare('forward_reverse_12',sharded,tangents)
probe=jnp.zeros_like(x).at[0].set(x[0]);cross=float(jnp.linalg.norm(serial.hessian(probe)[1:]))
if cross<1e-10:raise ValueError('Missing cross-k common-density coupling')
charge=float(split.density(split.q,split.volume).mean()*split.volume)
if abs(charge-2*nb)>1e-12:raise ValueError('Double Nk weight')
shard_counts=[len(v.addressable_shards) for v in [split.q,split.edges]]
if shard_counts!=[8,8] or any(s.data.shape[0]!=1 for s in split.q.addressable_shards):raise ValueError('Bad k ownership')
result=dict(time=datetime.datetime.now().isoformat(),passed=True,backend='cpu',devices=8,nk=8,
            errors=errs,cross_k_action_norm=cross,electrons=charge,serial_adjoint=rec,sharded_adjoint=srec,
            scope='Actual distributed local FFT/psum/Hessian/PCG QA on synthetic insulating blocks; real Si/Mg GPU physics and memory not yet certified')
(a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print('E5 CPU八设备分片：PASS。\n公共密度/耦合Hessian/伴随及12方向与单设备一致。\n实际材料GPU验收仍待资源和前置门。',flush=True)
