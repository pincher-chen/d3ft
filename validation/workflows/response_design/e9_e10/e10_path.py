"""Single-device E10 path block; frozen k4 density and checkpointed 56-band spectra."""
import argparse,sys,pickle,gc,os,time,warnings
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from common import *
from gamma_response import validated_radial_splines
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
from radial_quadrature import local_simpson_patch
from variational_ks import operator_factory
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch

def main(a):
 a.output.mkdir(exist_ok=False);m=read(a.source/'metadata.json');pt=read(a.source/'point.json');protocol=read(a.protocol);path=read(protocol['E10']['path_source']);frac=np.array(path['k_fractional']);indices=__import__('json').loads(a.indices)
 assert pt['physical_passed'] and sha(a.source/'state.pkl')==m['state_sha256'] and len(jax.devices())==1
 with (a.source/'state.pkl').open('rb') as f:state=pickle.load(f)
 full=np.asarray(state['pw']);seed={k:np.array(v)[:,:1].copy() for k,v in state.items()};seed['occ']=seed['occ']*64
 assert all(v.flags.writeable for v in seed.values())
 charge=float(np.sum(seed['occ']));assert abs(charge-96.)<=1e-6, ('Invalid Gamma seed charge',charge)
 dump(a.output/'seed_preflight.json',dict(charge=charge,all_writable=True,source_state_sha256=m['state_sha256'],copy_fix=True))
 print('E10_SEED_PREFLIGHT',charge,'writable copies',flush=True)
 c=ConfigDict(m['config']);c.k_grid_sizes=1;c.d3ft.parallel.k_pool=False;c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(a.source/'input.xyz');c.d3ft.checkpoint=dict(root=str(a.output/'checkpoints'),interval=500);c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6)
 with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,a.source) as info,validated_radial_splines():
  out=calc(c,init_params=seed,freq_mask=np.load(a.source/'mask.npy'));out.grid_kmax=info['grid_kmax'];out.k_frac=frac;out.density_grid=jnp.asarray(np.load(a.source/'density.npy'));factory=operator_factory(out)
  dump(a.output/'metadata.json',dict(source=str(a.source),source_state_sha256=m['state_sha256'],density_sha256=sha(a.source/'density.npy'),protocol_sha256=sha(a.protocol),path_sha256=sha(protocol['E10']['path_source']),bands=56,indices=indices,jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID')))
  for index in indices:
   old=a.resume_source/f'k_{index:04d}.json' if a.resume_source else None
   if old and old.exists():
    q=read(old)
    if q['source_state_sha256']!=m['state_sha256'] or not q['passed'] or max(q['residuals_Ha'])>1e-7 or q['orthogonality_max']>1e-8:raise ValueError('Invalid reused point')
    dump(a.output/f'k_{index:04d}.json',q);continue
   t=time.monotonic();op,ng=factory(index);nb=60
   @jax.jit
   def stream(x):
    padding=(-x.shape[1])%4;y=jnp.pad(x,((0,0),(0,padding))).reshape(ng,-1,4).transpose(1,0,2);z=jax.lax.map(jax.vmap(op,in_axes=1,out_axes=1),y);return z.transpose(1,0,2).reshape(ng,-1)[:,:x.shape[1]]
   mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(stream(jnp.asarray(x.reshape(-1,1))))[:,0],matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128);diagonal=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((ng,ng),matvec=lambda x:diagonal*x.reshape(-1),matmat=lambda x:diagonal[:,None]*x,dtype=np.complex128)
   cp=a.output/f'k_{index:04d}_subspace.npy';last=0;resume=a.resume_source/f'k_{index:04d}_subspace.npy' if a.resume_source else None
   if resume and resume.exists():
    initial=np.load(resume);last=read(a.resume_source/f'k_{index:04d}_solver.json')['iterations_completed']
   else:
    nearest=int(np.argmin(np.linalg.norm(np.asarray(pt['k_fractional'])-frac[index],axis=1)));rng=np.random.default_rng(20261019+index);initial=np.linalg.qr(np.concatenate([full[0,nearest],rng.normal(size=(ng,12))+1j*rng.normal(size=(ng,12))],axis=1))[0]
   passed=False
   for k in range(last//100,36):
    with warnings.catch_warnings(record=True):vals,q=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=100)
    q=np.linalg.qr(q)[0];hq=mat.matmat(q);h=q.conj().T@hq;vals,u=np.linalg.eigh((h+h.conj().T)/2);q=q@u;hq=hq@u;res=np.linalg.norm(hq-q*vals,axis=0);orth=np.max(abs(q.conj().T@q-np.eye(nb)));passed=bool(np.max(res[:56])<=1e-7 and orth<=1e-8)
    tmp=cp.with_suffix('.npy.tmp')
    with tmp.open('wb') as f:np.save(f,q)
    tmp.replace(cp);dump(a.output/f'k_{index:04d}_solver.json',dict(index=index,iterations_completed=(k+1)*100,max_iterations=3600,source_state_sha256=m['state_sha256'],max_residual_Ha=float(max(res[:56])),passed=passed,subspace_sha256=sha(cp)))
    print('E10_PATH_CHECKPOINT',index,(k+1)*100,float(max(res[:56])),flush=True)
    if passed:break
    initial=q
   row=dict(index=index,k_frac=frac[index].tolist(),eigenvalues_Ha=vals[:56].tolist(),residuals_Ha=res[:56].tolist(),orthogonality_max=float(orth),source_state_sha256=m['state_sha256'],source=str(a.source),seconds=time.monotonic()-t,passed=passed)
   dump(a.output/f'k_{index:04d}.json',row)
   if not passed:raise ValueError('Unchanged 56-band E10 Ritz condition not met')
   del op,mat,pre,q,hq,initial;jax.clear_caches();gc.collect()
  dump(a.output/'summary.json',dict(passed=True,indices=indices,bands=56,source=str(a.source)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--indices',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume-source',type=Path);main(ap.parse_args())
