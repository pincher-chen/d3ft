"""One-device batch of local spectra of a common accepted k4 density.

All 48 occupied and two edge/buffer eigenpairs must pass the same Ritz gate.
Batching shares reconstruction; each k still has its own local KB phase/T(G+k).
"""
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from gamma_response import dump,sha,validated_radial_splines
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from variational_ks import operator_factory
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--indices',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ritz-restarts',type=int,default=0);a=ap.parse_args()
if len(jax.devices())!=1:raise ValueError('One device per spectrum batch')
indices=json.loads(a.indices);a.output.mkdir(exist_ok=False)
m=json.loads((a.source/'metadata.json').read_text());pt=json.loads((a.source/'point.json').read_text());source_hash=sha(a.source/'state.pkl')
if source_hash!=m['state_sha256'] or not pt['physical_passed']:raise ValueError('Unaccepted k4 SCF')
if len(pt['k_fractional'])!=64 or any(k not in range(64) for k in indices):raise ValueError('Wrong k mesh/indices')
with (a.source/'state.pkl').open('rb') as f:full=pickle.load(f)
seed={key:value[:,:1] for key,value in full.items()};seed['occ']=seed['occ']*64
c=ConfigDict(m['config']);c.k_grid_sizes=1;c.d3ft.parallel.k_pool=False;c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(a.source/'input.xyz')
c.d3ft.checkpoint=dict(root=str(a.output/'checkpoints'),interval=500)
c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
completed=[];previous_empty=None
with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,a.source) as info,validated_radial_splines():
 out=calc(c,init_params=seed,freq_mask=np.load(a.source/'mask.npy'));out.k_frac=np.asarray(pt['k_fractional']);out.density_grid=jnp.asarray(np.load(a.source/'density.npy'));out.grid_kmax=info['grid_kmax']
 factory=operator_factory(out)
 for k in indices:
  start=time.monotonic();op,ng=factory(k);nb=50;block=4
  @jax.jit
  def stream(x):
   pad=(-x.shape[1])%block;y=jnp.pad(x,((0,0),(0,pad))).reshape(ng,-1,block).transpose(1,0,2)
   z=jax.lax.map(jax.vmap(op,in_axes=1,out_axes=1),y)
   return z.transpose(1,0,2).reshape(ng,-1)[:,:x.shape[1]]
  mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(stream(jnp.asarray(x.reshape(-1,1))))[:,0],matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128)
  rng=np.random.default_rng(20261016+k)
  empty=previous_empty if previous_empty is not None else rng.normal(size=(ng,2))+1j*rng.normal(size=(ng,2))
  initial=np.linalg.qr(np.concatenate([np.asarray(full['pw'])[0,k],empty],axis=1))[0]
  diag=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((ng,ng),matvec=lambda x:diag*x.reshape(-1),matmat=lambda x:diag[:,None]*x,dtype=np.complex128)
  for attempt in range(a.ritz_restarts+1):
   vals,q=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=1200)
   q=np.linalg.qr(q)[0];hq=mat.matmat(q);vals,rotate=np.linalg.eigh((q.conj().T@hq+(q.conj().T@hq).conj().T)/2);q=q@rotate;hq=hq@rotate
   residuals=np.linalg.norm(hq-q*vals[None,:],axis=0);orth=float(np.max(abs(q.conj().T@q-np.eye(nb))))
   np.save(a.output/f'k_{k:04d}_attempt{attempt}_orbitals.npy',q)
   row=dict(k_index=k,k_fractional=pt['k_fractional'][k],eigenvalues_Ha=vals.tolist(),residuals_Ha=residuals.tolist(),orthogonality_max=orth,
    source_state_sha256=source_hash,source=str(a.source),seconds=time.monotonic()-start,bands=50,maxiter=1200,tol=1e-10,ritz_restart_attempt=attempt,
    passed=bool(np.max(residuals)<=1e-9 and orth<=1e-10))
   dump(a.output/f'k_{k:04d}_attempt{attempt}.json',row)
   print('K4_RITZ_ATTEMPT '+json.dumps(dict(k=k,attempt=attempt,max_residual_Ha=float(np.max(residuals)),passed=row['passed'])),flush=True)
   if row['passed']:break
   initial=q
  dump(a.output/f'k_{k:04d}.json',row)
  if not row['passed']:raise ValueError('Unchanged k4 local Ritz/orthogonality gate failed after saved restarts')
  previous_empty=q[:,48:50];completed.append(k)
  dump(a.output/'status.json',dict(passed=False,stage='local_spectra_running',completed=completed,total_requested=len(indices)))
 dump(a.output/'status.json',dict(passed=True,stage='complete',completed=completed))
