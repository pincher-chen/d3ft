"""Independent low spectrum at every owned k point; fresh results, no earlier solver study data."""
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import sha,dump,validated_radial_splines,HA_EV
from bounded_xc import memory_bounded_xc
from fixed_radial_program import source_radial_program
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from variational_ks import operator_factory
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
a.output.mkdir(exist_ok=False);m=json.loads((a.source/'metadata.json').read_text());p=json.loads((a.source/'point.json').read_text());start=time.monotonic()
if sha(a.source/'state.pkl')!=m['state_sha256'] or p['residual']>1e-8 or p['acoustic_sum_rule_Ha_Bohr']>5e-8:raise ValueError('Owned stationary primal gate failed')
with (a.source/'state.pkl').open('rb') as f:state=pickle.load(f)
c=ConfigDict(m['config']);c.epoch=1;c.d3ft.optim.lr_schedule='constant';c.optimizer_args.learning_rate=0.;c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(a.source/'input.xyz')
with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,a.source) as info:
 out=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'))
 if info:out.grid_kmax=info['grid_kmax']
 with validated_radial_splines():factory=operator_factory(out)
 ne=float(sum(out.pseudopot.valence_charges));nocc=int(round(ne/2));nb=nocc+4;vectors=[];values=[];rows=[]
 for ik in range(len(out.k_frac)):
  op,ng=factory(ik);action=jax.jit(op);batch=jax.jit(jax.vmap(op,in_axes=1,out_axes=1));calls=[0]
  def matvec(x):calls[0]+=1;return np.asarray(action(jnp.asarray(x,dtype=jnp.complex128)))
  rng=np.random.default_rng(20261015+ik);x=rng.normal(size=ng)+1j*rng.normal(size=ng);x/=np.linalg.norm(x)
  vals,q=eigsh(LinearOperator((ng,ng),matvec=matvec,dtype=np.complex128),k=nb,which='SA',ncv=min(ng,max(48,6*nb)),tol=1e-12,maxiter=4000,v0=x)
  q=np.linalg.qr(q)[0];hq=np.asarray(batch(jnp.asarray(q)));h=q.conj().T@hq;vals,rotate=np.linalg.eigh((h+h.conj().T)/2);q=q@rotate;hq=hq@rotate
  residuals=np.linalg.norm(hq-q*vals[None,:],axis=0);orth=float(np.max(abs(q.conj().T@q-np.eye(nb))))
  if max(residuals)>1e-9 or orth>1e-10:raise ValueError(f'Independent eigenpair gate failed k{ik}')
  row=dict(k_index=ik,k_fractional=np.asarray(out.k_frac[ik]).tolist(),eigenvalues_Ha=vals.tolist(),residuals_Ha=residuals.tolist(),orthogonality_max=orth,matvec_calls=calls[0]);dump(a.output/f'k_{ik:03d}.json',row);rows.append(row);vectors.append(q);values.append(vals)
  print('E5_K_SPECTRUM',ik,'max_residual',float(max(residuals)),flush=True)
 np.save(a.output/'orbitals.npy',np.asarray(vectors))
 dump(a.output/'summary.json',dict(eigenvalues_Ha=np.asarray(values).tolist(),n_occupied=nocc,nk=len(values),k_fractional=np.asarray(out.k_frac).tolist(),
  maximum_ritz_residual_Ha=max(max(r['residuals_Ha']) for r in rows),saved_orbitals_sha256=sha(a.output/'orbitals.npy'),seconds=time.monotonic()-start))
 dump(a.output/'metadata.json',dict(source=str(a.source),source_hashes={n:sha(a.source/n) for n in ['state.pkl','metadata.json','mask.npy','density.npy','point.json','input.xyz']},jobid=__import__('os').environ.get('SLURM_JOB_ID'),scope='Independent full registered k spectrum on own stationary density; no new SCF claim'))
