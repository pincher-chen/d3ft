"""Independent Gamma low eigenspace on a new owned checkpoint, for E3 FD."""
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import sha,dump,validated_radial_splines
from bounded_xc import memory_bounded_xc
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'))
sys.path.insert(0,str(P/'validation/workflows/phase3'))
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from radial_quadrature import local_simpson_patch
from variational_ks import operator_factory
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(exist_ok=False);start=time.monotonic()
m=json.loads((a.source/'metadata.json').read_text());point=json.loads((a.source/'point.json').read_text())
if sha(a.source/'state.pkl')!=m['state_sha256']:raise ValueError('Owned checkpoint changed')
if point['residual']>1e-8 or point['acoustic_sum_rule_Ha_Bohr']>5e-8:raise ValueError('Unqualified parent')
c=ConfigDict(m['config']);c.epoch=1;c.d3ft.optim.lr_schedule='constant'
c.optimizer_args.learning_rate=0.;c.crystal_file_path_path=str(a.source/'input.xyz')
c.d3ft.parallel.k_pool=False
with (a.source/'state.pkl').open('rb') as f:state=pickle.load(f)
with plain_fft_patch(k_pool=False),local_simpson_patch(),validated_radial_splines(),memory_bounded_xc():
 out=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'))
 if len(out.k_frac)!=1:raise ValueError('E3 primal pilot is Gamma-only')
 factory=operator_factory(out);hop,ng=factory(0);apply=jax.jit(hop)
 batch=jax.jit(jax.vmap(hop,in_axes=1,out_axes=1))
 ne=float(sum(out.pseudopot.valence_charges));nocc=int(round(ne/2));nb=nocc+4;calls=[0]
 def action(x):
  calls[0]+=1
  return np.asarray(apply(jnp.asarray(x,dtype=jnp.complex128)))
 rng=np.random.default_rng(20261015)
 x=rng.normal(size=ng)+1j*rng.normal(size=ng);x/=np.linalg.norm(x)
 vals,q=eigsh(LinearOperator((ng,ng),matvec=action,dtype=np.complex128),
  k=nb,which='SA',ncv=min(ng,max(48,6*nb)),tol=1e-12,maxiter=5000,v0=x)
 q=np.linalg.qr(q)[0];hq=np.asarray(batch(jnp.asarray(q)))
 h=q.conj().T@hq;vals,rotation=np.linalg.eigh((h+h.conj().T)/2)
 q=q@rotation;hq=hq@rotation
 res=np.linalg.norm(hq-q*vals[None,:],axis=0)
 orth=float(np.max(abs(q.conj().T@q-np.eye(nb))))
 if np.max(res)>1e-9 or orth>1e-10:raise ValueError(f'Unqualified eigenspace {max(res)}, {orth}')
 vbm_sep=float(vals[nocc-1]-vals[nocc-2]);cbm_sep=float(vals[nocc+1]-vals[nocc])
 if min(vbm_sep,cbm_sep)<1e-5:raise ValueError('Band-edge branch unresolved')
 np.save(a.output/'orbitals.npy',q)
 summary=dict(eigenvalues_Ha=vals.tolist(),gap_eV=float((vals[nocc]-vals[nocc-1])*27.211386245988),
  n_occupied=nocc,vbm_adjacent_separation_Ha=vbm_sep,cbm_adjacent_separation_Ha=cbm_sep,
  maximum_ritz_residual_Ha=float(max(res)),orthogonality_max=orth,
  saved_orbitals_sha256=sha(a.output/'orbitals.npy'),seconds=time.monotonic()-start,matvec_calls=calls[0])
 dump(a.output/'summary.json',summary)
 dump(a.output/'metadata.json',dict(source_hashes={name:sha(a.source/name) for name in
  ['metadata.json','mask.npy','density.npy','state.pkl','input.xyz','point.json']},
  source=str(a.source),jobid=__import__('os').environ.get('SLURM_JOB_ID'),
  scope='Fresh independently solved own E3 Gamma spectrum, no earlier solver study data'))
 print('E3_PRIMAL_SPECTRUM',json.dumps(summary),flush=True)
