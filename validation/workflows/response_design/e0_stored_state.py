"""Evaluate an owned real checkpoint without optimizing it (E0 migration parity)."""
import argparse, datetime, hashlib, json, pickle, time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.calc.stress import free_energy_at_strain, ewald_translation_indices

def dump(p,d):
 p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 def stage(s):
  dump(a.output/'status.json',dict(stage=s,time=datetime.datetime.now().isoformat()));print('E0_STAGE',s,flush=True)
 stage('verify_real_checkpoint');m=json.loads((a.source/'metadata.json').read_text());sf=a.source/'state.pkl';h=hashlib.sha256(sf.read_bytes()).hexdigest()
 if m['state_sha256']!=h:raise ValueError('Checkpoint hash mismatch')
 with sf.open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(a.source/'input.xyz')
 from d3ft.parallel import plain_fft_patch
 with ExitStack() as stack:
  stack.enter_context(plain_fft_patch(k_pool=False))
  if m.get('physical',{}).get('local_radial_quadrature')=='simpson':
   from radial_quadrature import local_simpson_patch
   stack.enter_context(local_simpson_patch())
  stage('reconstruct_zero_update');out=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'));jax.block_until_ready(out.params_pw)
  unchanged=max(float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(state['pw']),jax.tree_util.tree_leaves(out.params_pw)))
  if unchanged>1e-14:raise ValueError(f'Zero learning rate altered saved orbitals: {unchanged}')
  kw=dict(config=out.config,crystal=out.crystal,freq_mask=out.freq_mask,make_coeff=out.make_coeff,get_occupation=out.get_occupation,pseudopot=out.pseudopot,beta_tables=out.beta_tables,kmax=out.beta_kmax,grid_kmax=out.grid_kmax,k_frac=out.k_frac,temperature=out.temperature,eps=out.final_eps,theta_nonlocal=float(c.d3ft.get('theta_nonlocal',0)),theta_nonlocal_l=c.d3ft.get('theta_nonlocal_l'),xc_params=c.d3ft.get('xc_params'),ewald_indices=ewald_translation_indices(out.crystal.cell_vectors,c.ewald_args['ewald_cutoff']))
  from functools import partial
  from d3ft.pseudopotential import local
  original=local.CubicSpline;local.CubicSpline=partial(original,check=False)
  try:
   stage('energy_and_complete_gradients');start=time.monotonic()
   f=lambda pw,occ,s,d:free_energy_at_strain(s,pw,occ,displacement=d,**kw)
   value,grads=jax.jit(jax.value_and_grad(f,argnums=(0,1,2,3)))(out.params_pw,out.params_occ,jnp.zeros((3,3)),jnp.zeros_like(out.crystal.positions));jax.block_until_ready(grads)
   arrays={}
   for label,g in zip(['pw','occ','strain','positions'],grads):
    for i,x in enumerate(jax.tree_util.tree_leaves(g)): arrays[f'{label}_{i}']=np.asarray(x)
   np.savez(a.output/'gradients.npz',**arrays)
   dump(a.output/'result.json',dict(passed_evaluation=True,energy_Ha=float(value),checkpoint_sha256=h,checkpoint_max_change=unchanged,core_density_is_none=out.core_density_grid is None,gradient_leaf_shapes={k:list(v.shape) for k,v in arrays.items()},seconds=time.monotonic()-start,scope='Parity on a real stored pilot state, not a claim of physical stationarity'))
   stage('completed')
  finally:local.CubicSpline=original
if __name__=='__main__':main()
