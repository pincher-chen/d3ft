"""Accepted-density NSCF parent-reference paths; physical edge vectors retained."""
import argparse,gc,hashlib,json,sys,time,os,pickle
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
P=Path(__file__).resolve().parents[3];R=P/'validation/results'
F=Path(os.environ['response_E8_RUNTIME'])
sys.path.insert(0,str(F/'validation/workflows/response_design/reverse_response'));sys.path.insert(0,str(F/'validation/workflows/phase3'));sys.path.insert(0,str(F/'validation/workflows/response_design'))
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from d3ft import pseudopotential as pp
from d3ft.grid.kpoints import k_vectors
from jrystal._src.grid import g_vectors,proper_grid_size
from variational_ks import operator_factory
from radial_quadrature import local_simpson_patch
from bounded_xc import memory_bounded_xc
# Reuse the vetted block algorithm without altering the frozen implementation.
solver_path=F/'validation/workflows/phase3/lowest_eigensystem.py'
solver_source=solver_path.read_text().replace('return vals,errors','return vals,errors,vecs')
solver_source=solver_source.replace('block_action=jax.jit(jax.vmap(op,in_axes=1,out_axes=1))', '''def block_action_fn(x):
        padding=(-x.shape[1])%4
        blocks=jnp.pad(x,((0,0),(0,padding))).reshape(dim,-1,4).transpose(1,0,2)
        result=jax.lax.map(jax.vmap(op,in_axes=1,out_axes=1),blocks)
        return result.transpose(1,0,2).reshape(dim,-1)[:,:x.shape[1]]
    block_action=jax.jit(block_action_fn)''')
solver_module={'__name__':'band_path_eigensolver'};exec(compile(solver_source,str(solver_path),'exec'),solver_module)
lowest=solver_module['lowest_block']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('state');ap.add_argument('output',type=Path);ap.add_argument('indices');a=ap.parse_args()
 protocol=read(a.protocol);s=protocol['states'][a.state];source=Path(s['source']);meta=read(source/'metadata.json');point=read(source/'point.json')
 for symbol,item in meta['physical']['pp'].items():
  if sha(item['path'])!=item['sha256']:raise ValueError('PP changed '+symbol)
 for n,h in s['source_sha256'].items():
  if sha(source/n)!=h:raise ValueError('Bound source changed: '+n)
 if not point['stationarity_pass'] or point['residual']>1e-6 or point['acoustic_sum_rule_Ha_Bohr']>5e-8:raise ValueError('Source acceptance failed')
 a.output.mkdir(parents=True,exist_ok=False)
 cfg=ConfigDict(meta['config']);cfg.epoch=1;cfg.optimizer_args.learning_rate=0.;cfg.d3ft.optim.lr_schedule='constant';cfg.crystal_file_path_path=str(source/'input.xyz');cfg.d3ft.parallel.k_pool=False
 if len(jax.devices())!=1:raise ValueError('Exactly one GPU required')
 kgrid=cfg.k_grid_sizes;cfg.k_grid_sizes=1;mask=np.load(source/'mask.npy');indices=json.loads(a.indices)
 with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
 with local_simpson_patch(),plain_fft_patch(k_pool=False),memory_bounded_xc():
  out=calc(cfg,init_params=state,freq_mask=mask)
  target=k_vectors(out.crystal.cell_vectors,proper_grid_size(kgrid),'gamma')
  g=g_vectors(out.crystal.cell_vectors,np.asarray(mask.shape))
  # Preserve the original SCF SBT table extent, checking that the entire path fits it.
  out.beta_kmax=pp.kmax_for_grid(g,target)
  frac=np.array(protocol['k_fractional']);cart=frac@np.asarray(out.crystal.reciprocal_vectors)
  selected=np.asarray(g).reshape(-1,3)[np.flatnonzero(mask.reshape(-1))]
  required=float(np.max(np.linalg.norm(selected[None,:,:]+cart[:,None,:],axis=-1)))
  if required>float(out.beta_kmax)+1e-10:raise ValueError('Path exceeds original projector table extent')
  out.beta_tables=pp.build_tables(out.pseudopot,out.beta_kmax);out.k_frac=frac;cfg.k_grid_sizes=kgrid
  density=np.load(source/'density.npy')
  if np.max(abs(np.asarray(out.density_grid)-density))>1e-9:raise ValueError('Read-only reconstruction changed the accepted source density')
  out.density_grid=jnp.asarray(density)
  volume=float(abs(np.linalg.det(out.crystal.cell_vectors)));electrons=float(np.asarray(out.get_occupation(out.params_occ,out.final_eps)).sum());charge=float(density.sum(axis=0).mean()*volume);nocc=int(round(electrons/2))
  if abs(electrons-2*nocc)>1e-6 or abs(charge-electrons)>1e-5*electrons:raise ValueError('Charge/occupation gate failed')
  factory=operator_factory(out);old=read(s['spectrum']);rows=[]
  dump(a.output/'metadata.json',dict(state=a.state,source=s,protocol_sha256=sha(a.protocol),worker_sha256=sha(__file__),solver_sha256=sha(solver_path),executed_solver_sha256=hashlib.sha256(solver_source.encode()).hexdigest(),FFT_band_block=4,runtime=str(F),occupied_bands=nocc,unoccupied_bands=8,electrons=electrons,density_electrons=charge,cell_bohr=np.asarray(out.crystal.cell_vectors).tolist(),Miller_mask_sha256=sha(source/'mask.npy'),devices=[str(d) for d in jax.devices()],beta_kmax=float(out.beta_kmax),requested_kmax=required,indices=indices))
  for i in indices:
   t=time.monotonic();op,dim=factory(i);vals,errs,vecs=lowest(op,dim,nocc+8,i)
   matched=[p for p in old['points'] if np.allclose(p['k_frac'],frac[i],rtol=0,atol=1e-10)]
   difference=max((float(np.max(abs(vals[:len(p['eigenvalues_Ha'])]-p['eigenvalues_Ha']))*27.211386245988) for p in matched),default=None)
   if difference is not None and difference>1e-6:raise ValueError(f'Existing spectrum mismatch {difference} eV at {i}')
   row=dict(index=i,k_frac=frac[i].tolist(),eigenvalues_Ha=vals.tolist(),residuals_Ha=errs.tolist(),vbm_Ha=float(vals[nocc-1]),cbm_Ha=float(vals[nocc]),seconds=time.monotonic()-t,existing_spectrum_max_difference_eV=difference)
   if i in protocol['edge_indices'].values() and a.state in ['start','reverse']:
    # Save both occupied and empty edge subspaces; no arbitrary splitting of degeneracies.
    chosen=sorted(set(int(j) for edge in [nocc-1,nocc] for j in np.flatnonzero(abs(vals-vals[edge])<1e-5)))
    np.savez_compressed(a.output/f'edge_{i:04d}.npz',coefficients=vecs[:,chosen],bands=np.array(chosen),eigenvalues_Ha=vals[chosen],k_fractional=frac[i],cell_bohr=np.asarray(out.crystal.cell_vectors),degeneracy_tolerance_Ha=np.array(1e-5))
    row['saved_edge_bands']=chosen
   dump(a.output/f'k_{i:04d}.json',row);rows.append(row);print('BAND_POINT',json.dumps({'state':a.state,'index':i,'seconds':row['seconds'],'max_residual_Ha':float(max(errs)),'parity_eV':difference}),flush=True)
   del op,vecs;jax.clear_caches();gc.collect()
  dump(a.output/'summary.json',dict(completed=True,indices=indices,total_points=len(rows),max_eigen_residual_Ha=max(max(r['residuals_Ha']) for r in rows)))
if __name__=='__main__':main()
