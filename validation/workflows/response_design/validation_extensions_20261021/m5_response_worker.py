"""Unchanged single-device KMeshResponse on the accepted64-k state. No soft fallback."""
import argparse,json,pickle,sys,gc
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--spectra',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();p=json.loads(a.protocol.read_text());R=Path(p['runtime']['mesh']);sys.path[:0]=[str(R),str(R/'validation/workflows/response_design'),str(R/'validation/workflows/response_design/reverse_response'),str(R/'validation/workflows/phase3')]
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import dump,sha,validated_radial_splines,HA_EV
from kmesh_response import KMeshResponse
from fixed_radial_program import source_radial_program
from radial_quadrature import local_simpson_patch
from bounded_xc import memory_bounded_xc
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from memory_probe import MemoryProbe
from mesh_target import select_mesh_target
a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output).start()
def stage(s,**kw):
 import datetime
 d={'time':datetime.datetime.now().astimezone().isoformat(),'stage':s,**kw};dump(a.output/'status.json',d);print('M5_STAGE '+json.dumps(d),flush=True)
try:
 if len(jax.devices())!=1:raise ValueError('Use existing one-device streamed implementation only')
 source=Path(p['source']);m=json.loads((source/'metadata.json').read_text())
 for name,h in p['source_hashes'].items():
  if sha(source/name)!=h:raise ValueError('Accepted4k source hash changed '+name)
 spec=json.loads(a.spectra.read_text());rows=sorted(spec['rows'],key=lambda r:r['k_index']);assert [r['k_index'] for r in rows]==list(range(64));vals=np.asarray([r['eigenvalues_Ha'] for r in rows]);_,target=select_mesh_target(vals,48,'hard');dump(a.output/'target.json',target)
 vectors=[]
 for row in rows:
  if not row['passed'] or max(row['residuals_Ha'])>1e-9 or row['source_state_sha256']!=p['source_hashes']['state.pkl']:raise ValueError('Spectral gate/source mismatch')
  vectors.append(np.load(row['orbitals_file']))
 vectors=np.stack(vectors)
 with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(source/'input.xyz')
 with c.d3ft.unlocked():
  for key in ['checkpoint','checkpoint_resume','charge_gate']:
   if key in c.d3ft:del c.d3ft[key]
 with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,source) as radial,validated_radial_splines():
  stage('read_only64k_reconstruction');out=calc(c,init_params=state,freq_mask=np.load(source/'mask.npy'));out.grid_kmax=radial['grid_kmax']
  stage('construct_existing_KMeshResponse');r=KMeshResponse(out,vectors,vals,p['theta0'],include_coordinates=True,objective='hard')
  charge=float(r.rho.mean()*r.volume);primal=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[:,None,:],axis=1)));dump(a.output/'primal.json',{'charge':charge,'residual_Ha':primal})
  if abs(charge-96)>96e-8 or primal>1e-7:raise ValueError('k4 rebuilt primal/charge gate failed')
  b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)));jax.block_until_ready(b);stage('adjoint_begin')
  lam,record=r.solve(b,lambda row:stage('adjoint_iteration',**row));gradient=np.asarray(r.reverse_gradient(lam))*HA_EV
  orth=float(jnp.max(abs(jnp.einsum('kga,kgb->kab',jnp.conj(r.q),lam))));translation=float(np.max(abs(gradient[6:].reshape(16,3).sum(axis=0))))
  if not np.isfinite(gradient).all() or orth>1e-9 or translation>1e-6:raise ValueError('k4 gradient projection/translation gate failed')
  old=np.asarray(json.loads(Path(p['k2_gradient_result']).read_text())['gradient_eV']);diff=gradient-old
  def compare(x,y):return {'cosine':float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),'norm_ratio_k4_over_k2':float(np.linalg.norm(x)/np.linalg.norm(y))}
  result={'passed':True,'objective':'hard','Nk':64,'n_variables':54,'gradient_eV':gradient.tolist(),'k2_gradient_eV':old.tolist(),'component_difference_eV':diff.tolist(),'canonical_mixed_unit_comparison':compare(gradient,old),'strain_comparison':compare(gradient[:6],old[:6]),'coordinate_comparison':compare(gradient[6:],old[6:]),'adjoint_record':record,'primal_residual_Ha':primal,'projection_max':orth,'translation_gradient_max':translation,'target':r.target_info}
  dump(a.output/'acceptance.json',result);stage('complete',passed=True)
except Exception as exc:stage('failed',error=repr(exc));raise
finally:probe.finish()
