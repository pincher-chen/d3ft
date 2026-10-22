"""Approved layout-only64k response. One8GPUprocess; no extra tangent solves."""
import argparse,datetime,gc,json,pickle,sys,time
from pathlib import Path
import numpy as np
start=time.monotonic()
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--remaining-gpu-hours',required=True,type=float);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();p=json.loads(a.protocol.read_text());rt=Path(p['runtime_path']);sys.path[:0]=[str(rt),str(rt/'validation/workflows/response_design/reverse_response'),str(rt/'validation/workflows/response_design'),str(rt/'validation/workflows/phase3')]
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import dump,sha,HA_EV,validated_radial_splines
from layout_sharded_response import ExplicitLayoutResponse
from mesh_target import select_mesh_target
from fixed_radial_program import source_radial_program
from radial_quadrature import local_simpson_patch
from bounded_xc import memory_bounded_xc
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from memory_probe import MemoryProbe
a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output).start();inner_records=[]
def stage(name,**kw):
 d={'time':datetime.datetime.now().astimezone().isoformat(),'stage':name,**kw};dump(a.output/'status.json',d);print('M5_SHARDED',json.dumps(d),flush=True)
def memory_gate(label):
 peaks=[probe.peak/1024];stats=[]
 for device in jax.devices():
  try:q=device.memory_stats() or {}
  except Exception:q={}
  stats.append({'device':device.id,'peak_bytes_in_use':q.get('peak_bytes_in_use')})
  if q.get('peak_bytes_in_use') is not None:peaks.append(q['peak_bytes_in_use']/2**30)
 peak=max(peaks);d={'passed':peak>0 and peak<78,'peak_GiB':peak,'limit_GiB':78,'per_device_runtime':stats,'sampled_peak_MiB':probe.peak,'sampling_is_lower_bound':True};dump(a.output/('memory_'+label+'.json'),d)
 if not d['passed']:raise ValueError('Original78GiB safety check failed')
 return d
try:
 if len(jax.devices())!=8 or jax.default_backend()!='gpu':raise ValueError('Eight actualGPUdevices required')
 source=Path(p['source']);m=json.loads((source/'metadata.json').read_text())
 for name,h in p['source_hashes'].items():
  if sha(source/name)!=h:raise ValueError('Source hash changed '+name)
 for sym,pp in m['physical']['pp'].items():
  if sha(pp['path'])!=pp['sha256']:raise ValueError('UPF changed '+sym)
 rows=sorted(json.loads((Path(p['original_root'])/'M5-A/reconstructed_spectrum.json').read_text())['rows'],key=lambda r:r['k_index']);assert [r['k_index'] for r in rows]==list(range(64));vectors=[]
 for row in rows:
  file=Path(row['orbitals_file'])
  if not row['passed'] or max(row['residuals_Ha'])>1e-9 or row['orthogonality_max']>1e-10 or row['source_state_sha256']!=p['source_hashes']['state.pkl'] or sha(file)!=row['orbitals_SHA256']:raise ValueError('Previouslyaccepted spectrum changed')
  vectors.append(np.load(file))
 vals=np.array([row['eigenvalues_Ha'] for row in rows]);_,target=select_mesh_target(vals,48,'hard');dump(a.output/'target.json',target)
 with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=True;c.crystal_file_path_path=str(source/'input.xyz')
 with c.d3ft.unlocked():
  for key in ['checkpoint','checkpoint_resume','charge_gate']:
   if key in c.d3ft:del c.d3ft[key]
 with plain_fft_patch(k_pool=True),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,source) as radial,validated_radial_splines():
  stage('kpool_readonly_reconstruction',devices=8,k_per_device=8)
  out=calc(c,init_params=state,freq_mask=np.load(source/'mask.npy'));jax.block_until_ready(out.params_pw);out.grid_kmax=radial['grid_kmax'];density_error=float(np.max(abs(np.asarray(out.density_grid)-np.load(source/'density.npy'))));dump(a.output/'readonly_density.json',{'max_abs_error':density_error,'unchanged_gate':1e-9})
  if density_error>1e-9:raise ValueError('Source density changed inread-only reconstruction')
  jax.clear_caches();gc.collect();memory_gate('read_only')
  stage('construct_explicit_layout_response');r=ExplicitLayoutResponse(out,np.asarray(vectors),vals,p['theta0'],include_coordinates=True,objective='hard');del vectors,state;gc.collect()
  inventory={}
  for name in ['q','edges','values','target_weights','k_frac','diagonal','projectors']:
   x=getattr(r,name);shards=x.addressable_shards
   if len(shards)!=8 or any(sh.data.shape[0]!=8 for sh in shards):raise ValueError('Global k array replicated or wrongownership: '+name)
   inventory[name]={'shape':list(x.shape),'local_shapes':[list(sh.data.shape) for sh in shards],'sharding':str(x.sharding)}
  dump(a.output/'layout_inventory.json',inventory)
  charge=float(r.rho.mean()*r.volume);primal_fn=jax.jit(lambda:jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[:,None,:],axis=1)),out_shardings=r.rep);primal=float(primal_fn());dump(a.output/'primal.json',{'charge':charge,'residual_Ha':primal,'density_max_abs_diff_saved':float(np.max(abs(np.asarray(r.rho)-np.load(source/'density.npy'))))})
  if abs(charge-96)>96e-8 or primal>1e-7:raise ValueError('Originalcharge/primal check failed')
  b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)));jax.block_until_ready(b);memory_gate('before_PCG')
  iteration=[1];solver_previous_seconds=[0.];steady_times=[]
  def inner_check(aa,bb,value):
   ref=float(np.vdot(aa,bb).real);err=abs(value-ref);tol=1e-9+1e-11*abs(ref);d={'iteration':iteration[0],'reference':ref,'actual':value,'error':err,'tolerance':tol,'passed':err<=tol};inner_records.append(d)
   with (a.output/'inner_comparisons.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
   if not d['passed']:raise ValueError('Device/NumPy inner product mismatch')
  r.inner_probe=inner_check
  def callback(row):
   i=row['iteration'];elapsed=time.monotonic()-start
   delta_seconds=row['seconds']-solver_previous_seconds[0];solver_previous_seconds[0]=row['seconds']
   if i>=2:steady_times.append(delta_seconds)
   if i<=2:
    records=[q for q in inner_records if q['iteration']==i]
    if not records:raise ValueError('Missingrealdeviceinnercomparison')
    memory_gate('PCG'+str(i));dump(a.output/f'inner_iteration_{i}.json',{'passed':all(q['passed'] for q in records),'records':records,'PCG':row})
    if i==2:r.inner_probe=None;dump(a.output/'real_two_iteration_acceptance.json',{'passed':True,'primal_Ha':primal,'charge':charge,'inner_records':len(inner_records)})
   iteration[0]=i+1
   per=max(steady_times[-3:]) if steady_times else 220.;remaining=max(32-i,2)*per+1440;predicted_new_GPUh=8*(elapsed+remaining)/3600;stage('adjoint_iteration',**row,predicted_new_GPUh=predicted_new_GPUh,allowed_new_GPUh=a.remaining_gpu_hours,steady_iteration_seconds_used=per)
   if predicted_new_GPUh>a.remaining_gpu_hours:raise RuntimeError('BudgetPredictionStop: totalwouldexceed35GPUh; unchangedsolvercriteria')
  stage('adjoint_begin');lam,record=r.solve(b,callback);memory_gate('converged_PCG');stage('geometry_reverse_gradient');gradient=np.asarray(r.reverse_gradient(lam))*HA_EV
  orth=float(jnp.max(abs(jnp.einsum('kga,kgb->kab',jnp.conj(r.q),lam))));translation=float(np.max(abs(gradient[6:].reshape(16,3).sum(axis=0))))
  if not np.isfinite(gradient).all() or orth>1e-9 or translation>1e-6:raise ValueError('Originalgradientprojection/translation check failed')
  old=np.array(json.loads(Path(p['k2_gradient_result']).read_text())['gradient_eV']);delta=gradient-old
  def compare(x,y):return {'cosine':float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),'norm_ratio_k4_over_k2':float(np.linalg.norm(x)/np.linalg.norm(y))}
  result={'time':datetime.datetime.now().astimezone().isoformat(),'passed':True,'Nk':64,'devices':8,'k_per_device':8,'objective':'hard','gradient_eV':gradient.tolist(),'k2_gradient_eV':old.tolist(),'component_difference_eV':delta.tolist(),'canonical_mixed_unit_comparison':compare(gradient,old),'strain_comparison':compare(gradient[:6],old[:6]),'coordinate_comparison':compare(gradient[6:],old[6:]),'adjoint_record':record,'primal_residual_Ha':primal,'projection_max':orth,'translation_gradient_max':translation,'memory':memory_gate('full_response'),'active_GPU_hours':8*(time.monotonic()-start)/3600,'no_response_mathematical_changes':True,'scope':'hard64kfull54gradient; mixed-unit and separated comparisons, no closeness gate for k2vs k4'};dump(a.output/'acceptance.json',result);stage('complete',passed=True)
except Exception as ex:stage('failed',error=repr(ex));raise
finally:probe.finish()
