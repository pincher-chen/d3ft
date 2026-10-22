"""Staged MgSiN2 k2 production entry: fresh SCF, local spectrum, coupled adjoint.

No step launches itself. CPU coordinator must grant verified idle devices and
the preceding acceptance gates. All outputs are fresh owned response computations.
"""
import argparse,datetime,gc,json,os,pickle,sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from gamma_response import dump,sha,validated_radial_splines,HA_EV
from sharded_kmesh_response import ShardedKMeshResponse
from kmesh_response import KMeshResponse
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
from memory_probe import MemoryProbe
from projected_solver import real_inner
from kmesh_restart import gamma_orbital_warm_start
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from residual_report import fused_report
from variational_ks import operator_factory
from d3ft.calc.ground_state import calc
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--stage',choices=['scf','spectrum','response'],required=True);ap.add_argument('--k-index',type=int);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.protocol.read_text())
for f in p.get('scf_required_acceptance_files',p['required_acceptance_files']) if a.stage=='scf' else p['required_acceptance_files']:
 if not json.loads(Path(f).read_text())['passed']:raise ValueError('Preceding gate not passed '+f)
a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output,interval=.2 if p.get('single_device_streamed') else 1.).start()
def memory_gate(label):
 try:stats=jax.devices()[0].memory_stats() or {}
 except Exception:stats={}
 runtime_peak=stats.get('peak_bytes_in_use');sampled=probe.peak*1024**2
 if not sampled and runtime_peak is None:raise ValueError('No usable GPU memory measurement')
 peak=max(sampled,float(runtime_peak or 0));passed=peak<78*1024**3
 result=dict(passed=passed,stage=label,peak_bytes=peak,peak_GiB=peak/1024**3,limit_GiB=78,sampled_peak_MiB=probe.peak,runtime_memory_stats=stats,scope='Maximum runtime allocator peak if available and PID-specific sampled GPU usage; samples alone are a lower bound')
 dump(a.output/('memory_'+label+'.json'),result)
 if not passed:raise ValueError('Single-device memory gate >=78 GiB; do not submit full response')
 return result
def stage(name,**kw):
 row=dict(time=datetime.datetime.now().isoformat(),stage=name,jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),**kw);dump(a.output/'status.json',row);print('E5_MG',json.dumps(row),flush=True)
try:
 source=Path(p['source']);scf=Path(p['fresh_scf_directory']);m=json.loads((source/'metadata.json').read_text())
 for sym,pp in m['physical']['pp'].items():
  if sha(pp['path'])!=pp['sha256']:raise ValueError('Changed PP '+sym)
 c=ConfigDict(m['config']);c.crystal_file_path_path=str(source/'input.xyz');c.k_grid_sizes=2;c.d3ft.parallel.k_pool=True
 with (source/'state.pkl').open('rb') as f:parent=pickle.load(f)
 if a.stage=='scf':
  if len(jax.devices())!=p.get('device_count',8) or 8%len(jax.devices()):raise ValueError('Protocol device count must divide eight k blocks')
  c.epoch=p['epochs'];c.d3ft.optim.lr_schedule='cosine';c.d3ft.optim.lr_final=1e-6;c.optimizer_args.learning_rate=.01
  if c.occupation!='uniform' or c.d3ft.occupation.enabled:raise ValueError('E5 fixed insulating occupation protocol required')
  c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
  warm=gamma_orbital_warm_start(parent,8);stage('fresh_k2_reoptimization',epochs=c.epoch,occupation_initialization='target mesh uniform 2/Nk; Gamma occ not copied')
  with plain_fft_patch(k_pool=True),local_simpson_patch(),memory_bounded_xc(),validated_radial_splines():
   out=calc(c,init_params=warm,freq_mask=np.load(source/'mask.npy'));jax.block_until_ready(out.params_pw)
   np.save(a.output/'mask.npy',np.asarray(out.freq_mask));np.save(a.output/'density.npy',np.asarray(out.density_grid));np.save(a.output/'energy_history.npy',np.asarray(out.energy_history))
   with (a.output/'state.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5)
   with (a.output/'beta_tables.pkl').open('xb') as f:pickle.dump(out.beta_tables,f,protocol=5)
   (a.output/'input.xyz').write_bytes((source/'input.xyz').read_bytes())
   charge=float(np.asarray(out.density_grid).sum()*abs(np.linalg.det(np.asarray(out.crystal.cell_vectors)))/np.prod(out.density_grid.shape[-3:]))
   if p.get('smoke_only',False):
    history=np.asarray(out.energy_history)
    passed=abs(charge-96)<=1e-6 and np.isfinite(history).all() and np.isfinite(out.total_energy) and out.total_energy < p['smoke_energy_upper_Ha'] and out.total_energy > p['smoke_energy_lower_Ha'] and history[-1]<=history[0]+1e-6
    dump(a.output/'smoke_acceptance.json',dict(passed=bool(passed),charge=charge,expected_electrons=96,steps=len(history),energy_Ha=float(out.total_energy),energy_bounds_Ha=[p['smoke_energy_lower_Ha'],p['smoke_energy_upper_Ha']],electronic_energy_change_Ha=float(history[-1]-history[0]),scope='100-step charge/finite-energy/decrease check only; not a converged SCF certificate'))
    if not passed:raise ValueError('100-step smoke charge/energy gate failed')
    stage('completed',passed=True,scope='smoke only; no spectrum/response/E6 release');sys.exit(0)
   report=fused_report(out);forces=force_report(out)
   point=dict(energy_Ha=float(out.total_energy),residual=float(report['variational_residual']),acoustic_sum_rule_Ha_Bohr=float(forces['acoustic_sum_rule']),forces_Ha_Bohr=np.asarray(forces['forces_Ha_Bohr']).tolist(),
    stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),charge=charge,electrons=96.,k_fractional=np.asarray(out.k_frac).tolist(),stationarity_pass=float(report['variational_residual'])<=1e-8)
   dump(a.output/'point.json',point)
   physical={**m['physical'],'k':[2]}
   saved_config=c.to_dict();saved_config['d3ft']['charge_gate'].pop('report_path',None)
   dump(a.output/'metadata.json',dict(config=saved_config,physical=physical,source=str(source),state_sha256=sha(a.output/'state.pkl'),
    source_hashes={n:sha(source/n) for n in ['metadata.json','state.pkl','mask.npy','input.xyz']},
    warm_start='Own Gamma orbitals broadcast only; target-mesh uniform occupations freshly initialized at 2/Nk; all k2 electrons reoptimized here',new_response_state=True,
    frozen_radial_program=dict(tables_file='beta_tables.pkl',tables_sha256=sha(a.output/'beta_tables.pkl'),grid_kmax=float(out.grid_kmax),beta_kmax=float(out.beta_kmax)),
    code_sha256={str(f.relative_to(P)):sha(f) for f in (P/'d3ft').rglob('*.py')}))
   if not point['stationarity_pass'] or point['acoustic_sum_rule_Ha_Bohr']>5e-8 or abs(charge-96)>96e-8:raise ValueError('Fresh k2 physical state failed')
 elif a.stage=='spectrum':
  if len(jax.devices())!=1 or a.k_index not in range(8):raise ValueError('One-device local k spectrum')
  m=json.loads((scf/'metadata.json').read_text());point=json.loads((scf/'point.json').read_text())
  if not point['stationarity_pass']:raise ValueError('Fresh SCF not accepted')
  with (scf/'state.pkl').open('rb') as f:state=pickle.load(f)
  ik=a.k_index;state={name:x[:,ik:ik+1] for name,x in state.items()};state['occ']=state['occ']*8;c=ConfigDict(m['config']);c.k_grid_sizes=1;c.d3ft.parallel.k_pool=False;c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(scf/'input.xyz')
  c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
  with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,scf) as tables,validated_radial_splines():
   out=calc(c,init_params=state,freq_mask=np.load(scf/'mask.npy'));out.k_frac=np.array(point['k_fractional'])[ik:ik+1];out.density_grid=jnp.asarray(np.load(scf/'density.npy'));out.grid_kmax=tables['grid_kmax']
   op,ng=operator_factory(out)(0);nb=52;block=4
   @jax.jit
   def stream(x):
    pad=(-x.shape[1])%block;y=jnp.pad(x,((0,0),(0,pad))).reshape(ng,-1,block).transpose(1,0,2)
    z=jax.lax.map(jax.vmap(op,in_axes=1,out_axes=1),y)
    return z.transpose(1,0,2).reshape(ng,-1)[:,:x.shape[1]]
   mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(stream(jnp.asarray(x.reshape(-1,1))))[:,0],matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128)
   rng=np.random.default_rng(20261015+ik);initial=np.linalg.qr(np.concatenate([state['pw'][0,0],rng.normal(size=(ng,4))+1j*rng.normal(size=(ng,4))],axis=1))[0]
   restart=p.get('spectrum_restart',{}).get(str(ik))
   if restart:
    if sha(restart['orbitals'])!=restart['sha256'] or restart['source_state_sha256']!=sha(scf/'state.pkl'):raise ValueError('Changed spectral restart/source')
    initial=np.load(restart['orbitals'])
    if initial.shape!=(ng,nb):raise ValueError('Spectral restart dimensions mismatch')
    initial=np.linalg.qr(initial)[0]
   diagonal=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((ng,ng),matvec=lambda x:diagonal*x.reshape(-1),matmat=lambda x:diagonal[:,None]*x,dtype=np.complex128)
   vals,q=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=p.get('spectrum_maxiter',400));q=np.linalg.qr(q)[0];hq=mat.matmat(q);h=q.conj().T@hq;vals,rotate=np.linalg.eigh((h+h.conj().T)/2);q=q@rotate;hq=hq@rotate
   residuals=np.linalg.norm(hq-q*vals[None,:],axis=0);orth=float(np.max(abs(q.conj().T@q-np.eye(nb))))
   np.save(a.output/'orbitals.npy',q);dump(a.output/'summary.json',dict(k_index=ik,k_fractional=point['k_fractional'][ik],eigenvalues_Ha=vals.tolist(),residuals_Ha=residuals.tolist(),orthogonality_max=orth,
    source=str(scf),source_state_sha256=sha(scf/'state.pkl'),saved_orbitals_sha256=sha(a.output/'orbitals.npy'),restart=restart,maxiter=p.get('spectrum_maxiter',400)))
   if max(residuals)>1e-9 or orth>1e-10:raise ValueError('Local k eigenpair gate failed')
 else:
  if len(jax.devices())!=p.get('device_count',8) or 8%len(jax.devices()):raise ValueError('Protocol device count must divide eight k blocks')
  m=json.loads((scf/'metadata.json').read_text());c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(scf/'input.xyz')
  with (scf/'state.pkl').open('rb') as f:state=pickle.load(f)
  c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
  serial=p.get('single_device_streamed',False)
  if serial:
   if len(jax.devices())!=1:raise ValueError('Exactly one visible GPU; no cross-device sharding allowed')
   c.k_grid_sizes=1;c.d3ft.parallel.k_pool=False
   seed={name:x[:,:1] for name,x in state.items()};seed['occ']=seed['occ']*8
  rows=[];vectors=[]
  for ik in range(8):
   orb=Path(p['spectrum_directories'][ik]) if p.get('spectrum_directories') else Path(p['spectrum_root'])/f'k{ik}';row=json.loads((orb/'summary.json').read_text())
   if sha(scf/'state.pkl')!=row['source_state_sha256'] or sha(orb/'orbitals.npy')!=row['saved_orbitals_sha256'] or max(row['residuals_Ha'])>1e-9:raise ValueError('Changed/failed local k spectrum')
   rows.append(row);vectors.append(np.load(orb/'orbitals.npy'))
  with plain_fft_patch(k_pool=not serial),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,scf) as tables,validated_radial_splines():
   out=calc(c,init_params=seed if serial else state,freq_mask=np.load(scf/'mask.npy'));out.grid_kmax=tables['grid_kmax']
   if serial:
    # Reconstruct geometry/PP on one Gamma block only. Never expand eight
    # occupied FFT tensors in calc: the response streams compact k orbitals.
    out.k_frac=np.asarray(json.loads((scf/'point.json').read_text())['k_fractional'])
    out.params_pw=jnp.asarray(state['pw']);out.params_occ=jnp.asarray(state['occ']);out.density_grid=jnp.asarray(np.load(scf/'density.npy'));out.config.k_grid_sizes=2
    jax.clear_caches();gc.collect()
   r=(KMeshResponse if serial else ShardedKMeshResponse)(out,np.asarray(vectors),np.array([x['eigenvalues_Ha'] for x in rows]),p['theta0'],include_coordinates=True)
   residual=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[:,None,:],axis=1)));charge=float(r.rho.mean()*r.volume)
   if residual>1e-7 or abs(charge-96)>96e-8:raise ValueError('Coupled rebuilt primal mismatch')
   b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)))
   if serial and p.get('pcg_smoke_only',False):
    first=r.hessian(b);jax.block_until_ready(first);del first
    memory_gate('before_PCG');stage('single_GPU_memory_gate_passed')
   if p.get('pcg_smoke_only',False):
    import projected_solver as solver
    comparisons=[];iterations=[]
    class TwoStepsDone(Exception):pass
    def compare_inner(aa,bb,value):
     reference=float(np.real(np.vdot(aa,bb)));actual=float(value);error=abs(actual-reference);tol=1e-9+1e-11*abs(reference)
     comparisons.append(dict(reference=reference,actual=actual,error=error,tolerance=tol,passed=bool(np.isfinite(actual) and error<=tol)))
     if not comparisons[-1]['passed']:raise ValueError('GPU/CPU inner comparison failed')
    def two_steps(row):
     iterations.append(row);stage('two_step_PCG_probe',**row)
     if row['iteration']==2:raise TwoStepsDone()
    solver.inner_probe=compare_inner
    try:r.solve(b,two_steps)
    except TwoStepsDone:pass
    finally:solver.inner_probe=None
    passed=len(iterations)==2 and bool(comparisons) and all(x['passed'] for x in comparisons)
    memory=memory_gate('after_PCG') if serial else None
    dump(a.output/'smoke_acceptance.json',dict(passed=passed,iterations=iterations,inner_comparisons=comparisons,memory=memory,single_device_streamed=serial,scope='Actual Mg two PCG iterations; each scalar compared with independent NumPy dot on same vectors. This is not a CPU DFT Hessian or convergence certificate',primal_residual_Ha=residual,charge=charge))
    if not passed:raise ValueError('Actual Mg two-step probe failed')
    stage('completed',passed=True,scope='two-iteration smoke only; full adjoint not certified');sys.exit(0)
   if p.get('validate_first_two_inner',False):
    import projected_solver as solver
    inner_records=[];online_iteration=[1];online_gate=[False]
    def online_inner(aa,bb,value):
     reference=float(np.real(np.vdot(aa,bb)));actual=float(value);error=abs(actual-reference);tol=1e-9+1e-11*abs(reference)
     entry=dict(iteration=online_iteration[0],reference=reference,actual=actual,error=error,tolerance=tol,passed=bool(np.isfinite(actual) and error<=tol),shape_a=list(aa.shape),shape_b=list(bb.shape),scope='NumPy dot on the exact same two runtime vectors used for GPU contraction')
     inner_records.append(entry)
     with (a.output/'inner_comparisons.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
     if not entry['passed']:raise ValueError('Online CPU/GPU inner-product gate failed')
    def online_iteration_check(row):
     i=row['iteration']
     if i<=2:
      entries=[x for x in inner_records if x['iteration']==i]
      if len(entries)<2 or not all(x['passed'] for x in entries):raise ValueError('Missing/failed online inner records at iteration '+str(i))
      dump(a.output/f'inner_iteration_{i}.json',dict(iteration=i,passed=True,comparisons=entries,PCG=row))
      if i==2:
       online_gate[0]=True;dump(a.output/'first_two_inner_acceptance.json',dict(passed=True,iterations=[1,2],comparisons=inner_records,scope='Online first-two-iteration CPU/GPU contractions; not a full convergence certificate'));solver.inner_probe=None
      online_iteration[0]=i+1
     stage('adjoint_iteration',**row)
    solver.inner_probe=online_inner
    try:lam,record=r.solve(b,online_iteration_check)
    finally:solver.inner_probe=None
    if not online_gate[0]:raise ValueError('First-two-iteration inner gate not completed')
   else:lam,record=r.solve(b,lambda row:stage('adjoint_iteration',**row))
   gradient=np.asarray(r.reverse_gradient(lam))*HA_EV
   directions=np.asarray(p['tangent_directions']);checks=[];bare=r.bare(r.zero)
   for i,d in enumerate(directions):
    u,rec=r.solve(-r.project(r.external(jnp.asarray(d))),lambda row:stage('tangent_iteration',direction=i,**row));forward=float((jnp.dot(bare,d)+4*r.w*real_inner(b,u))*HA_EV);reverse=float(gradient@d)
    checks.append(dict(direction=i,forward_eV=forward,reverse_eV=reverse,difference=abs(forward-reverse),passed=abs(forward-reverse)<=5e-6+1e-6*max(abs(forward),abs(reverse)),record=rec))
   translation=float(np.max(abs(gradient[6:].reshape(16,3).sum(axis=0))));passed=all(x['passed'] for x in checks) and translation<=1e-6
   full_memory=memory_gate('full_response') if serial else None
   result=dict(time=datetime.datetime.now().isoformat(),passed=passed,gradient_eV=gradient.tolist(),target=r.target_info,adjoint_record=record,tangent_checks=checks,translation_sum_max=translation,primal_residual_Ha=residual,
    sharding={name:str(getattr(r,name).sharding) for name in ['q','edges','diagonal','projectors']},single_device_streamed=serial,memory=full_memory,
    scope='Fresh MgSiN2 k2 '+('single-device streamed' if serial else 'sharded')+' coupled response/mode pilot; mixed FD, actual design and k4 transfer not yet certified')
   dump(a.output/'result.json',result)
   if not passed:raise ValueError('Mg k2 response pilot failed')
 stage('completed',passed=True)
except Exception as exc:stage('failed',error=str(exc));raise
finally:probe.finish()
