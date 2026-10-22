"""Bounded true-DFT three-parameter full fit + 3 material LOO folds.

Only Si/BN/diamond reference rows enter fits. Fixed kappa/mu/theta_p is the
latest user decision; rank-deficient folds use the declared LM regularizer and
are not claimed to uniquely identify three parameters. External holdouts stay sealed.
"""
import argparse,datetime as dt,fcntl,json,os,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from e1_three_parameter_queue import read,dump,alive
from run_identity import reserve_name
CORE=['Si','BN','diamond'];AXES=np.array([5,6,1]);SCALE=np.array([.1,.02,.05]);BOUNDS=np.array([.0804,.0219514973,.05])

def dataset(materials,states,references):
 if not set(materials)<=set(CORE):raise ValueError('Unapproved/holdout material in fit')
 rows=[];r=[];J=[]
 for material in materials:
  state=states[material];refs=references[material];n=len(refs)
  for name,target in refs.items():
   denom=(.005 if name=='a0_A' else .05)*target*np.sqrt(n)
   grad=state['jacobian'][name]
   r.append((state['summary'][name]-target)/denom);J.append(np.asarray(grad)[AXES]*SCALE/denom);rows.append([material,name])
 return np.asarray(r),np.asarray(J),rows

def proposal(vector,r,J,damping,radius):
 u,s,vt=np.linalg.svd(J,full_matrices=False);z=-vt.T@((s/(s*s+damping))*(u.T@r));z*=min(1.,radius/max(np.linalg.norm(z),1e-30))
 out=np.asarray(vector).copy();out[AXES]=np.clip(out[AXES]+SCALE*z,-BOUNDS,BOUNDS)
 rank=int(np.sum(s>max(s.max(),1.)*1e-10));pred=r+J@((out[AXES]-np.asarray(vector)[AXES])/SCALE)
 return out,dict(loss_before=float(r@r),predicted_loss=float(pred@pred),singular_values=s.tolist(),rank=rank,parameters=3,unique_three_parameter_identifiability=rank==3,damping=damping,radius=radius)

def accept(before,after,predicted):
 reduction=before-after;rho=reduction/(before-predicted) if before>predicted else None
 return dict(accepted=bool(rho is not None and rho>.1 and reduction>0),loss_before=before,loss_after=after,predicted_loss=predicted,actual_reduction=reduction,rho=rho)

def priority_dependency_finished(c):
 """Missing/running upstream evidence never releases GPUs to training."""
 index=c.get('priority_dependency_index')
 if not index:return not c.get('gate_E5_before_training',False)
 try:
  root=Path(read(index)['root']);status=read(root/'status.json')
 except (OSError,ValueError,KeyError):return False
 if status.get('stage')=='completed':return (root/'acceptance.json').exists()
 # Timeouts/budget stops are infrastructure interruptions, not failed physics.
 return (status.get('classification')=='physical_gate_failure'
     and (root/'physical_failure_acceptance.json').exists())

class InfrastructureInterruption(RuntimeError):pass

def main(T):
 import e7_ct_pilot as pilot
 R=Path(read(T/'protocol.json')['runtime']);sys.path.insert(0,str(R/'validation/workflows/phase3'));from equilibrium_jacobian import derive
 os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY));c=read(T/'protocol.json');P=Path(c['project']);Y=pilot.E;Q=Y/'gpu_reservations.json';spent=c.get('initial_training_GPU_hours',0.);groups=c['gpus'];cache={};allfits={};running=[];probe_lock=threading.Lock();policy_lock=threading.Lock();probe_time=0.;last_memory={}
 from e7_shared_budget import SharedBudget
 shared=SharedBudget(c['shared_budget_file']) if c.get('shared_budget_file') else None
 def now():return dt.datetime.now().astimezone().isoformat()
 def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,gpus=groups,training_GPU_hours=spent,E7_total_spent_GPU_hours=shared.summary()["spent_GPU_hours"] if shared else pilot.spent()+spent,E7_budget_GPU_hours=400,**kw))
 def note(s):
  for p in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
   with p.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
 def release(which):
  with Q.with_name('gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   for g in which:
    if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
   dump(Q,owners)
 def memory():
  nonlocal probe_time,last_memory
  with probe_lock:
   if time.monotonic()-probe_time<15:return last_memory
   r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45)
   last_memory={int(v[0]):int(v[1]) for l in r.stdout.splitlines() if len(v:=l.split(','))==2} if r.returncode==0 else {}
   probe_time=time.monotonic();dump(T/'resource_memory.json',dict(time=now(),memory_MiB=last_memory))
   return last_memory
 def allowed():
  with policy_lock:
   request=T/'yield_to_E5.json'
   if priority_dependency_finished(c):
    if request.exists():request.replace(T/'yield_to_E5_fulfilled.json')
    return True
   if not c.get('opportunistic_until_E5_ready'):return False
   if request.exists():return False  # Latch until E5 has accepted terminal evidence.
   mem=memory();external=c.get('E5_other_gpus',[0,1,5,6,7])
   if not all(g in mem for g in external):return False
   if any(mem[g]>3 for g in external):return True
   dump(request,dict(time=now(),reason='All E5 GPUs outside E7 are idle; finish current DFT point and exit75',memory_MiB=mem,do_not_cancel=True))
   return False
 def claim(which=None):
  which=groups if which is None else which
  if not allowed():return False
  cert=[read(Y/(m+'_independent_acceptance.json')) if (Y/(m+'_independent_acceptance.json')).exists() else {} for m in CORE]
  if not all(x.get('passed') for x in cert):return False
  mem=memory()
  if any(mem.get(g,10**9)>3 for g in which):return False
  with Q.with_name('gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   if any(str(g) in owners for g in which):return False
   for g in which:owners[str(g)]=dict(owner='E7 opportunistic full3+LOO3; cooperative point-boundary E5 priority',status_file=str(T/'status.json'),controller_pid=os.getpid())
   dump(Q,owners)
  return True
 def validate(material,folder,vector,settings,mask):
  folder=Path(folder);s=read(folder/'summary.json');m=read(folder/'metadata.json');jac=derive(folder)
  if s['material']!=material or not np.allclose(s['parameter_vector'],vector,rtol=0,atol=1e-14):raise ValueError('Parameter/material mismatch')
  if not np.array_equal(np.load(folder/'mask.npy'),np.load(mask)):raise ValueError('Changed Miller basis')
  for key,value in [('grid_sizes',settings['grid']),('cutoff_energy',settings['cutoff_Ha']),('k_grid_sizes',settings['k'])]:
   if m['config'][key]!=value:raise ValueError('Changed '+key)
  if m['config']['d3ft'].get('xc_grid_factor',1)!=settings['xc_grid_factor']:raise ValueError('Changed XC grid')
  if m['pseudopotentials']!=read(Path(c['baselines'][material])/'metadata.json')['pseudopotentials']:raise ValueError('Changed PP')
  dump(folder/'equilibrium_jacobian.json',jac)
  return dict(folder=str(folder),summary=s,jacobian=dict(a0_A=(np.asarray(jac['dlogV_equilibrium_dp'])*s['a0_A']/3).tolist(),B_GPa=jac['dB_equilibrium_dp']))
 def evaluate(material,vector,current,gpu,logical,reuse=None):
  nonlocal spent
  cfg=c['settings'][material];mask=Path(c['baselines'][material])/'mask.npy';key=(material,tuple(float(x) for x in vector))
  if key in cache:return cache[key]
  tag=reserve_name(T,logical+'_'+material,output_parent=T/'runs',log_parent=T);tmp='/dev/shm/'+tag
  estimate=c['estimated_worker_seconds_by_material'][material];timeout=max(c['worker_timeout_seconds'],int(np.ceil(2.5*estimate/3600)*3600))
  if shared:shared.reserve(tag,timeout/3600)
  if c.get('dynamic_gpu_pool'):
   acquired=False
   while not acquired:
    for candidate_gpu in groups:
     if claim([candidate_gpu]):gpu=candidate_gpu;acquired=True;break
    if not acquired:
     dump(T/(material+'_resource_wait.json'),dict(time=now(),stage='waiting_shared_GPU_pool',fit=c.get('selected_fit'),material=material))
     time.sleep(10)
  base=c['environment'];env={**os.environ,**base,'PYTHONPATH':c['runtime']+':'+c['jrystal'],'D3FT_PHASE3_RESULTS_ROOT':str(T/'runs'),'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax','PYTHONPYCACHEPREFIX':tmp+'/pycache','D3FT_TRACE_STAGES':'1','D3FT_CHECKPOINT_ROOT':str(T/'runs'/tag/'checkpoints'),'D3FT_CHECKPOINT_INTERVAL':'500','D3FT_YIELD_REQUEST':str(T/'yield_to_E5.json')}
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E7-fit','env','CUDA_VISIBLE_DEVICES='+str(gpu),'timeout',str(timeout),c['python'],'-u',str(R/'validation/workflows/phase3/worker.py'),'--material',material,'--vector',json.dumps(np.asarray(vector).tolist()),'--tag',tag,'--mode','evaluate','--bulk-only','--grid',str(cfg['grid']),'--cutoff',str(cfg['cutoff_Ha']),'--k',str(cfg['k']),'--epochs','3000','--empty','0','--strain-step',str(cfg['strain_step']),'--xc-grid-factor',str(cfg['xc_grid_factor']),'--a0',str(current['summary']['a0_A']),'--fixed-mask',str(mask),'--fused-report']
  if reuse is not None:cmd+=['--reuse-from',str(reuse)]
  log=(T/(tag+'.log')).open('xb');start=time.monotonic();p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running.append(p);rec=dict(time=now(),pid=p.pid,gpus=[gpu],jobid=250285,command=cmd,output=str(T/'runs'/tag),status='running',timeout_seconds=timeout,estimated_seconds=estimate,timeout_multiplier=timeout/estimate,checkpoint_interval=500);dump(T/(tag+'_launch.json'),rec);
  while p.poll() is None:
   allowed()  # Observe E5 readiness; worker exits only after a saved POINT.
   dump(T/(material+'_worker_status.json'),dict(time=now(),stage='draining_current_DFT_point_for_E5' if (T/'yield_to_E5.json').exists() else 'DFT_active',gpu=gpu,pid=p.pid,output=str(T/'runs'/tag)))
   time.sleep(10)
  code=p.returncode;log.close();hours=(time.monotonic()-start)/3600
  # Only main thread changes spent after a wave; record each child independently.
  rec.update(end=now(),returncode=code,GPU_hours=hours,status='completed' if code==0 else 'failed');dump(T/(tag+'_launch.json'),rec)
  release([gpu])
  if shared:shared.finish(tag,hours,'completed' if code==0 else 'cooperative_paused' if code==75 else 'failed')
  if code==75:
   if not (T/'runs'/tag/'cooperative_pause.json').exists():raise InfrastructureInterruption('exit75 without durable point certificate')
   rec.update(status='cooperative_paused',classification='E5_priority_not_physical_failure');dump(T/(tag+'_launch.json'),rec)
   while not c.get('dynamic_gpu_pool') and not claim([gpu]):
    dump(T/(material+'_resource_wait.json'),dict(time=now(),stage='waiting_E5_after_saved_DFT_point',gpu=gpu,reuse_from=str(T/'runs'/tag)))
    time.sleep(20)
   return evaluate(material,vector,current,gpu,logical+'_resume',reuse=T/'runs'/tag)
  if code==124:raise InfrastructureInterruption('DFT worker timeout124, not a physical rejection '+tag)
  if code:raise RuntimeError('DFT candidate failed rc='+str(code)+' '+tag)
  if ('DONE '+tag) not in (T/(tag+'.log')).read_text():raise ValueError('Missing worker DONE')
  answer=validate(material,T/'runs'/tag,vector,cfg,mask);cache[key]=answer;dump(T/(tag+'_validated_cache.json'),dict(material=material,vector=np.asarray(vector).tolist(),state=answer));note('E7 '+str(c.get('selected_fit','full+LOO'))+' '+logical+' '+material+'真实DFT子步骤完成，已通过原数值门并保存Jacobian；封存材料未使用。');return answer
 def batch(materials,vector,current,logical):
  nonlocal spent
  fresh=[m for m in materials if (m,tuple(float(x) for x in vector)) not in cache]
  needed=[groups[i] for i,m in enumerate(materials) if m in fresh]
  if needed and not c.get('dynamic_gpu_pool'):
   while not claim(needed):state('waiting_E5_or_available_training_cards',fit_wave=logical);time.sleep(20)
  reserve=sum(max(c['worker_timeout_seconds'],int(np.ceil(2.5*c['estimated_worker_seconds_by_material'][m]/3600)*3600))/3600 for m in fresh)
  if pilot.spent()+spent+reserve>400:raise RuntimeError('Cannot reserve E7 wave within400GPUh')
  results={};errors=[];before={p.name for p in T.glob('*_launch.json')}
  with ThreadPoolExecutor(max_workers=3) as pool:
   futures={m:pool.submit(evaluate,m,vector,current[m],groups[i],logical) for i,m in enumerate(materials)}
   for m,f in futures.items():
    try:results[m]=f.result()
    except Exception as exc:errors.append(dict(material=m,error=repr(exc),infrastructure=isinstance(exc,InfrastructureInterruption)))
  spent+=sum(read(p).get('GPU_hours',0.) for p in T.glob('*_launch.json') if p.name not in before);dump(T/'budget.json',dict(training_GPU_hours=spent,pilot_GPU_hours=pilot.spent(),budget=400))
  if errors:
   dump(T/(logical+'_failed_wave.json'),dict(errors=errors))
   if any(e['infrastructure'] for e in errors):raise InfrastructureInterruption('Timeout wave interrupted; not a rejected physical candidate')
   raise RuntimeError('DFT wave failed; failures retained, no linear acceptance')
  return results
 def fit(name,materials,baseline):
  current={m:baseline[m] for m in materials};v=np.zeros(7);damping=1.;radius=.2;history=[];accepted=0
  for trial in range(3):
   r,J,rows=dataset(materials,current,c['references']);candidate,prop=proposal(v,r,J,damping,radius);prop.update(training_materials=materials,training_rows=rows,vector_before=v.tolist(),candidate_vector=candidate.tolist(),sealed_external_holdouts=['SiC','GaP']);dump(T/(name+'_trial'+str(trial)+'_proposal.json'),prop)
   if name=='full' and trial==0 and c.get('adopted_vector') is not None:
    if not np.allclose(candidate,c['adopted_vector'],rtol=0,atol=1e-14):raise ValueError('Adopted first-wave proposal differs')
    candidate=np.asarray(c['adopted_vector'])
   if np.linalg.norm((candidate[AXES]-v[AXES])/SCALE)<1e-5:break
   state('fit_candidate_DFT',fit=name,trial=trial,training_materials=materials)
   try:
    calculated=batch(materials,candidate,current,name+'_trial'+str(trial));rr,_,_=dataset(materials,calculated,c['references']);decision=accept(float(r@r),float(rr@rr),prop['predicted_loss'])
   except InfrastructureInterruption:raise
   except RuntimeError as exc:
    if 'reserve' in str(exc):raise
    decision=dict(accepted=False,reason=str(exc));calculated=None
   history.append(dict(trial=trial,proposal=prop,acceptance=decision));dump(T/(name+'_history.json'),history)
   if decision['accepted']:current=calculated;v=candidate;accepted+=1;damping=max(1e-6,damping/2);radius=min(.4,radius*1.5)
   else:damping*=4;radius/=2
  result=dict(completed=True,training_materials=materials,parameter_names=['x:kappa','x:mu','theta_p'],vector=v.tolist(),accepted_updates=accepted,trials=len(history),max_updates=3,max_attempts=3,states={m:r['folder'] for m,r in current.items()},history=history,unique_parameter_identifiability=prop['rank']==3,validation_used_for_updates=False,external_holdouts_accessed=False,converged_to_optimum_claim=False,nonzero_parameter_fine_validation='not certified by baseline gates; separate pending numerical validation')
  dump(T/(name+'_result.json'),result);note('E7 '+name+'训练停止：'+str(accepted)+'接受更新/'+str(len(history))+'尝试；仅本折训练loss接受，尚不称泛化或参数细化PASS。');return result
 try:
  (T/'runs').mkdir()
  for seed in c.get('seed_validated_states',[]):
   value=validate(seed['material'],seed['folder'],np.asarray(seed['vector']),c['settings'][seed['material']],Path(c['baselines'][seed['material']])/'mask.npy')
   cache[(seed['material'],tuple(seed['vector']))]=value
  # Adopt the already-running first wave without signalling or restarting any
  # GPU child. Preserve real end times and release its cards before E5 recovery.
  pending=list(c.get('adopted_running_wave',[]))
  if shared:
   for rec in pending:shared.reserve('adopted_'+Path(rec['output']).name,c['worker_timeout_seconds']/3600)
  while pending:
   state('adopted_wave_draining_for_E5',active_pids=[r['pid'] for r in pending])
   for rec in list(pending):
    if alive(rec['pid']):continue
    folder=Path(rec['output']);logtext=Path(rec['log']).read_text();tag=folder.name
    start=dt.datetime.fromisoformat(rec['time']);start=start if start.tzinfo else start.astimezone()
    hours=max(0,(dt.datetime.now().astimezone()-start).total_seconds()/3600)
    complete=(folder/'summary.json').exists() and ('DONE '+tag) in logtext
    result=dict(rec,end=now(),GPU_hours=hours,adopted_existing_GPU_child=True,
       status='completed' if complete else 'interrupted',returncode=0 if complete else 124 if 'exit code 124' in logtext.lower() else None,
       classification='completed' if complete else 'infrastructure_or_unresolved_interruption',physical_failure_claim_allowed=False)
    dump(T/('adopted_'+tag+'_launch.json'),result);spent+=hours
    if shared:shared.finish('adopted_'+tag,hours,'completed' if complete else 'unresolved_infrastructure')
    if complete:
     value=validate(rec['material'],folder,np.asarray(rec.get('vector',c['adopted_vector'])),c['settings'][rec['material']],Path(c['baselines'][rec['material']])/'mask.npy')
     cache[(rec['material'],tuple(rec.get('vector',c['adopted_vector'])))]=value
    with Q.with_name('gpu_reservations.lock').open('a') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
     for g in rec['gpus']:
      if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
     dump(Q,owners)
    pending.remove(rec)
   dump(T/'budget.json',dict(training_GPU_hours=spent,pilot_GPU_hours=pilot.spent(),budget=400))
   if pending:time.sleep(20)
  state('waiting_training_certificates_and_opportunistic_cards')
  while not c.get('dynamic_gpu_pool') and not claim():state('waiting_training_certificates_and_opportunistic_cards');time.sleep(30)
  release(groups)
  baseline={}
  for m in CORE:
   baseline[m]=validate(m,c['baselines'][m],np.zeros(7),c['settings'][m],Path(c['baselines'][m])/'mask.npy');cache[(m,tuple(np.zeros(7)))]=baseline[m]
  selected=c.get('selected_fit')
  if selected in (None,'full'):allfits['full']=fit('full',CORE,baseline)
  for held in CORE:
   name='LOO_'+held
   if selected is not None and selected!=name:continue
   train=[m for m in CORE if m!=held];result=fit(name,train,baseline)
   # Freeze this fold's vector before independently reading its held-out material.
   vector=np.asarray(result['vector']);evaluated=batch([held],vector,{held:baseline[held]},name+'_validation');rr,_,rows=dataset([held],evaluated,c['references']);result['validation']=dict(material=held,vector_frozen_before_evaluation=True,rows=rows,loss=float(rr@rr),state=evaluated[held]['folder'],used_for_model_selection=False);dump(T/(name+'_result.json'),result);allfits[name]=result
  dump(T/'result.json',dict(completed=True,full_fit_and_three_LOO=allfits,training_GPU_hours=spent,total_E7_GPU_hours=pilot.spent()+spent,sealed_holdouts=['SiC','GaP'],external_generalization_certified=False,nonzero_parameter_numerical_refinement_complete=False));state('selected_fit_complete' if selected else 'three_material_full_fit_and_LOO_complete',selected_fit=selected,accepted_updates={k:v['accepted_updates'] for k,v in allfits.items()});note('E7 '+str(selected or 'full+3LOO')+' 已完成并保存真实DFT结果；SiC/GaP仍封存，非零参数加密复核未冒充PASS。')
 except InfrastructureInterruption as exc:state('infrastructure_interrupted',classification='timeout_not_physical_failure',error=repr(exc),partial_fits=list(allfits));note('E7基础设施超时中断，不记物理候选失败/不消耗一次拒绝更新，原记录保留：'+repr(exc));raise
 except Exception as exc:state('failed_or_budget_limited',error=repr(exc),partial_fits=list(allfits));note('E7全拟合/三折队列停止：'+repr(exc)+'；已有结果和失败保留，不读取封存留出。');raise
 finally:
  for child in running:
   if child.poll() is None:child.wait()
  with Q.with_name('gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   for g in groups:
    if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
   dump(Q,owners)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
