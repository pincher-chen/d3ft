"""Automatic E10 path queue on hn35 after both eight-card SCFs complete."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import *

def main(B):
 B=B.resolve();c=read(B/'protocol.json');P=Path(c['project']);W=P/'validation/workflows/response_design/e9_e10';states=read(B/'SCF_states.json');path=read(c['E10']['path_source']);count=len(path['k_fractional']);active={};done={};records=[];spent=0.;failures=[];todo=[(name,[i for i in range(count) if i%8==g],g,None) for name in states for g in range(8)]
 smoke=read(B/'path_smoke_acceptance.json');assert smoke['passed'] and smoke['state']=='E4_initial'
 todo=[(name,indices,g,smoke['output'] if name=='E4_initial' and 0 in indices else resume) for name,indices,g,resume in todo]
 recovery=read(B/'path_recovery_current.json');prior=read(Path(recovery['root'])/'original_failed_attempts.json');prior_hours=prior['total_failed_GPU_hours'];smoke_hours=smoke['GPU_hours'];scf_hours=read(B/'GPU_budget.json')['total_GPU_hours']
 def budget():
  live=sum((time.monotonic()-j['start'])/3600 for j in active.values())
  value=dict(cap=None,record_only=True,completed_GPU_hours=spent,active_GPU_hours=live,prior_failed_GPU_hours=prior_hours,smoke_GPU_hours=smoke_hours,SCF_GPU_hours=scf_hours,total_E10_GPU_hours=scf_hours+prior_hours+smoke_hours+spent+live,attempts=records,active_attempts=[j['record'] for j in active.values()],accounting='GPU wall hours; seven original startup failures estimated from final log timestamps')
  dump(B/'path_budget.json',value);dump(B/'E10_GPU_accounting.json',value)
 def status(stage,**kw):
  dump(B/'path_status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,node='hn35',completed_batches=len(done),total_batches=16,completed_path_GPU_hours=spent,active={str(g):dict(name=j['name'],pid=j['proc'].pid,output=str(j['out'])) for g,j in active.items()},**kw))
  top=read(B/'status.json');top.update(time=now(),stage='E10_'+stage,path_controller_pid=os.getpid(),E9='paused');dump(B/'status.json',top)
  budget()
 def note(t):
  for p in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
   with p.open('a') as f:f.write('\n- '+now()+' '+t+'\n')
 while todo or active:
  for g,j in list(active.items()):
   if j['proc'].poll() is None:continue
   j['log'].close();r=j['record'];r.update(returncode=j['proc'].returncode,end=now(),GPU_hours=(time.monotonic()-j['start'])/3600);spent+=r['GPU_hours'];dump(j['root']/'launch.json',r);records.append(r);active.pop(g)
   if r['returncode']==124:
    todo.insert(0,(j['name'],j['indices'],g,str(j['out'])));note('E10路径batch exit124，保留已完成k点和每100迭代谱子空间，原门新唯一名自动续算。')
   elif r['returncode']!=0:
    failures.append(r);note('E10路径非timeout子进程失败，现场保留，停止派发新batch；已在跑的子进程自然完成并逐项记账。')
   else:done[(j['name'],g)]=str(j['out'])
  if failures and not active:
   status('failure_retained_all_children_reaped',failed_records=failures);raise RuntimeError('E10 path failure; all child exits recorded')
  for entry in ([] if failures else list(todo)):
   name,indices,g,resume=entry
   if g in active:continue
   run=unique(B/'path_runs',name+'_GPU'+str(g));out=run/'output';runtime=Path(c['E5_protocol']['runtime']);tmp='/dev/shm/'+run.name;env={**os.environ,**c['E5_protocol']['environment']};env.update(PYTHONPATH=':'.join(map(str,[W,runtime,runtime/'validation/workflows/response_design/reverse_response',runtime/'validation/workflows/response_design',runtime/'validation/workflows/phase3',Path(c['jrystal'])])),TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax')
   args=['--source',states[name],'--protocol',str(B/'protocol.json'),'--indices',__import__('json').dumps(indices),'--output',str(out)]
   if resume:args+=['--resume-source',resume]
   cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E10-path','env','CUDA_VISIBLE_DEVICES='+str(g),'timeout','--signal=TERM','--kill-after=30s','43200',c['python'],'-u',str(W/'e10_path.py'),*args]
   log=(run/'stdout.log').open('xb');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record=dict(time=now(),pid=proc.pid,node='hn35',jobid=250285,gpus=[g],name=name,indices=indices,output=str(out),command=cmd,timeout_seconds=43200);dump(run/'launch.json',record);active[g]=dict(proc=proc,log=log,name=name,indices=indices,out=out,root=run,start=time.monotonic(),record=record);todo.remove(entry)
  status('path_spectra_running',pending_batches=len(todo),accepted_batches=done if False else [{'state':k[0],'gpu':k[1],'output':v} for k,v in done.items()]);budget();time.sleep(15)
 import numpy as np
 result={}
 for name in states:
  points={}
  for (n,g),folder in done.items():
   if n==name:
    for f in Path(folder).glob('k_*.json'):
     if f.stem.endswith('_solver'):continue
     q=read(f);assert q['passed'] and len(q['eigenvalues_Ha'])==56 and max(q['residuals_Ha'])<=1e-7;points[q['index']]=q
  assert set(points)==set(range(count));vals=np.array([points[i]['eigenvalues_Ha'] for i in range(count)]);gamma=next(i for i,k in enumerate(path['k_fractional']) if np.max(abs(np.asarray(k)))<1e-12)
  result[name]=dict(passed=True,points=count,bands=56,path_gap_eV=float((vals[:,48].min()-vals[:,47].max())*27.211386245988),Gamma_transition_eV=float((vals[gamma,48]-vals[gamma,47])*27.211386245988),maximum_ritz_Ha=max(max(q['residuals_Ha']) for q in points.values()),source=states[name],path_sha256=sha(c['E10']['path_source']),alignment='own sampled path VBM',not_continuous_BZ_convergence=True)
  dump(B/(name+'_path_result.json'),result[name]);np.savez_compressed(B/(name+'_path_bands.npz'),eigenvalues_Ha=vals,k_fractional=path['k_fractional'])
 dump(B/'E10_acceptance.json',dict(completed=True,passed=True,states=result,path_GPU_hours=spent,not_ionic_relaxation=True,smoke_reused=True));status('complete',result=result);note('E10 E4两态4³密度/原parent-reference路径56带完成，路径gap与Γ跃迁已保存；结果待图文审阅。E9尚未启动，不冒称已自动接续。')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
