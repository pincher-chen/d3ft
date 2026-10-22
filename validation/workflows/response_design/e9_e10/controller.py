"""Detached hn35-only controller: E10 SCF first, checkpoint124 recovery, then spectra/E9."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import *

def main(B):
 B=B.resolve();c=read(B/'protocol.json');P=Path(c['project']);W=P/'validation/workflows/response_design/e9_e10';records=[];total=0.;start=time.monotonic()
 def status(stage,**kw):dump(B/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,node='hn35',completed_GPU_hours=total,launches=records,**kw))
 def note(text):
  for f in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
   with f.open('a') as o:o.write('\n- '+now()+' '+text+'\n')
 def free(gpus):
  for attempt in range(3):
   try:
    q=subprocess.run(['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=60)
    if q.returncode==0:
     mem={int(v[0]):int(v[1]) for line in q.stdout.splitlines() if len(v:=line.split(','))==2};dump(B/'GPU_probe.json',dict(time=now(),memory_MiB=mem));return all(mem.get(g,10**9)<=3 for g in gpus)
   except subprocess.TimeoutExpired:pass
   time.sleep(60)
  return False
 def run_scf(name):
  nonlocal total
  history=B/'SCF'/name;history.mkdir(parents=True,exist_ok=True);resume=None
  done=history/'accepted.json'
  if done.exists():return read(done)['output']
  for attempt in range(10):
   while not free(list(range(8))):status('waiting_full8_for_'+name);time.sleep(60)
   run=unique(history,'SCF');out=run/'output';runtime=Path(c['E5_protocol']['runtime']);tmp='/dev/shm/'+run.name;env={**os.environ,**c['E5_protocol']['environment']};env.update(PYTHONPATH=':'.join(map(str,[runtime,runtime/'validation/workflows/response_design/reverse_response',runtime/'validation/workflows/response_design',runtime/'validation/workflows/phase3',Path(c['jrystal'])])),TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',D3FT_TRACE_STAGES='1',PYTHONUNBUFFERED='1')
   completed=0 if resume is None else int(read(resume/'latest.json')['iteration']);epochs=3000-completed
   args=['--reference',str(B/'references'/name),'--parent',str(c['E5_parent']),'--devices','8','--epochs',str(max(1,epochs)),'--output',str(out)]
   if resume is not None:
    if epochs==0:args=['--reference',str(B/'references'/name),'--parent',str(resume.parent.parent),'--devices','8','--epochs','1','--read-only','--output',str(out)]
    else:args+=['--resume-checkpoint',str(resume)]
   cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c32','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E10','env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','timeout','--signal=TERM','--kill-after=30s','43200',c['python'],'-u',str(W/'e10_scf.py'),*args]
   with (run/'stdout.log').open('xb') as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started=time.monotonic();rec=dict(time=now(),pid=proc.pid,jobid=250285,node='hn35',gpus=list(range(8)),mode='E10_SCF',state=name,output=str(out),command=cmd,timeout_seconds=43200,resume_checkpoint=None if resume is None else str(resume));records.append(rec);dump(run/'launch.json',rec);note('E10 '+name+'八卡4³ SCF已提交hn35/250285，srunPID'+str(proc.pid)+'；12h/500步检查点，原模型/门不变。')
    lastnote=time.monotonic()
    while proc.poll() is None:
     cp=latest_checkpoint(run,3000);status('E10_'+name+'_SCF_running',active_pid=proc.pid,active_GPU_hours=8*(time.monotonic()-started)/3600,checkpoint_iteration=None if cp is None else cp[0],current_run=str(run))
     if time.monotonic()-lastnote>=900:
      with (B/'progress.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(time=now(),stage=name,checkpoint_iteration=None if cp is None else cp[0],GPU_hours=total+8*(time.monotonic()-started)/3600))+'\n')
      lastnote=time.monotonic()
     time.sleep(20)
   hours=8*(time.monotonic()-started)/3600;total+=hours;rec.update(returncode=proc.returncode,GPU_hours=hours,end=now());dump(run/'launch.json',rec);dump(B/'GPU_budget.json',dict(cap=None,record_only=True,total_GPU_hours=total,attempts=records))
   if proc.returncode==0:
    pt=read(out/'point.json')
    if not pt['physical_passed']:raise ValueError('Physical checks failed; no recovery/relaxation of tolerances')
    dump(done,dict(time=now(),passed=True,output=str(out),point_sha256=sha(out/'point.json')));note('E10 '+name+' SCF完成，原电荷/驻点/ASR门通过；开始/等待路径谱。');return str(out)
   if proc.returncode!=124:raise RuntimeError('Non-timeout E10 failure, retained '+str(run))
   cp=latest_checkpoint(history,3000)
   if cp is None:raise RuntimeError('Exit124 without validated checkpoint; manual diagnosis required')
   resume=cp[1];note('E10 '+name+' exit124基础设施暂停，保留attempt并自动从'+str(cp[0])+'步检查点续算；不算物理失败。')
  raise RuntimeError('Infrastructure recovery attempts exhausted')
 try:
  status('starting_pre_registered_E10_SCF')
  states={name:run_scf(name) for name in c['E10']['states']};dump(B/'SCF_states.json',states);status('E10_SCF_pair_complete_waiting_CPU_prepared_followup')
  # New scripts are CPU-prepared while SCF runs; the ready sentinel is atomic.
  while not (B/'followup_ready.json').exists():status('E10_SCF_complete_waiting_followup_ready');time.sleep(30)
  cmd=[sys.executable,str(W/'followup.py'),'--root',str(B)];subprocess.run(cmd,check=True);status('complete')
 except Exception as exc:status('failed',error=repr(exc));note('E9/E10控制器停止：'+repr(exc)+'；现场保留、物理失败不自动重试。');raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
