"""CPU-only E9 handover after AlP, one GPU, immutable numerical protocol."""
import argparse,os,subprocess,sys,time,fcntl
from pathlib import Path
from common import read,dump,now,sha,unique

def _run(root):
 root=root.resolve();c=read(root/'registered_protocol.json');control=read(root/'resume_authorization.json');B=Path(control['B_root']);records=read(root/'GPU_budget.json')['attempts'];p=read(root/'backend_protocol.json')
 def status(stage,**kw):
  used=sum(a['GPU_hours'] if a.get('GPU_hours') is not None else (time.time()-a['start_epoch'])/3600 for a in records)
  dump(root/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,node='hn35',GPU_hours=used,**kw))
 def note(text):
  for f in [Path(c['project'])/'paper/response_design_EXPERIMENTS_20261014.md',Path(c['project'])/'response_design_to_do_list.md']:
   with f.open('a') as stream:stream.write('\n- '+now()+' '+text+'\n')
 def parents_done():
  try:
   b=read(B/'budget.json');return all('AlP_'+m in b['logical_results'] for m in ['fitted','PBE']) and not any(a.get('GPU_hours') is None for a in b['attempts'])
  except (OSError,KeyError,ValueError):return False
 while not parents_done():status('waiting_AlP_pair_complete_and_accepted_no_E9_GPU_submitted');time.sleep(30)
 while True:
  try:
   probe=subprocess.run(['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45)
   memory={int(a):int(b) for a,b in (line.split(',') for line in probe.stdout.splitlines())} if probe.returncode==0 else {}
   if memory.get(0,10**9)>64:status('waiting_GPU0_natural_release');time.sleep(60);continue
  except (OSError,ValueError,subprocess.TimeoutExpired):status('GPU_probe_unknown_waiting');time.sleep(60);continue
  for path,h in read(root/'execution_SHA.json').items():
   if sha(path)!=h:raise ValueError('Frozen E9 execution source changed: '+path)
  attempt=unique(root/'attempts','E9');tmp='/dev/shm/'+attempt.name;runtime=Path(p['runtime']);execution=Path(__file__).parent
  env={**os.environ,**c['E5_protocol']['environment']};env.update(PYTHONPATH=':'.join(map(str,[execution,runtime,runtime/'validation/workflows/response_design',runtime/'validation/workflows/response_design/reverse_response',runtime/'validation/workflows/phase3',Path(c['jrystal'])])),TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',D3FT_TRACE_STAGES='1',PYTHONUNBUFFERED='1')
  for key in ['D3FT_YIELD_REQUEST','D3FT_EVALUATION_FREEZE_CERT','D3FT_RESUME_SOURCE','D3FT_RESUME_REPORT','D3FT_CHECKPOINT_ROOT','D3FT_CHECKPOINT_INTERVAL']:env.pop(key,None)
  command=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E9','env','CUDA_VISIBLE_DEVICES=0','timeout','--signal=TERM','--kill-after=30s','43200',c['python'],'-u',str(execution/'e9_worker.py'),'--root',str(root)]
  record=dict(time=now(),start_epoch=time.time(),GPU_hours=None,node='hn35',jobid=250285,gpus=[0],command=command)
  note('E9按10/19预注册提交时间='+record['time']+'；AlP两套验收、自然释放后hn35/250285 GPU0，12h/500步，初始内弛豫→冻结目标→受力约束设计；首步实测后自动报告。')
  dump(attempt/'launch_intent.json',record)
  with (attempt/'stdout.log').open('xb') as log:
   proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=proc.pid;records.append(record);dump(attempt/'launch.json',record);dump(root/'GPU_budget.json',dict(attempts=records))
   while proc.poll() is None:status('E9_running',active_pid=proc.pid,current_attempt=str(attempt));time.sleep(20)
  record.update(returncode=proc.returncode,end=now(),GPU_hours=(time.time()-record['start_epoch'])/3600);dump(attempt/'launch.json',record);dump(root/'GPU_budget.json',dict(attempts=records))
  if proc.returncode==124:
   note('E9实际exit124基础设施暂停，完整电子/离子/目标轨迹原样保留；新唯一attempt只恢复最近完整检查点/已接受状态，不改总步数/门。');status('timeout_recovery_pending');continue
  if proc.returncode!=0:
   status('failed_retained_no_automatic_non124_retry',returncode=proc.returncode);note('E9非124失败保留，停止自动续算；GaAs保持优先队列后等待报告。');return
  result=read(root/'acceptance.json');status('complete',passed=result['passed']);dump(root/'resource_release.json',dict(time=now(),released=True,passed=result['passed'],E9_terminal_result=True));note('E9预注册流程结束，结果='+str(result['passed'])+'，GPUh='+str(sum(a['GPU_hours'] for a in records))+'；力/电子/谱及目标结果原样保留，资源释放后GaAs仅做首点成本探针。');return

def main(root):
 with (root/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return _run(root)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args()
 try:main(args.root)
 except Exception as exc:dump(args.root/'controller_error.json',dict(time=now(),error=repr(exc)));raise
