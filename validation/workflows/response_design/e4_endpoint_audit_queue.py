"""One free card endpoint audit; keep E6/GPU0 and diamond/GPU2/5 untouched."""
import argparse,datetime as dt,fcntl,os,subprocess,time,json
from pathlib import Path
from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from e1_three_parameter_queue import read,dump
from run_identity import reserve_name

def main(T):
 os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY));c=read(T/'protocol.json');P=Path(c['project']);Y=P/'validation/results/response_design_20261015_e7_ct_pilot';Q=Y/'gpu_reservations.json';GPU=None;proc=None;record=None
 def now():return dt.datetime.now().astimezone().isoformat()
 def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,gpus=[] if GPU is None else [GPU],**kw))
 def note(text):
  for f in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
   with f.open('a') as out:out.write('\n- '+now()+' '+text+'\n')
 def claim():
  nonlocal GPU
  r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=45)
  memory={int(v[0]):int(v[1]) for line in r.stdout.splitlines() if len(v:=line.split(','))==2};dump(T/'GPU_probe.json',dict(time=now(),memory_MiB=memory))
  with (Y/'gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   free={g for g in c['gpu_candidates'] if memory.get(g,10**9)<=3 and str(g) not in owners}
   # Let ready two-card E1 points take intact pairs first; a residual lone idle
   # card is sufficient for this audit and never overlaps an existing child.
   idx=read(P/'validation/results/response_design_20261014_e1_alp/extension_20261015/active_three_parameter_scope.json');e1=read(Path(idx['root'])/'status.json')
   if e1.get('pending') and e1.get('stage')=='three_parameter_FD_running_or_waiting_released_cards':
    for pair in [{1,4},{3,6}]:
     if pair<=free:free-=pair
   for g in c['gpu_candidates']:
    if g in free:
     GPU=g;owners[str(g)]=dict(owner='E4 independent Gamma54 endpoint audit',controller_pid=os.getpid(),status_file=str(T/'status.json'));dump(Q,owners);return True
  return False
 try:
  if c['prior_E4_upper_GPU_hours']+c['timeout_seconds']/3600>100:raise ValueError('E4 audit budget reservation exceeds100GPUh')
  state('waiting_one_free_card_after_E1_ready_pairs')
  while not claim():state('waiting_one_free_card_after_E1_ready_pairs');time.sleep(20)
  tag=reserve_name(T,'independent_reopt_Ritz_audit',output_parent=T,log_parent=T);out=T/tag;tmp='/dev/shm/'+tag
  base=read(P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json')['environment'];env={**os.environ,**base,'PYTHONPATH':c['runtime']+':'+base['PYTHONPATH'].split(':')[-1],'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax','D3FT_TRACE_STAGES':'1'}
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E4-audit','env','CUDA_VISIBLE_DEVICES='+str(GPU),'timeout',str(c['timeout_seconds']),c['python'],'-u',str(T/'audit_worker.py'),'--protocol',str(T/'protocol.json'),'--output',str(out)]
  start=time.monotonic();log=(T/(tag+'.log')).open('xb');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record=dict(time=now(),pid=proc.pid,jobid=250285,gpus=[GPU],command=cmd,output=str(out),status='running',timeout_seconds=c['timeout_seconds']);dump(T/'GPU_launch.json',record)
  note('E4独立端点审计新进程已上250285/GPU'+str(GPU)+'，固定2000步电子重优化、52态新谱及谱重占据审计；旧终点保留，不称离子平衡。')
  while proc.poll() is None:
   worker=read(out/'status.json') if (out/'status.json').exists() else None;state('independent_endpoint_audit_running',worker=worker,pid=proc.pid,elapsed_seconds=time.monotonic()-start);time.sleep(15)
  log.close();record.update(end=now(),returncode=proc.returncode,GPU_hours=(time.monotonic()-start)/3600,status='completed' if not proc.returncode else 'failed');dump(T/'GPU_launch.json',record)
  if proc.returncode or not (out/'acceptance.json').exists():raise RuntimeError('Independent endpoint child incomplete/failed; retain raw record')
  result=read(out/'acceptance.json');dump(T/'acceptance.json',dict(result,output=str(out),audit_GPU_hours=record['GPU_hours']))
  state('complete' if result['passed'] else 'failed',passed=result['passed'],output=str(out));note('E4独立电子重优化/谱/残差/占据审计 '+('PASS' if result['passed'] else 'FAIL')+'；终点非平衡力和位移/能量保留，未自动改论文。')
 except Exception as exc:state('failed',passed=False,error=str(exc));note('E4独立端点审计停止：'+repr(exc)+'；失败保留，不放宽门。');raise
 finally:
  if proc is not None and proc.poll() is None:proc.wait()
  with (Y/'gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   if GPU is not None and owners.get(str(GPU),{}).get('status_file')==str(T/'status.json'):owners.pop(str(GPU));dump(Q,owners)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
