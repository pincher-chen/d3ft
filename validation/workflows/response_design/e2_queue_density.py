"""Bounded background full-k audit shards, with automatic collection and budget."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import datetime,json,os,queue,shutil,subprocess,threading,time,sys
P=Path(__file__).resolve().parents[3];A=P/'validation/results/response_design_20261014_e2_audit';E=P/'validation/results/response_design_20261014_e0_migration';B=P/'validation/results/response_design_20261014_e1_alp';protocol=json.loads((A/'density_audit_protocol.json').read_text());R=Path(protocol['runtime']);env0=json.loads((E/'commands/alp_new_qe_tail.json').read_text())['environment'];env0['PYTHONPATH']=str(R)+':'+env0['PYTHONPATH'].split(':',1)[1];python='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python';driver=R/'validation/workflows/response_design/e2_density_audit.py'
if not protocol.get('legacy_calibration_audits_enabled',True):
 print('Six legacy calibration states are deferred to E7; this queue is disabled.',flush=True);sys.exit(0)
if '--resume' not in sys.argv:
 fd=os.open(A/'density_queue.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
shutil.copyfile(P/'validation/workflows/response_design/e2_density_audit.py',driver)
inventory=json.loads((A/'inventory.json').read_text());rows=[r for r in inventory['states'] if not r['passed'] and not r.get('excluded_from_E2',False)];tasks=queue.Queue();remaining={};failed={};sources={};lock=threading.Lock();spent=5.0;reserved=0.;threads=[];started=set();completed=0;planned=0
for row in rows:
 name=row['label'];src=Path(row['source']);Nk=int(json.loads((src/'metadata.json').read_text())['config']['k_grid_sizes'])**3;sources[name]=src;remaining[name]=0;failed[name]=False;(A/'density_full'/name/'shards').mkdir(parents=True,exist_ok=True)
 for start in range(0,Nk,protocol['k_block_size']):tasks.put((name,start,min(protocol['k_block_size'],Nk-start)));remaining[name]+=1;planned+=1

def status(**extra):
 d=dict(time=datetime.datetime.now().isoformat(),stage='running',planned_shards=planned,completed_shards=completed,pending_shards=tasks.qsize(),conservative_E2_GPU_hours=spent,reserved_upper_bound_GPU_hours=reserved,gpus=sorted(started));d.update(extra);tmp=A/'density_queue_status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(A/'density_queue_status.json')

def invoke(gpu,name,start,count,merge=False):
 out=A/'density_full'/name/('final' if merge else f'shards/{start:04d}');log=out.parent/(out.name+'.log');tmp=f'/dev/shm/response_design_e2_density_{name}_{start}_{gpu}';env=env0.copy();env.update(TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',PYTHONPYCACHEPREFIX=tmp+'/pycache');cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e2','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout',str(protocol['per_shard_timeout_seconds']),python,'-u',str(driver),'--source',str(sources[name]),'--output',str(out)]
 if merge:cmd+=['--merge-from',str(A/'density_full'/name/'shards')]
 else:cmd+=['--start',str(start),'--limit',str(count)]
 rp=out.parent/(out.name+'_launch.json')
 if rp.exists():
  record=json.loads(rp.read_text());pid=record.get('pid')
  print('E2_ADOPT_EXISTING',name,start,'PID',pid,flush=True)
  while pid:
   stat=Path(f'/proc/{pid}/stat')
   if not stat.exists() or stat.read_text().split()[2]=='Z':break
   time.sleep(10)
  complete=(out/'audit.json').exists() if merge else (out/'partial.json').exists()
  record.update(end=datetime.datetime.now().isoformat(),returncode=0 if complete else 1,GPU_hours=(datetime.datetime.now()-datetime.datetime.fromisoformat(record['start'])).total_seconds()/3600,status='adopted_finished' if complete else 'adopted_failed')
  rp.write_text(json.dumps(record,indent=2)+'\n');return record,out
 if merge:cmd[cmd.index('timeout')+1]='1800'
 record=dict(start=datetime.datetime.now().isoformat(),jobid=250285,gpu=gpu,command=cmd,environment=env,status='running');tick=time.monotonic()
 with log.open('xb') as f:
  proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=proc.pid;rp.write_text(json.dumps(record,indent=2)+'\n');code=proc.wait()
 record.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=(time.monotonic()-tick)/3600,status='finished' if code==0 else 'failed');rp.write_text(json.dumps(record,indent=2)+'\n');return record,out

def worker(gpu):
 global spent,reserved,completed
 cap=protocol['per_shard_timeout_seconds']/3600
 while True:
  with lock:
   if tasks.empty():return
   if spent+reserved+cap+0.5>protocol['hard_budget_total_E2_GPU_hours']:
    if reserved==0:status(stage='budget_stopped');return
    wait=True
   else:
    wait=False;task=tasks.get();reserved+=cap;status()
  if wait:time.sleep(15);continue
  name,start,count=task;print('E2_SHARD_START',name,start,count,'GPU',gpu,flush=True);rec,out=invoke(gpu,name,start,count)
  with lock:
   spent+=rec['GPU_hours'];reserved-=cap;completed+=1;remaining[name]-=1;failed[name]|=rec['returncode']!=0 or not (out/'partial.json').exists();merge=remaining[name]==0 and not failed[name];status()
  print('E2_SHARD_END',name,start,'returncode',rec['returncode'],'GPUh',rec['GPU_hours'],flush=True)
  if merge:
   with lock:
    can_merge=spent+reserved+0.5<=protocol['hard_budget_total_E2_GPU_hours']
    if can_merge:reserved+=0.5
   if not can_merge:
    print('E2_MERGE_BUDGET_STOP',name,flush=True);tasks.task_done();continue
   rec,folder=invoke(gpu,name,0,0,merge=True)
   with lock:spent+=rec['GPU_hours'];reserved-=0.5;status()
   f=folder/'audit.json';passed=rec['returncode']==0 and f.exists() and json.loads(f.read_text())['acceptance']['passed'];print('E2_CASE',name,'PASS' if passed else 'FAIL',flush=True)
   with (P/'response_design_to_do_list.md').open('a') as h:h.write(f'\n- {datetime.datetime.now().isoformat()} E2密度审计 {name}: '+('PASS' if passed else 'FAIL')+f'；新完整k低谱/重占据证据 {f.relative_to(P)}，原优化器轨道未存的限制保留。\n')
  tasks.task_done()

def start_lane(gpu):
 with lock:started.add(gpu);status()
 t=threading.Thread(target=worker,args=(gpu,),daemon=True);threads.append(t);t.start()
for gpu in protocol['gpus']:start_lane(gpu)
extra_started=False
while any(t.is_alive() for t in threads):
 # Only expand after the E1 controller declares completion, and query confirms
 # all four E1 devices are free. hn48 is never used.
 st=B/'status.json'
 if not extra_started and st.exists() and not tasks.empty():
  try:
   state=json.loads(st.read_text())
   if state['stage']=='K4_K10_computations_complete':
    result=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
    values={int(line.split(',')[0]):[int(x.strip()) for x in line.split(',')[1:]] for line in result.stdout.splitlines() if len(line.split(','))==3}
    if all(values.get(g,[99999,100])[0]<=3 and values.get(g,[99999,100])[1]==0 for g in [2,3,4,6]):
     for gpu in [2,3,4,6]:start_lane(gpu)
     extra_started=True;print('E2_AUTO_EXPAND after E1 GPU release: 8 audit lanes',flush=True)
  except Exception as exc:print('E2_RELEASE_QUERY',str(exc),flush=True)
 time.sleep(20)
with lock:status(stage='queue_finished',unsubmitted_shards=tasks.qsize())
inv=json.loads((A/'inventory.json').read_text())
for row in inv['states']:
 f=A/'density_full'/row['label']/'final/audit.json'
 if f.exists():row.update(status='完成密度/自由能固定点审计' if json.loads(f.read_text())['acceptance']['passed'] else '失败',passed=json.loads(f.read_text())['acceptance']['passed'],audit=str(f),original_optimizer_commutator_available=False)
(A/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
print('E2 density queue finished; original optimizer-commutator limitations retained; physical acceptance required.',flush=True)
