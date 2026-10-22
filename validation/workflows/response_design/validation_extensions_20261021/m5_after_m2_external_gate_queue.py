"""Detached eight-card queue: M2+hn35idle -> certifiedhn48gate ->64k (no repeatedprobe)."""
import datetime,json,os,subprocess,time,traceback,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parents[3];R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root']);Q=Path(json.loads((R/'M5-A/current_sharded.json').read_text())['root']);p=json.loads((Q/'protocol.json').read_text());read=lambda f:json.loads(Path(f).read_text());JOBS=[];H=Path(read(Q/'precheck_hn48_current.json')['root'])
def now():return datetime.datetime.now().astimezone().isoformat()
def dump(f,d):
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def verify_tested_sources():
 for file,h in read(Q/'CPU_acceptance.json')['source_SHA256'].items():
  if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=h:raise RuntimeError('CPU-testedsourcechanged: '+file)
def stage(name,**kw):dump(Q/'status.json',{'time':now(),'stage':name,'pid':os.getpid(),**kw});print('M5_QUEUE',name,kw,flush=True)
def ledger():
 rows=[{**j,'GPU_hours':8*(j.get('end_epoch',time.time())-j['start_epoch'])/3600} for j in JOBS];external=read(H/'GPU_ledger.json')['GPU_hours'] if (H/'GPU_ledger.json').exists() else 0.;total=p['historical_GPU_hours']+external+sum(j['GPU_hours'] for j in rows);dump(Q/'GPU_ledger.json',{'time':now(),'total_M5_GPU_hours':total,'historical_GPU_hours':p['historical_GPU_hours'],'cap':35,'external_hn48_GPU_hours':external,'external_hn48_ledger':str(H/'GPU_ledger.json'),'jobs':rows})
 effective=read(R/'GPU_ledger.json');m3=R/'M3-A/current_continuation.json'
 if m3.exists():
  c=Path(read(m3)['root'])/'GPU_ledger.json'
  if c.exists():effective['GPU_hours']['M3-A']=read(c)['total_M3_GPU_hours']
 effective['GPU_hours']['M5-A']=total;effective['time']=now();effective['M5_sharded_ledger']=str(Q/'GPU_ledger.json');dump(R/'GPU_ledger_effective.json',effective);return total

def snapshot():
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 for attempt in range(3):
  try:
   proc=subprocess.run(cmd,capture_output=True,text=True,timeout=45,check=True);rows=[[int(float(x.strip())) for x in line.split(',')] for line in proc.stdout.strip().splitlines()];assert len(rows)==8;return {'time':now(),'rows':rows,'raw':proc.stdout,'node':'hn35','jobid':250285}
  except Exception as ex:print('PROBE_RETRY',attempt,repr(ex),flush=True);time.sleep(5)
 return None

def launch(tag,script,args,timeout):
 output=Q/('run_'+tag);assert not output.exists();used=ledger()
 if used+8*(timeout+25)/3600>35:raise RuntimeError('Budget reservation exceeds35GPUh before launch')
 rt=Path(p['runtime_path']);env=os.environ.copy();env.update(p['environment']);tmp=f'/dev/shm/m5_sharded_{os.getpid()}_{tag}';env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env['PYTHONPATH']=':'.join([str(rt),str(rt/'validation/workflows/response_design/reverse_response'),str(rt/'validation/workflows/response_design'),str(rt/'validation/workflows/phase3'),p['jrystal']])
 for key in list(env):
  if key.startswith('D3FT_') and any(t in key for t in ['CHECKPOINT','RESUME','RESULTS_ROOT','YIELD','RESPONSE_SINGLE_DEVICE']):env.pop(key)
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c8','env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','bash','-c','mkdir -p "$TMPDIR"; printf "STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"','worker','timeout','--signal=TERM','--kill-after=20s',str(timeout),p['python'],str(script),*map(str,args),'--output',str(output)]
 rec={'label':tag,'start_epoch':time.time(),'gpu_count':8,'node':'hn35','jobid':250285,'output':str(output),'timeout_seconds':timeout,'command':cmd};JOBS.append(rec);(Q/'logs').mkdir(exist_ok=True)
 with (Q/'logs'/(tag+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(Q/'logs'/(tag+'_launch.json'),rec);ledger()
  while proc.poll() is None:time.sleep(10);ledger()
  rec.update(returncode=proc.returncode,end_epoch=time.time());dump(Q/'logs'/(tag+'_completion.json'),rec);ledger()
 return rec
if __name__=='__main__':
 assert read(Q/'CPU_acceptance.json')['passed'];assert hashlib.sha256((Q/'protocol.json').read_bytes()).hexdigest()==read(Q/'protocol_SHA256.json')['protocol.json']
 assert read(H/'queue_CPU_acceptance.json')['passed']
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==read(H/'queue_CPU_acceptance.json')['source_SHA256'][str(Path(__file__))]
 with (Q/'manager_started_external_hn48.json').open('x') as f:json.dump({'pid':os.getpid(),'time':now()},f)
 stage('waiting_M2_and_hn35_idle_external_hn48_gate');ledger()
 while not (R/'M2-A/summary.json').exists():time.sleep(30)
 while True:
  snap=snapshot()
  if snap and all(row[1:]==[0,0] for row in snap['rows']):break
  time.sleep(30)
 verify_tested_sources()
 dump(Q/'prelaunch_occupancy.json',snap);dump(Q/'preceding_M2_summary.json',read(R/'M2-A/summary.json'))
 try:
  rt=Path(p['runtime_path']);D=rt/'validation/workflows/response_design/reverse_response'
  while not any((H/name).exists() for name in ['certificate.json','failure.json','skipped_keep_original_plan.json']):
   stage('waiting_external_hn48_layout_result');time.sleep(10)
  if (H/'certificate.json').exists():
   cert=read(H/'certificate.json');assert cert['passed'] and cert['node']=='hn48' and cert['jobid']==275669 and cert['devices']==4 and cert['nk']==32 and cert['k_per_device']==8
   if hashlib.sha256(Path(cert['GPU_acceptance']).read_bytes()).hexdigest()!=cert['GPU_acceptance_SHA256']:raise ValueError('PassedexternalGPUrecordchanged')
   for file,digest in cert['production_core_SHA256'].items():
    if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('CorechangedsinceexternalGPUgate: '+file)
   dump(Q/'adopted_hn48_GPU_certificate.json',cert);stage('external_hn48_GPU_gate_passed_skip_duplicate8cardfixture')
  elif (H/'skipped_keep_original_plan.json').exists():
   stage('externalgate_notexecuted_resource_or_semantics_unsupported_keep_original8cardgate')
   x=launch('GPU_layout_probe',D/'check_layout_fixture.py',['--backend','gpu','--reference',Q/'cpu_fixture/reference.npz'],p['probe_timeout_seconds'])
   if x['returncode'] or not read(Path(x['output'])/'acceptance.json')['passed']:raise RuntimeError('Original8cardGPUgate failed')
  else:raise RuntimeError('Actualhn48GPUlayoutgate failed; no64kproduction; preservefailure')
  used=ledger();remaining=35-used;timeout=int(remaining/8*3600)-90
  if timeout<1800:raise RuntimeError('Insufficient originalbudgetfor64k; stopbeforeproduction')
  stage('full64k_adjoint_running',remaining_GPUh=remaining)
  x=launch('64k_explicit_layout',D/'m5_sharded_worker.py',['--protocol',Q/'protocol.json','--remaining-gpu-hours',remaining],timeout)
  if x['returncode']:raise RuntimeError('64k worker failed/stopped; see originalretainedlog')
  acceptance=read(Path(x['output'])/'acceptance.json');assert acceptance['passed'];summary={'passed':True,'outcome':'hard64k54gradient_completed','acceptance':str(Path(x['output'])/'acceptance.json'),'comparison':acceptance['canonical_mixed_unit_comparison']}
 except Exception as ex:traceback.print_exc();summary={'passed':False,'outcome':'stopped_failure_or_budget','error':repr(ex),'traceback':traceback.format_exc(),'no_retry_no_gate_relaxation':True}
 summary.update(time=now(),GPU_hours=ledger(),cap_GPU_hours=35);dump(Q/'summary.json',summary);stage('terminal',**summary)
 line=f"\n- {now()} M5八卡接续终态：{summary['outcome']}，累计{summary['GPU_hours']:.4f}/35GPUh（含旧5.34246与真实GPU探针）；记录`{Q}/summary.json`；M2先结束、未放宽数学/物理门、所有旧失败保留。\n"
 for name in ['response_design_to_do_list.md','paper/response_design_EXPERIMENTS_20261014.md']:
  with (P/name).open('a') as f:f.write(line)
 print('M5八卡结束',summary['outcome'],'\nGPUh',summary['GPU_hours'],'/35\n记录',Q,flush=True)
