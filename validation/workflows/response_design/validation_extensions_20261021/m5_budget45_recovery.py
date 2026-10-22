"""One authorized instrumentation-only M5 retry; original45GPUh shared cap."""
import datetime, hashlib, json, os, subprocess, time, traceback
from pathlib import Path

P=Path(__file__).resolve().parents[4]
R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
Q=Path(json.loads((R/'M5-A/current_sharded.json').read_text())['root'])
D=Path(json.loads((Q/'current_budget45.json').read_text())['root'])
read=lambda f:json.loads(Path(f).read_text())
p=read(D/'protocol.json'); jobs=[]
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
 f=Path(f);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def stage(name,**kw):
 dump(D/'status.json',{'time':now(),'pid':os.getpid(),'stage':name,**kw});print(name,kw,flush=True)
def ledger():
 rows=[{**j,'GPU_hours':8*(j.get('end_epoch',time.time())-j['start_epoch'])/3600} for j in jobs]
 total=p['historical_GPU_hours']+sum(j['GPU_hours'] for j in rows)
 dump(D/'GPU_ledger.json',{'time':now(),'total_M5_GPU_hours':total,'prior_GPU_hours':p['historical_GPU_hours'],
  'prior_ledger':p['previous_GPU_ledger'],'prior_ledger_SHA256':read(D/'CPU_acceptance.json')['prior_ledger_SHA256'],
  'cap':45,'jobs':rows})
 effective=read(R/'GPU_ledger_effective.json');effective['time']=now();effective['GPU_hours']['M5-A']=total
 effective['M5_sharded_ledger']=str(D/'GPU_ledger.json');effective['caps']['M5-A']=45;effective['M5_cap_change_history']=str(D/'cap_change_history.json');dump(R/'GPU_ledger_effective.json',effective)
 return total
def snapshot():
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1',
      'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 for i in range(3):
  try:
   v=subprocess.run(cmd,text=True,capture_output=True,timeout=45,check=True)
   rows=[[int(float(x.strip())) for x in line.split(',')] for line in v.stdout.strip().splitlines()]
   assert len(rows)==8
   return {'time':now(),'node':'hn35','jobid':250285,'rows':rows,'raw':v.stdout}
  except Exception:
   if i==2:raise
   time.sleep(5)
def run():
 acceptance=read(D/'CPU_acceptance.json');assert acceptance['passed']
 for file,h in acceptance['source_SHA256'].items():assert sha(file)==h,('Tested source changed',file)
 assert sha(Path(p['previous_GPU_ledger']))==acceptance['prior_ledger_SHA256']
 assert read(R/'M2-A/summary.json')['passed']
 with (D/'manager_started_cpu_r2.json').open('x') as f:json.dump({'time':now(),'pid':os.getpid()},f)
 stage('waiting_hn35_all8idle')
 while True:
  s=snapshot()
  if all(row[1:]==[0,0] for row in s['rows']):break
  time.sleep(30)
 dump(D/'prelaunch_occupancy.json',s)
 used=ledger();remaining=45-used;timeout=int(remaining/8*3600)-90
 assert timeout>=1800
 assert used+8*(timeout+25)/3600<=45
 rt=Path(p['runtime_path']);out=D/'run_64k_budget45';assert not out.exists()
 env=os.environ.copy();env.update(p['environment']);tmp=f'/dev/shm/m5_probe_fix_{os.getpid()}'
 env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',TMPDIR=tmp,
  PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',PYTHONUNBUFFERED='1',
  OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['PYTHONPATH']=':'.join(map(str,[rt,rt/'validation/workflows/response_design/reverse_response',
  rt/'validation/workflows/response_design',rt/'validation/workflows/phase3',p['jrystal']]))
 for key in list(env):
  if key.startswith('D3FT_') and any(t in key for t in ['CHECKPOINT','RESUME','RESULTS_ROOT','YIELD','RESPONSE_SINGLE_DEVICE']):env.pop(key)
 script=D/'m5_budget45_worker.py'
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c8',
  'env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','bash','-c',
  'mkdir -p "$TMPDIR"; printf "STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"',
  'worker','timeout','--signal=TERM','--kill-after=20s',str(timeout),p['python'],str(script),
  '--protocol',str(D/'protocol.json'),'--remaining-gpu-hours',str(remaining),'--output',str(out)]
 record={'time':now(),'start_epoch':time.time(),'label':'64k_budget45','gpu_count':8,
  'node':'hn35','jobid':250285,'command':cmd,'timeout_seconds':timeout,'output':str(out)}
 with (D/'worker.log').open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=P,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  record['pid']=proc.pid;jobs.append(record);dump(D/'launch.json',record)
  stage('full64k_adjoint_running',prior_GPU_hours=used,remaining_GPU_hours=remaining,worker_srun_pid=proc.pid)
  while proc.poll() is None:ledger();time.sleep(10)
 record.update(end_epoch=time.time(),returncode=proc.returncode);dump(D/'completion.json',record);ledger()
 if proc.returncode:raise RuntimeError('Worker stopped; preserve logs and budget/physics/instrumentation reason; no automatic retry')
 a=read(out/'acceptance.json');assert a['passed']
 import numpy as np, csv
 g=np.array(a['gradient_eV']);old=np.array(a['k2_gradient_eV']);assert g.shape==old.shape==(54,)
 assert np.isfinite(g).all() and a['projection_max']<=1e-9 and a['translation_gradient_max']<=1e-6
 names=['eps_xx','eps_yy','eps_zz','gamma_yz','gamma_xz','gamma_xy']+[f'u{i}_{axis}' for i in range(16) for axis in 'xyz']
 with (D/'gradients_k4_vs_k2.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['component','variable','unit','k2_gradient','k4_gradient','k4_minus_k2'])
  for i,(x,y) in enumerate(zip(old,g)):
   w.writerow([i,names[i],'eV / engineering strain' if i<6 else 'eV / Bohr',x,y,y-x])
 comparison={}
 for label,sl in [('canonical_mixed_units',slice(None)),('strain',slice(6)),('coordinates',slice(6,None))]:
  x=g[sl];y=old[sl];comparison[label]={'cosine':float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),'norm_ratio_k4_over_k2':float(np.linalg.norm(x)/np.linalg.norm(y))}
 comparison.update(projection_max=a['projection_max'],translation_gradient_max=a['translation_gradient_max'],csv=str(D/'gradients_k4_vs_k2.csv'),worker_acceptance=str(out/'acceptance.json'),mixed_unit_note='Canonical full54 cosine is unit-dependent; strain and coordinate blocks also reported separately')
 dump(D/'comparison_summary.json',comparison)
 return {'passed':True,'outcome':'hard64k54gradient_completed','acceptance':str(out/'acceptance.json'),
         'comparison':a['canonical_mixed_unit_comparison']}
if __name__=='__main__':
 try:summary=run()
 except Exception as e:
  traceback.print_exc();summary={'passed':False,'outcome':'stopped_failure_or_budget',
   'error':repr(e),'traceback':traceback.format_exc(),'no_physical_failure_claim_without_log_analysis':True}
 summary.update(time=now(),GPU_hours=ledger(),cap_GPU_hours=45)
 dump(D/'summary.json',summary);stage('terminal',**summary)
 dump(R/'M5-A/current_recovery_summary.json',{'root':str(D),'summary':str(D/'summary.json')})
 line=f"\n- {now()} M5总上限45GPUh续提终态：{summary['outcome']}，累计{summary['GPU_hours']:.5f}/45GPUh（完整保留旧15.37308账本及所有原失败）。记录`{D}/summary.json`；未改响应数学/原数值门/预算规则，未改正式稿。\n"
 for name in ['paper/response_design_EXPERIMENTS_20261014.md','response_design_to_do_list.md']:
  with (P/name).open('a') as f:f.write(line)
 print(summary,flush=True)
