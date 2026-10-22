"""Detached, bounded four-experiment queue; hn35 only, immutable worker outputs."""
import concurrent.futures as cf,datetime,hashlib,json,os,queue,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;P=W.parents[3]
ROOT=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
read=lambda p:json.loads(Path(p).read_text())
PRO={k:read(ROOT/k/'protocol.json') for k in ['M4-A','M3-A','M5-A','M2-A']}
LOCK=threading.RLock();JOBS=read(ROOT/'infrastructure_binding_failure.json')['jobs'];STATES={};STOP_FD=threading.Event()
def now():return datetime.datetime.now().astimezone().isoformat()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def update(k,stage,**kw):
 with LOCK:
  STATES[k]={'time':now(),'stage':stage,**kw};dump(ROOT/k/'controller_status.json',STATES[k]);dump(ROOT/'controller_status.json',{'time':now(),'pid':os.getpid(),'experiments':STATES})
 print(k,stage,kw,flush=True)
def ledger():
 with LOCK:
  live=[];totals={k:0. for k in PRO}
  for r in JOBS:
   q={**r};q['GPU_hours']=max(0,(q.get('end_epoch',time.time())-q['start_epoch']))/3600;totals[q['experiment']]+=q['GPU_hours'];live.append(q)
  dump(ROOT/'GPU_ledger.json',{'time':now(),'GPU_hours':totals,'caps':{k:p['GPU_hours_cap'] for k,p in PRO.items()},'jobs':live})
  return totals
SNAPCMD=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
def snapshot():
 for attempt in range(3):
  try:
   r=subprocess.run(SNAPCMD,capture_output=True,text=True,timeout=45,check=True);rows=[[int(float(x.strip())) for x in line.split(',')] for line in r.stdout.strip().splitlines()];assert len(rows)==8
   return {'time':now(),'node':'hn35','jobid':250285,'query':SNAPCMD,'raw':r.stdout,'rows':rows}
  except Exception as ex:
   print('GPU_PROBE_RETRY',attempt,repr(ex),flush=True);time.sleep(5)
 return None

def run(k,label,gpu,rt,script,args,timeout):
 label=label+'_r2';p=PRO[k];folder=ROOT/k;logs=folder/'logs';logs.mkdir(exist_ok=True)
 # Each output is a child that the driver creates, never pre-created here.
 out=folder/('run_'+label);assert not out.exists(),('Immutable run exists',out)
 while True:
  snap=snapshot()
  if snap and snap['rows'][gpu][1:]==[0,0]:break
  update(k,'waiting_assigned_card',gpu=gpu,label=label);time.sleep(30)
 with LOCK:
  totals=ledger();reserved=sum(max(0,j['timeout_seconds']-(time.time()-j['start_epoch']))/3600 for j in JOBS if j['experiment']==k and 'end_epoch' not in j)
  if totals[k]+reserved+(timeout+25)/3600>p['GPU_hours_cap']:raise RuntimeError('Reserved timeout would exceed fixed cap')
 env=os.environ.copy();env.update(p['environment']);tmp=f'/dev/shm/internal_review_{os.getpid()}_{k}_{label}'
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 rt=Path(rt);env['PYTHONPATH']=':'.join([str(rt),str(rt/'validation/workflows/response_design/reverse_response'),str(rt/'validation/workflows/response_design'),str(rt/'validation/workflows/phase3'),p['jrystal']])
 # No inherited resume/output/yield paths are allowed to write old results.
 for key in list(env):
  if key.startswith('D3FT_') and any(t in key for t in ['CHECKPOINT','RESUME','YIELD','RESULTS_ROOT']):env.pop(key)
 if k=='M2-A':env.pop('D3FT_RESPONSE_SINGLE_DEVICE',None)
 command=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; printf "SLURM_STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"','worker','env','CUDA_VISIBLE_DEVICES='+str(gpu),'timeout','--signal=TERM','--kill-after=20s',str(timeout),p['python'],str(script),*map(str,args),'--output',str(out)]
 record={'experiment':k,'label':label,'gpu':gpu,'node':'hn35','jobid':250285,'time':now(),'start_epoch':time.time(),'timeout_seconds':timeout,'output':str(out),'command':command,'log':str(logs/(label+'.log'))}
 with (logs/(label+'.log')).open('x') as log:
  proc=subprocess.Popen(command,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=proc.pid
  with LOCK:JOBS.append(record);dump(logs/(label+'_launch.json'),record);ledger()
  while proc.poll() is None:
   time.sleep(15);ledger()
   if k=='M2-A':
    snap=snapshot()
    if snap:
     with (folder/'occupancy.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
     if any(row[1]>0 or row[2]>0 for row in snap['rows'] if row[0]!=gpu):dump(folder/'exclusive_violation.json',snap)
    else:dump(folder/'exclusive_violation.json',{'time':now(),'error':'occupancy query failed; exclusivity cannot be certified'})
  record.update(returncode=proc.returncode,end_epoch=time.time());dump(logs/(label+'_completion.json'),record);ledger()
 return record

def terminal(k,result):
 with LOCK:
  result={**result,'time':now(),'GPU_hours':ledger()[k],'cap_GPU_hours':PRO[k]['GPU_hours_cap']};dump(ROOT/k/'summary.json',result);update(k,'terminal',**result)
  line=f"\n- {now()} {k}：{result.get('outcome')}；{result['GPU_hours']:.4f}/{result['cap_GPU_hours']} GPU·h；记录`{ROOT.relative_to(P)}/{k}/summary.json`。原门与失败记录保留，未改正式稿。\n"
  for name in ['response_design_to_do_list.md','paper/response_design_EXPERIMENTS_20261014.md']:
   with (P/name).open('a') as f:f.write(line)
 print(f"{k} 完成：{result.get('outcome')}\n机时 {result['GPU_hours']:.4f}/{result['cap_GPU_hours']} GPU·h\n记录 {ROOT/k/'summary.json'}",flush=True)

def m4():
 k='M4-A';p=PRO[k];update(k,'design_running',gpu=0)
 r=run(k,'coordinates48',0,p['runtime']['gamma'],W/'m4_worker.py',['--protocol',ROOT/k/'protocol.json'],p['timeout_seconds'])
 if r['returncode']:raise RuntimeError('Worker returncode '+str(r['returncode']))
 a=read(Path(r['output'])/'acceptance.json');return {'outcome':'target_reached' if a['passed'] else 'completed_target_not_reached','acceptance':str(Path(r['output'])/'acceptance.json'),'trajectory_summary':a['trajectory_summary']}

def m3():
 k='M3-A';p=PRO[k];tasks=queue.Queue();results=[]
 for direction in range(5):
  for h in p['steps']:
   for sign in [-1,1]:tasks.put((direction,h,sign))
 def worker(gpu):
  while not STOP_FD.is_set():
   try:d,h,sign=tasks.get_nowait()
   except queue.Empty:return
   label=f'd{d}_h{h:g}_s{sign:+d}';update(k,'FD_points_running',gpu=gpu,label=label,completed=len(results))
   r=run(k,label,gpu,p['runtime']['e3'],Path(p['runtime']['e3'])/'validation/workflows/response_design/reverse_response/e3_mg_fd_point.py',['--root',p['context'],'--direction',d,'--h',h,'--sign',sign,'--epochs',p['epochs']],p['point_timeout_seconds']);r.update(direction=d,h=h,sign=sign)
   with LOCK:results.append(r);dump(ROOT/k/'point_index.json',results)
   if r['returncode']:STOP_FD.set()
 with cf.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,p['gpus']))
 if len(results)!=20 or any(r['returncode'] for r in results):raise RuntimeError('Failed/incomplete points; no new directions/retry; inspect point_index')
 g=np.array(read(p['source_gradient_certificate'])['reverse_gradient_eV']);rows=[]
 for d,vec in enumerate(p['directions']):
  errors=[]
  for h in p['steps']:
   pair=[r for r in results if r['direction']==d and r['h']==h];gaps={}
   for r in pair:
    out=Path(r['output']);m=read(out/'metadata.json');q=read(out/'point.json');s=read(out/'spectrum_summary.json')
    assert read(out/'status.json')['stage']=='completed';assert all(m[x] for x in ['fixed_parent_miller_basis','fixed_parent_beta_tables','fixed_parent_local_radial_nodes'])
    assert q['residual']<=1e-8 and q['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(q['integrated_valence_density']-q['electrons'])<=1e-8*q['electrons']
    assert s['maximum_ritz_residual_Ha']<=1e-9 and s['orthogonality_max']<=1e-10 and min(s['vbm_adjacent_separation_Ha'],s['cbm_adjacent_separation_Ha'])>=1e-5
    for key,mkey in [('positions_bohr','requested_positions_bohr'),('cell_bohr','requested_cell_bohr')]:assert np.max(abs(np.array(q[key])-np.array(m[mkey])))<2e-11
    assert np.array_equal(np.array(m['direction_vector']),np.array(vec));gaps[r['sign']]=s['gap_eV']
   ad=float(g@np.array(vec));fd=(gaps[1]-gaps[-1])/(2*h);err=abs(fd-ad);errors.append(err);tol=3e-5+.002*abs(ad)
   rows.append({'direction':d,'h':h,'adjoint_eV':ad,'FD_eV':fd,'absolute_error_eV':err,'relative_error':None if ad==0 else err/abs(ad),'tolerance_eV':tol,'passed':err<=tol})
  for row in rows[-2:]:row['h_error_over_half_h_error']=None if errors[1]==0 else errors[0]/errors[1]
 a={'passed':all(r['passed'] for r in rows),'rows':rows,'seed':p['seed'],'direction_units':'engineering strain + Cartesian Bohr; numerical unit vector','point_index':str(ROOT/k/'point_index.json')};dump(ROOT/k/'acceptance.json',a)
 return {'outcome':'derivative_comparison_passed' if a['passed'] else 'derivative_comparison_failed','acceptance':str(ROOT/k/'acceptance.json')}

def m5():
 k='M5-A';p=PRO[k];update(k,'frozen_density_spectrum_reconstruction',gpus=p['spectra_gpus'])
 def worker(pair):
  offset,gpu=pair;return run(k,f'spectrum_block{offset}',gpu,p['runtime']['mesh'],Path(p['runtime']['mesh'])/'validation/workflows/response_design/reverse_response/e6_k4_spectra.py',['--source',p['source'],'--indices',json.dumps(list(range(offset,64,3))),'--ritz-restarts',p['spectral_restarts']],p['spectra_timeout_seconds'])
 with cf.ThreadPoolExecutor(max_workers=3) as pool:runs=list(pool.map(worker,enumerate(p['spectra_gpus'])))
 if any(r['returncode'] for r in runs):raise RuntimeError('Frozen spectrum reconstruction failed; no semantic fallback')
 rows=[]
 for r in runs:
  out=Path(r['output']);assert read(out/'status.json')['passed']
  for f in sorted(out.glob('k_????.json')):
   row=read(f);orb=out/f"k_{row['k_index']:04d}_attempt{row['ritz_restart_attempt']}_orbitals.npy";assert orb.exists() and row['passed'];row['orbitals_file']=str(orb);row['orbitals_SHA256']=sha(orb);rows.append(row)
 rows.sort(key=lambda r:r['k_index']);assert [r['k_index'] for r in rows]==list(range(64));spectrum=ROOT/k/'reconstructed_spectrum.json';dump(spectrum,{'rows':rows,'same_accepted_density':True})
 update(k,'hard64k_adjoint',gpu=5)
 r=run(k,'hard64k_adjoint',5,p['runtime']['mesh'],W/'m5_response_worker.py',['--protocol',ROOT/k/'protocol.json','--spectra',spectrum],p['response_timeout_seconds'])
 if r['returncode']:raise RuntimeError('Existing hard64k response unsupported/failed; see retained log, no fallback')
 a=read(Path(r['output'])/'acceptance.json');return {'outcome':'gradient_completed','acceptance':str(Path(r['output'])/'acceptance.json'),'comparison':a['canonical_mixed_unit_comparison']}

def m2():
 k='M2-A';p=PRO[k];update(k,'waiting_three_experiments_and_exclusive_node')
 while not all((ROOT/j/'summary.json').exists() for j in ['M4-A','M3-A','M5-A']):time.sleep(30)
 while True:
  snap=snapshot()
  if snap and all(row[1:]==[0,0] for row in snap['rows']):break
  update(k,'waiting_all8_idle');time.sleep(60)
 dump(ROOT/k/'exclusive_prelaunch.json',snap);update(k,'exclusive_timing_running',gpu=0)
 r=run(k,'exclusive_N54',0,p['runtime']['e3'],Path(p['runtime']['e3'])/'validation/workflows/response_design/reverse_response/m2_exclusive_timing.py',['--root',p['E3_root'],'--N','54','--budget-hours','32'],p['timeout_seconds'])
 if r['returncode']:raise RuntimeError('Timing incomplete/invalid; retain partials and do not retry')
 a=read(Path(r['output'])/'result.json');assert a['passed'] and len(a['reverse'])==len(a['forward'])==3 and not (ROOT/k/'exclusive_violation.json').exists()
 return {'outcome':'exclusive3plus3_completed','acceptance':str(Path(r['output'])/'result.json'),'active_GPU_hours':a['active_GPU_hours'],'whole8_reserved_GPU_hours':a['whole_eight_GPU_node_reservation_hours']}

def guarded(k,fn):
 try:terminal(k,fn())
 except Exception as ex:
  traceback.print_exc();terminal(k,{'outcome':'stopped_failure_or_resource_limit','error':repr(ex),'traceback':traceback.format_exc(),'failure_kind':'inspect retained worker logs; no automatic physical-failure claim'})
if __name__=='__main__':
 assert read(ROOT/'CPU_acceptance.json')['passed']
 assert read(ROOT/'binding_CPU_acceptance.json')['passed']
 for k,p in PRO.items():assert sha(ROOT/k/'protocol.json')==read(ROOT/'CPU_acceptance.json')['protocol_SHA256'][k]
 dump(ROOT/'controller_launch_r2.json',{'pid':os.getpid(),'time':now(),'script':str(Path(__file__)),'SHA256':sha(__file__),'node':'hn35','jobid':250285})
 # Prevent accidental duplicate manager with immutable launch sentinel.
 with (ROOT/'manager_started_r2.json').open('x') as f:json.dump({'pid':os.getpid(),'time':now()},f)
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(guarded,k,fn) for k,fn in [('M4-A',m4),('M3-A',m3),('M5-A',m5),('M2-A',m2)]]
  for f in futures:f.result()
 dump(ROOT/'all_finished.json',{'time':now(),'GPU_hours':ledger()})
