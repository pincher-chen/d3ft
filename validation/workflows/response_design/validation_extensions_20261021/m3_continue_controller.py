"""Authorized FD stopping-rule amendment; hn48 only, separate append-only history."""
import concurrent.futures as cf,datetime,hashlib,json,os,queue,subprocess,threading,time,traceback
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;P=W.parents[3];R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root']);C=Path(json.loads((R/'M3-A/current_continuation.json').read_text())['root']);p=json.loads((C/'protocol.json').read_text());read=lambda f:json.loads(Path(f).read_text());LOCK=threading.RLock();JOBS=[]
def now():return datetime.datetime.now().astimezone().isoformat()
def dump(f,d):
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def ledger():
 with LOCK:
  rows=[{**j,'GPU_hours':(j.get('end_epoch',time.time())-j['start_epoch'])/3600} for j in JOBS];hours=p['historical_GPU_hours']+sum(j['GPU_hours'] for j in rows);d={'time':now(),'historical_GPU_hours':p['historical_GPU_hours'],'total_M3_GPU_hours':hours,'cap':25,'jobs':rows};dump(C/'GPU_ledger.json',d)
  combined=read(R/'GPU_ledger.json');combined['GPU_hours']['M3-A']=hours;combined['time']=now();combined['M3_continuation_ledger']=str(C/'GPU_ledger.json');dump(R/'GPU_ledger_effective.json',combined)
  return hours
def run(task,gpu):
 d,h,sign,resume=task;label=f'd{d}_h{h:g}_s{sign:+d}'+('_resume2000' if resume else '_4000');out=C/('run_'+label);assert not out.exists()
 env=os.environ.copy();env.update(p['environment']);rt=Path(p['runtime']['e3']);tmp=f'/dev/shm/m3_cont_{os.getpid()}_{label}';env.update(CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env['PYTHONPATH']=':'.join([str(rt),str(rt/'validation/workflows/response_design/reverse_response'),str(rt/'validation/workflows/response_design'),str(rt/'validation/workflows/phase3'),p['jrystal']])
 for key in list(env):
  if key.startswith('D3FT_') and any(x in key for x in ['CHECKPOINT','RESUME','RESULTS_ROOT','YIELD']):env.pop(key)
 args=['--root',p['context'],'--direction',str(d),'--h',str(h),'--sign',str(sign),'--epochs',str(2000 if resume else 4000),'--output',str(out)]
 if resume:args+=['--resume-from',p['resume_from']]
 cmd=['srun','--jobid=275669','--overlap','--nodelist=hn48','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c4','env','CUDA_VISIBLE_DEVICES='+str(gpu),'bash','-c','mkdir -p "$TMPDIR"; printf "STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"','worker','timeout','--signal=TERM','--kill-after=20s',str(p['point_timeout_seconds']),p['python'],str(rt/'validation/workflows/response_design/reverse_response/e3_mg_fd_continue_20261021.py'),*args]
 with LOCK:
  used=ledger();reserved=sum(max(0,j['timeout_seconds']-(time.time()-j['start_epoch']))/3600 for j in JOBS if 'end_epoch' not in j)
  if used+reserved+(p['point_timeout_seconds']+25)/3600>25:raise RuntimeError('25GPUh reservation guard; no new point')
  rec={'direction':d,'h':h,'sign':sign,'resume':resume,'gpu':gpu,'node':'hn48','jobid':275669,'output':str(out),'label':label,'start_epoch':time.time(),'timeout_seconds':p['point_timeout_seconds'],'command':cmd};JOBS.append(rec)
 (C/'logs').mkdir(exist_ok=True)
 with (C/'logs'/(label+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with LOCK:rec['pid']=proc.pid;dump(C/'logs'/(label+'_launch.json'),rec);ledger()
  while proc.poll() is None:time.sleep(15);ledger()
  with LOCK:rec.update(end_epoch=time.time(),returncode=proc.returncode);dump(C/'logs'/(label+'_completion.json'),rec);ledger()
 print('POINT_COMPLETE',label,'returncode',rec['returncode'],flush=True)
 return rec

def aggregate(old):
 good={ (q['direction'],q['h'],q['sign']):q for q in old if q['returncode']==0}
 for q in JOBS:
  if q.get('returncode')==0:good[q['direction'],q['h'],q['sign']]=q
 dump(C/'merged_good_point_index.json',list(good.values()));g=np.array(read(p['source_gradient_certificate'])['reverse_gradient_eV']);rows=[];incomplete=[]
 for i,d in enumerate(p['directions']):
  block=[]
  for h in p['steps']:
   keys=[(i,h,s) for s in [-1,1]]
   if not all(key in good for key in keys):incomplete.append({'direction':i,'h':h,'reason':'missing or rejected same registered point'});continue
   gaps={}
   for key in keys:
    x=good[key];o=Path(x['output']);a=read(o/'point.json');m=read(o/'metadata.json');s=read(o/'spectrum_summary.json');assert read(o/'status.json')['stage']=='completed'
    assert a['residual']<=1e-8 and a['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(a['integrated_valence_density']-a['electrons'])<=1e-8*a['electrons'];assert s['maximum_ritz_residual_Ha']<=1e-9 and s['orthogonality_max']<=1e-10 and min(s['vbm_adjacent_separation_Ha'],s['cbm_adjacent_separation_Ha'])>=1e-5
    assert all(m[k] for k in ['fixed_parent_miller_basis','fixed_parent_beta_tables','fixed_parent_local_radial_nodes']);assert np.array_equal(m['direction_vector'],d)
    for k,v in [('positions_bohr','requested_positions_bohr'),('cell_bohr','requested_cell_bohr')]:assert np.max(abs(np.array(a[k])-np.array(m[v])))<2e-11
    gaps[key[-1]]=s['gap_eV']
   ad=float(g@np.array(d));fd=(gaps[1]-gaps[-1])/(2*h);error=abs(fd-ad);block.append({'direction':i,'h':h,'adjoint_eV':ad,'FD_eV':fd,'absolute_error_eV':error,'relative_error':error/abs(ad) if ad else None,'tolerance_eV':3e-5+.002*abs(ad),'passed':error<=3e-5+.002*abs(ad)})
  if len(block)==2:
   shrink=block[0]['absolute_error_eV']/block[1]['absolute_error_eV'] if block[1]['absolute_error_eV'] else None
   for row in block:row['h_to_half_h_error_ratio']=shrink
  rows+=block
 return {'passed':len(good)==20 and len(rows)==10 and all(x['passed'] for x in rows),'good_points':len(good),'comparisons':rows,'incomplete':incomplete,'no_directions_replaced':True,'original_failure_preserved':True,'GPU_hours':ledger(),'cap':25}
if __name__=='__main__':
 assert read(C/'CPU_acceptance.json')['passed'];assert hashlib.sha256((C/'protocol.json').read_bytes()).hexdigest()==read(C/'protocol_SHA256.json')['protocol.json']
 with (C/'manager_started.json').open('x') as f:json.dump({'pid':os.getpid(),'time':now()},f)
 old=read(p['original_point_index']);keys={(x['direction'],x['h'],x['sign']) for x in old};tasks=queue.Queue();tasks.put((0,.001,-1,True))
 for i in range(5):
  for h in p['steps']:
   for s in [-1,1]:
    if (i,h,s) not in keys:tasks.put((i,h,s,False))
 assert tasks.qsize()==13
 def worker(gpu):
  while True:
   try:task=tasks.get_nowait()
   except queue.Empty:return
   try:run(task,gpu)
   except Exception as ex:
    print('POINT_LAUNCH_REJECTED',repr(ex),flush=True);dump(C/f'GPU{gpu}_controller_error.json',{'time':now(),'error':repr(ex),'task':task});return
 try:
  with cf.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,[0,1,2,3]))
  result=aggregate(old);dump(C/'acceptance.json',result);dump(C/'status.json',{'time':now(),'stage':'complete','passed':result['passed'],'good_points':result['good_points']})
 except Exception as ex:traceback.print_exc();result={'passed':False,'error':repr(ex),'GPU_hours':ledger()};dump(C/'status.json',{'time':now(),'stage':'failed',**result})
 line=f"\n- {now()} M3-A透明修订续算结束：passed={result['passed']}，合格点{result.get('good_points','见记录')}/20，累计{ledger():.4f}/25GPUh；原失败保留，记录`{C}/acceptance.json`。\n"
 for name in ['response_design_to_do_list.md','paper/response_design_EXPERIMENTS_20261014.md']:
  with (P/name).open('a') as f:f.write(line)
 print('M3续算结束',result['passed'],'\n累计GPUh',ledger(),'\n记录',C,flush=True)
