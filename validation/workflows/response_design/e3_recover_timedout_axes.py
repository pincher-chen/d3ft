"""Recover only missing accepted tangent axes after a 124 timeout, no scancel.

Runs nothing if primary mode acceptance succeeds. Physics failures are NOT
retried. Original partial results/failed logs remain immutable; merged evidence
is written to a new directory with every underlying source named and hashed.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import json,os,time,datetime,subprocess,signal,csv,hashlib
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry';R=E/'runtime'
cfg=json.loads((E/'development_protocol.json').read_text())
def read(p):return json.loads(p.read_text())
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):dump(E/'mode_recovery_status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),**kw))
try:
 status('waiting_primary_outcome')
 while True:
  if (E/'mg_mode_acceptance.json').exists() and read(E/'mg_mode_acceptance.json')['passed']:status('not_needed');__import__('sys').exit(0)
  s=read(E/'mg_queue_status.json')
  if s['stage']=='failed':break
  time.sleep(30)
 dump(E/'mg_primary_failed_status.json',s)
 launches=[read(E/'commands'/f'mg_forward_{i}.json') for i in range(3)]
 if any(r.get('returncode') not in [0,124] for r in launches):raise RuntimeError('Not a pure timeout; no automatic physics retry')
 if read(E/'commands/mg_reverse.json').get('returncode')!=0:raise RuntimeError('Adjoint failed; no timeout salvage')
 # Supersede only our gate-only CPU dispatcher, before any FD GPU submission.
 old=read(E/'mg_fd_controller_refined_launch.json')['pid']
 if not list((E/'commands').glob('mg_fd_d*.json')):
  try:
   cmdline=Path(f'/proc/{old}/cmdline').read_bytes()
   if b'e3_queue_mg_fd.py' in cmdline:os.kill(old,signal.SIGTERM)
  except FileNotFoundError:pass
 else:raise RuntimeError('FD tasks already submitted; do not launch duplicates')
 base=read(P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json');env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
 jobs=[];groups=[]
 for i,rec in enumerate(launches):
  original=E/f'mg_forward_{i}';file=original/'result.json' if (original/'result.json').exists() else original/'partial_results.json'
  data=read(file);accepted=data['results'];expected=rec['axes']
  if not set(map(int,accepted)).issubset(set(expected)):raise RuntimeError('Unexpected saved axes')
  for value in accepted.values():
   if value['response_record']['converged'] is not True:raise RuntimeError('Unqualified partial solve')
  missing=[a for a in expected if str(a) not in accepted];group=dict(i=i,accepted=accepted,missing=missing,original_file=str(file));groups.append(group)
  if not missing:continue
  tag=f'mg_forward_timeout_recovery_{i}';gpu=rec['gpu'];tmp='/dev/shm/response_design_e3_'+tag
  te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
  # Primary workers are finished; these original bound cards are idle unless
  # explicitly reused in the meantime. Query immediately before each launch.
  while True:
   q=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45)
   if q.returncode:raise RuntimeError('GPU availability query failed')
   mem,util=map(int,q.stdout.strip().split(','))
   if mem<=3 and util==0:break
   time.sleep(20)
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','7200',cfg['python'],'-u',str(R/'validation/workflows/response_design/reverse_response/gamma_response.py'),str(E/'mg_response_protocol.json'),'--mode','forward','--axes',*map(str,missing),'--output',str(E/tag)]
  launch=dict(tag=tag,jobid=250285,gpu=gpu,axes=missing,command=cmd,start=datetime.datetime.now().isoformat(),status='running')
  dump(E/'commands'/f'{tag}.json',launch);f=(E/'logs'/f'{tag}.log').open('xb');p=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);launch['pid']=p.pid;dump(E/'commands'/f'{tag}.json',launch);jobs.append((p,f,launch,group))
 status('recovering_missing_axes',groups=[dict(i=g['i'],missing=g['missing']) for g in groups],max_extra_GPU_hours=6,E3_total_budget=120)
 while any(p.poll() is None for p,*_ in jobs):time.sleep(10)
 for proc,f,rec,group in jobs:
  f.close();end=read(E/rec['tag']/'status.json')['time'];hours=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(rec['start'])).total_seconds()/3600
  rec.update(returncode=proc.returncode,end=end,GPU_hours=hours,status='completed' if proc.returncode==0 else 'failed');dump(E/'commands'/(rec['tag']+'.json'),rec)
  if proc.returncode:raise RuntimeError('Missing-axis recovery failed; no further retry')
  extra=read(E/rec['tag']/'result.json')['results']
  if set(extra)!=set(map(str,group['missing'])):raise RuntimeError('Recovery incomplete')
  group['accepted']={**group['accepted'],**extra};group['recovery_file']=str(E/rec['tag']/'result.json')
 forward={};memory=[];provenance=[]
 for group in groups:
  for key,value in group['accepted'].items():
   if key in forward:raise RuntimeError('Duplicated axis')
   forward[key]=value
  for name in [group['original_file'],group.get('recovery_file')]:
   if name:provenance.append(dict(file=name,sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest()))
 for folder in [E/'mg_reverse',*[E/f'mg_forward_{i}' for i in range(3)],*[E/rec['tag'] for _,_,rec,_ in jobs]]:
  if (folder/'memory_peak.json').exists():memory.append(dict(tag=folder.name,**read(folder/'memory_peak.json')))
  else:
   with (folder/'memory_samples.csv').open() as f:samples=list(csv.DictReader(f))
   if not samples:raise RuntimeError('No original timeout memory evidence')
   memory.append(dict(tag=folder.name,peak_used_MiB=max(float(r['used_MiB']) for r in samples),samples=len(samples),gpu_UUIDs=sorted({r['gpu_uuid'] for r in samples}),reconstructed_from_raw_CSV=True,scope='Sampled lower bound before timed-out process; original file retained'))
 if set(forward)!=set(map(str,range(54))):raise RuntimeError('Not full54')
 reverse=read(E/'mg_reverse/result.json');g=np.asarray(reverse['results']['gap_gradient_eV']);t=np.array([forward[str(i)]['gap_derivative_eV'] for i in range(54)])
 old=np.array(read(E.parent/'response_design_20261014_e0_migration/mgsin2_reverse_retry2/result.json')['results']['gap_gradient_eV'])
 checks=dict(full54=bool(np.all(abs(g-t)<=5e-6+1e-6*np.maximum(abs(g),abs(t)))),translation=bool(max(abs(g[6:].reshape(16,3).sum(axis=0)))<=1e-6 and max(abs(t[6:].reshape(16,3).sum(axis=0)))<=1e-6),six_regression=bool(max(abs(g[:6]-old))<=5e-6),nontrivial_atomic=bool(np.linalg.norm(g[6:])>1e-5),memory=all(r['samples']>0 and r['gpu_UUIDs'] for r in memory))
 report=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,reverse_gradient_eV=g.tolist(),forward_gradient_eV=t.tolist(),max_mode_error=float(max(abs(g-t))),memory=memory,provenance=provenance,timeout_recovery=True,scope='Development comparison only; original timeouts retained; mixed FD/formal still gated')
 dump(E/'mg_mode_acceptance_after_recovery.json',report)
 if not report['passed']:raise RuntimeError('Recovered complete54 failed physical checks')
 if (E/'mg_mode_acceptance.json').exists():dump(E/'mg_mode_acceptance_before_recovery.json',read(E/'mg_mode_acceptance.json'))
 dump(E/'mg_mode_acceptance.json',report);dump(E/'mg_queue_status.json',dict(stage='Mg54_mode_PASS_after_timeout_recovery',attempt='recovery',time=datetime.datetime.now().isoformat(),jobid=250285))
 status('recovered_full54_mode_PASS')
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3开发前向仅超时后，原已收敛轴保留，只补缺轴；完整54模式/平移/显存门PASS，原超时保留，额外最多6GPUh从E3总120内调配。混合FD/正式计时仍门控。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3纯超时缺轴恢复后全54验收PASS；物理失败未重试。\n')
 cmd=['python','-u',str(P/'validation/workflows/response_design/e3_queue_mg_fd.py'),'--attempt','recovery']
 with (E/'mg_fd_controller_recovery.log').open('xb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 dump(E/'mg_fd_controller_recovery_launch.json',dict(pid=p.pid,command=cmd,time=datetime.datetime.now().isoformat()))
 print('E3完整54：超时缺轴补齐后PASS，原失败保留。\n来源/残差/显存见mg_mode_acceptance_after_recovery.json。\n自动接续混合FD；正式计时仍待全部门通过。',flush=True)
except Exception as exc:status('failed',error=str(exc));raise
