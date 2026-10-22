"""Si-gated MgSiN2 full-mode development queue, never uses hn48."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import datetime,json,os,subprocess,time,hashlib
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry'
cfg=json.loads((E/'development_protocol.json').read_text());R=Path(cfg['runtime'])
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())
env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
ap=__import__('argparse').ArgumentParser();ap.add_argument('--attempt',default='initial');ap.add_argument('--si-gate',default='si_acceptance.json');args=ap.parse_args()
if args.si_gate!='si_acceptance.json':
 old=json.loads((E/'mg_controller_retry1_launch.json').read_text())['pid']
 while True:
  try:state=Path(f'/proc/{old}/stat').read_text().split(') ',1)[1].split()[0]
  except FileNotFoundError:break
  if state=='Z':break
  time.sleep(5)
os.close(os.open(E/f'mg_queue_{args.attempt}.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def dump(path,obj):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(path)
def status(stage,**kw):
 d=dict(time=datetime.datetime.now().isoformat(),jobid=250285,formal_timing=False,stage=stage,attempt=args.attempt,**kw)
 dump(E/'mg_queue_status.json',d);print('E3_MG_QUEUE',stage,kw,flush=True)
def free_gpus():
 q=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1',
  'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 r=subprocess.run(q,capture_output=True,text=True,timeout=45)
 if r.returncode:raise RuntimeError('GPU query failed '+r.stderr[-1000:])
 out=[]
 for row in r.stdout.splitlines():
  i,m,u=map(int,row.split(','))
  if i in cfg['gpu_candidates'] and m<=3 and u==0:out.append(i)
 return out
try:
 status('waiting_Si_complete_derivative_acceptance')
 while not (E/args.si_gate).exists():
  s=E/('si_refinement_status.json' if args.si_gate!='si_acceptance.json' else 'si_queue_status.json')
  if s.exists() and json.loads(s.read_text())['stage']=='failed':raise RuntimeError('Si validation failed; Mg not submitted')
  time.sleep(20)
 if not json.loads((E/args.si_gate).read_text())['passed']:raise RuntimeError('Si FD/mode/translation gate failed')
 layout=json.loads((E/'geometry_layout_checks.json').read_text());assert layout['passed']
 status('waiting_four_free_development_gpus')
 while True:
  gpus=free_gpus()
  if len(gpus)>=4:break
  time.sleep(20)
 strain=[.003,-.002,.001,.002,-.001,.0015]
 rp=dict(source=cfg['Mg']['source'],orbitals=cfg['Mg']['orbitals'],runtime=str(R),theta0=strain+[0.]*48,
  include_coordinates=True,displacement_unit='bohr',development_memory_probe=True,
  source_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
   (R/'validation/workflows/response_design/reverse_response').glob('*.py')})
 dump(E/'mg_response_protocol.json',rp)
 rng=np.random.default_rng(cfg['Mg']['directions_seed']);directions=[]
 for i in range(3):
  strain_d=rng.normal(size=6);strain_d*=.3/np.linalg.norm(strain_d)
  atoms=rng.normal(size=(16,3));atoms-=atoms.mean(axis=0);atoms*=.7/np.linalg.norm(atoms)
  directions.append(np.r_[strain_d,atoms.reshape(-1)].tolist())
 dump(E/'mg_fd_directions.json',dict(seed=cfg['Mg']['directions_seed'],directions=directions,
  steps=cfg['Mg']['FD_steps'],frozen_before_Mg_derivatives=True,
  definition='scalar mixed step t: strain += t*d_strain, Cartesian Bohr += t*d_atomic; atomic translations projected out'))
 jobs=[];specs=[('mg_reverse','reverse',[],3600)]
 for i,axes in enumerate(np.array_split(np.arange(54),3)):
  specs.append((f'mg_forward_{i}','forward',axes.tolist(),14400))
 for gpu,(tag,mode,axes,limit) in zip(gpus,specs):
  tmp='/dev/shm/response_design_e3_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4',
   'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}',
   'timeout',str(limit),cfg['python'],'-u',str(R/'validation/workflows/response_design/reverse_response/gamma_response.py'),
   str(E/'mg_response_protocol.json'),'--mode',mode,'--output',str(E/tag)]
  if axes:cmd+=['--axes',*map(str,axes)]
  rec=dict(tag=tag,jobid=250285,gpu=gpu,command=cmd,axes=axes,start=datetime.datetime.now().isoformat(),status='running',environment={k:te[k] for k in env})
  dump(E/'commands'/f'{tag}.json',rec)
  f=(E/'logs'/f'{tag}.log').open('xb');start=time.monotonic()
  proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  rec['pid']=proc.pid;dump(E/'commands'/f'{tag}.json',rec);jobs.append((proc,f,start,rec))
 status('full_54_responses_running',gpus=gpus,hard_substep_max_GPU_hours=13)
 while any(proc.poll() is None for proc,*_ in jobs):time.sleep(10)
 spent=0.;failed=[]
 for proc,f,start,rec in jobs:
  f.close();code=proc.returncode
  # Elapsed end inferred from durable worker terminal status, not later slow peer completion.
  statusfile=E/rec['tag']/'status.json';terminal=json.loads(statusfile.read_text()) if statusfile.exists() else {}
  end=terminal.get('time',datetime.datetime.now().isoformat())
  duration=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(rec['start'])).total_seconds()
  rec.update(returncode=code,end=end,GPU_hours=max(0.,duration)/3600,status='completed' if code==0 else 'failed')
  spent+=rec['GPU_hours'];dump(E/'commands'/(rec['tag']+'.json'),rec)
  if code:failed.append(rec['tag'])
 if failed:raise RuntimeError('Responses failed; no FD queue: '+str(failed))
 gate=R/'validation/workflows/response_design/reverse_response/e3_mg_mode_acceptance.py'
 with (E/'logs/mg_mode_check.log').open('xb') as f:
  run=subprocess.run(['python','-u',str(gate),'--root',str(E)],stdout=f,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError('Mg full-mode/translation/memory acceptance failed')
 status('Mg_54_mode_translation_memory_PASS_FD_pending',spent_GPU_hours=spent)
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3子步骤MgSiN2：54分量forward/reverse、整体平移、旧六维回归和显存短测PASS；混合方向h/h2重优化FD仍待验收，不提交正式计时。证据E3/mg_mode_acceptance.json。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3 MgSiN2 54维模式/平移/显存子步骤通过，混合FD和正式计时待做。\n')
 print('E3 MgSiN2：54 分量模式、平移与显存短测 PASS。\n证据：mg_mode_acceptance.json；混合 FD 尚待验收。\n正式计时未提交，hn48 未使用。',flush=True)
except Exception as exc:
 status('failed',error=str(exc));raise
