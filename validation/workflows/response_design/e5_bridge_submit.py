"""Real-GPU two-step gates before fresh Mg coupled adjoint/forward."""
import argparse,datetime,json,os,shutil,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_distributed'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();T=a.root
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(Path(q).read_text())
def dump(q,r):
 q=Path(q);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(q)
def status(stage,**kw):
 row=dict(time=now(),stage=stage,jobid=250285,gpus=[0,5],root=str(T),**kw)
 for q in [T/'status.json',E/'queue_status.json',E/'four_card_20261015/queue_status.json']:dump(q,row)
def alive(pid):
 q=Path(f'/proc/{pid}/stat');return q.exists() and q.read_text().split(') ',1)[1].split()[0]!='Z'
try:
 gate=T/'gpu_two_steps/acceptance.json';start=time.monotonic()
 while not gate.exists():
  status('waiting_GPU_two_step_CPU_serial_comparison')
  if time.monotonic()-start>660:raise RuntimeError('GPU toy gate missing; no full adjoint submitted')
  time.sleep(5)
 if not read(gate)['passed']:raise RuntimeError('GPU toy gate failed')
 probe=read(T/'gpu_probe_launch.json')
 while alive(probe['pid']):time.sleep(1)
 probe_end=now();toy_hours=2*(datetime.datetime.fromisoformat(probe_end)-datetime.datetime.fromisoformat(probe['time'])).total_seconds()/3600
 dump(T/'gpu_probe_accounting.json',dict(GPU_hours=toy_hours,scope='Conservative observed completion after gate/child exit',source=str(T/'gpu_probe_launch.json')))
 old=Path(read(T/'previous_attempt.json')['root']);p=read(old/'protocol.json');R=T/'production_runtime';shutil.copytree(T/'runtime',R)
 rel=Path('validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py');shutil.copy2(P/rel,R/rel)
 p.update(runtime=str(R),adopted_k_indices=list(range(8)),prior_GPU_hours=read(old/'status.json')['spent_GPU_hours']+toy_hours,pcg_smoke_only=True)
 p['required_acceptance_files']=[*p['required_acceptance_files'],str(gate)]
 dump(T/'mg_smoke_protocol.json',p)
 original=read(T/'gpu_probe_launch.json');env={**os.environ,**original['environment']};env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
 tmp='/dev/shm/'+T.name+'_Mg2';env.update(TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',D3FT_TRACE_STAGES='1')
 PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
 out=T/'actual_Mg_two_iterations';cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5-Mg2','env','CUDA_VISIBLE_DEVICES=0,5','timeout','1800',PY,'-u',str(R/rel),'--protocol',str(T/'mg_smoke_protocol.json'),'--stage','response','--output',str(out)]
 f=(T/'actual_Mg_two_iterations.log').open('xb');started=time.monotonic();proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(time=now(),pid=proc.pid,jobid=250285,gpus=[0,5],command=cmd,environment=original['environment']|{k:env[k] for k in ['PYTHONPATH','TMPDIR','PYTHONPYCACHEPREFIX','JAX_COMPILATION_CACHE_DIR','D3FT_TRACE_STAGES']},timeout_seconds=1800)
 dump(T/'Mg_probe_launch.json',rec)
 while proc.poll() is None:status('actual_Mg_two_iteration_probe_running',pid=proc.pid,elapsed_seconds=time.monotonic()-started);time.sleep(15)
 f.close();h=2*(time.monotonic()-started)/3600;rec.update(end=now(),returncode=proc.returncode,GPU_hours=h);dump(T/'Mg_probe_launch.json',rec)
 if proc.returncode or not read(out/'smoke_acceptance.json')['passed']:raise RuntimeError('Mg two-iteration gate failed; full adjoint not submitted')
 p.update(pcg_smoke_only=False,prior_GPU_hours=p['prior_GPU_hours']+h)
 p['required_acceptance_files'].append(str(out/'smoke_acceptance.json'));dump(T/'protocol.json',p)
 cmd=['python','-u',str(P/'validation/workflows/response_design/e5_resume_spectra_response.py'),'--root',str(T)]
 f=(T/'controller.log').open('xb');child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 dump(T/'controller_launch.json',dict(time=now(),pid=child.pid,command=cmd,jobid=250285,gpus=[0,5],two_GPU_gates_passed=True));dump(E/'active_followup.json',dict(root=str(T),previous_root=str(old)))
 print('Both real GPU gates PASS; new full response controller '+str(child.pid)+' submitted.',flush=True)
except Exception as exc:status('failed',passed=False,error=str(exc),full_adjoint_submitted=False);raise
