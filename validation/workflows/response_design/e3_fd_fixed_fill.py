"""Resume preregistered Mg FD on fixed cards, preserving every old attempt."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,json,os,subprocess,time
from run_identity import reserve_name,unique_output_path
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry'
R=E/'runtime';A=unique_output_path(E/'fd_retry4000_fixed');A.mkdir(exist_ok=False)
PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
def now():return datetime.datetime.now().isoformat()
def dump(q,d):
 t=q.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(q)
def status(stage,**kw):dump(E/'mg_fd_queue_status.json',dict(time=now(),stage=stage,jobid=250285,attempt='retry4000_fixed',**kw))
def note(s):
 for q in (P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md'):
  with q.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
def run(task,gpu):
 tag=reserve_name(E,f"mg_fd_d{task['direction']}_h{task['h']:g}_s{task['sign']:+d}_retry4000",output_parent=E,log_parent=A)
 out=E/tag;tmp='/dev/shm/response_design_e3_'+tag
 env=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment']
 env.update(PYTHONPATH=str(R)+':'+env['PYTHONPATH'].split(':')[-1],TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax',PYTHONPYCACHEPREFIX=tmp+'/pycache')
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','5400',PY,'-u',str(R/'validation/workflows/response_design/reverse_response/e3_mg_fd_point.py'),'--root',str(E),'--direction',str(task['direction']),'--h',str(task['h']),'--sign',str(task['sign']),'--epochs','4000','--output',str(out)]
 rec=dict(**task,tag=tag,start=now(),jobid=250285,gpu=gpu,command=cmd,environment=env,status='running',output=str(out));start=time.monotonic()
 with (A/(tag+'.log')).open('xb') as f:
  proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(E/'commands'/(tag+'.json'),rec);code=proc.wait()
 rec.update(end=now(),returncode=code,GPU_hours=(time.monotonic()-start)/3600,status='completed' if code==0 else 'failed');dump(E/'commands'/(tag+'.json'),rec)
 return rec

old_index=json.loads((E/'mg_fd_index.json').read_text());dump(A/'old_index.json',old_index)
dump(A/'old_queue_status.json',json.loads((E/'mg_fd_queue_status.json').read_text()))
assert json.loads((E/'mg_mode_acceptance.json').read_text())['passed']
status('retry_mixed_FD_running',gpus=[2,3,4,5,7],epochs=4000,physics_gate_unchanged=True)
directions=json.loads((E/'mg_fd_directions.json').read_text());done=[];tasks=[]
for d in range(3):
 for h in directions['steps']:
  for sign in (-1,1):
   valid=[r for r in old_index if r['direction']==d and r['h']==h and r['sign']==sign and json.loads((Path(r['output'])/'status.json').read_text()).get('stage')=='completed']
   if valid:done.append(valid[0])
   else:tasks.append(dict(direction=d,h=h,sign=sign))
spent=sum(json.loads(q.read_text()).get('GPU_hours',0) for q in (E/'commands').glob('mg_fd_*.json'))
dump(A/'protocol.json',dict(time=now(),GPUs=[2,3,4,5,7],epochs=4000,prior_GPU_hours=spent,cap_GPU_hours=22,retained_successes=done,remaining_tasks=tasks,gate_electronic_Ha=1e-8))
note('E3完整54mode PASS；原混合FD +h点1.10025e-8略超1e-8，失败保留；固定GPU2/3/4/5/7独立重算余下11点，epochs4000，不放宽数值门。')
try:
 for offset in range(0,len(tasks),5):
  batch=tasks[offset:offset+5]
  if spent+1.5*len(batch)>22:raise RuntimeError('FD total 22 GPUh cap; no further tasks submitted')
  with ThreadPoolExecutor(max_workers=5) as pool:
   jobs=[pool.submit(run,t,g) for t,g in zip(batch,[2,3,4,5,7])]
   records=[j.result() for j in jobs]
  spent+=sum(r['GPU_hours'] for r in records)
  for r in records:
   if r['returncode']==0:done.append({k:r[k] for k in ('direction','h','sign','output')})
  dump(E/'mg_fd_index.json',done);status('retry_mixed_FD_running',gpus=[2,3,4,5,7],completed_points=len(done),spent_GPU_hours=spent)
 if len(done)!=12:raise RuntimeError('Some retry FD points failed; preserved, full gate not passed')
 with (A/'acceptance.log').open('xb') as f:
  c=subprocess.run(['python',str(R/'validation/workflows/response_design/reverse_response/e3_mg_fd_acceptance.py'),'--root',str(E)],stdout=f,stderr=subprocess.STDOUT)
 if c.returncode:raise RuntimeError('Mixed h/h2 physical/numerical acceptance failed')
 status('E3_development_derivative_gates_PASS_formal_timing_pending',spent_GPU_hours=spent)
 note('E3全部54mode及三个混合方向h/h2重优化FD PASS；hn48正式计时由既有独占门自动接续。')
 print('E3 混合FD补算及原门验收 PASS。\n三个方向h/h2完整，旧失败保留。\nhn48正式计时自动接续。',flush=True)
except Exception as exc:
 status('failed',error=str(exc),spent_GPU_hours=spent);note('E3混合FD补算失败：'+str(exc)+'；不认定正式计时可开始。');raise
