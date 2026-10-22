"""Background mixed-direction FD, only after Si and Mg mode acceptance."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
from run_identity import reserve_name
import datetime,json,os,subprocess,time
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry'
cfg=json.loads((E/'development_protocol.json').read_text());R=Path(cfg['runtime'])
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())
env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
ap=__import__('argparse').ArgumentParser();ap.add_argument('--attempt',default='initial');args=ap.parse_args()
if args.attempt=='refined':
 old=json.loads((E/'mg_fd_controller_launch.json').read_text())['pid']
 while True:
  try:state=Path(f'/proc/{old}/stat').read_text().split(') ',1)[1].split()[0]
  except FileNotFoundError:break
  if state=='Z':break
  time.sleep(5)
os.close(os.open(E/f'mg_fd_queue_{args.attempt}.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def dump(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):
 dump(E/'mg_fd_queue_status.json',dict(time=datetime.datetime.now().isoformat(),jobid=250285,formal_timing=False,stage=stage,attempt=args.attempt,**kw));print('E3_MG_FD_QUEUE',stage,kw,flush=True)
try:
 status('waiting_Mg_full_mode_gate')
 while not (E/'mg_mode_acceptance.json').exists():
  s=E/'mg_queue_status.json'
  if s.exists() and json.loads(s.read_text())['stage']=='failed':raise RuntimeError('Mg mode queue failed; FD not submitted')
  time.sleep(20)
 if not json.loads((E/'mg_mode_acceptance.json').read_text())['passed']:raise RuntimeError('Mg mode gate did not pass')
 d=json.loads((E/'mg_fd_directions.json').read_text());tasks=[]
 for direction in range(3):
  for h in d['steps']:
   for sign in [-1,1]:tasks.append(dict(direction=direction,h=h,sign=sign))
 # Pilot batch: one complete pair, then remaining points; fail before adding further waves.
 batches=[tasks[:2],tasks[2:6],tasks[6:10],tasks[10:]];done=[];spent=0.
 for number,batch in enumerate(batches):
  if spent+len(batch)*1.5>22:raise RuntimeError('FD development hard GPU budget exceeded; no new submission')
  status('waiting_free_gpus',batch=number,spent_GPU_hours=spent)
  while True:
   q=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
   r=subprocess.run(q,capture_output=True,text=True,timeout=45)
   if r.returncode:raise RuntimeError('GPU query failed')
   free=[]
   for row in r.stdout.splitlines():
    i,m,u=map(int,row.split(','))
    if i in cfg['gpu_candidates'] and m<=3 and u==0:free.append(i)
   if len(free)>=len(batch):break
   time.sleep(20)
  jobs=[]
  for gpu,task in zip(free,batch):
   tag=reserve_name(E,f"mg_fd_d{task['direction']}_h{task['h']:g}_s{task['sign']:+d}",output_parent=E,log_parent=E/'logs');out=E/tag;tmp='/dev/shm/response_design_e3_'+tag
   te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
   cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','5400',cfg['python'],'-u',str(R/'validation/workflows/response_design/reverse_response/e3_mg_fd_point.py'),'--root',str(E),'--direction',str(task['direction']),'--h',str(task['h']),'--sign',str(task['sign']),'--output',str(out)]
   rec=dict(**task,tag=tag,jobid=250285,gpu=gpu,command=cmd,output=str(out),start=datetime.datetime.now().isoformat(),status='running',environment={k:te[k] for k in env})
   dump(E/'commands'/f'{tag}.json',rec);f=(E/'logs'/f'{tag}.log').open('xb')
   proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(E/'commands'/f'{tag}.json',rec);jobs.append((proc,f,rec))
  status('mixed_FD_running',batch=number,gpus=free[:len(batch)],tags=[r['tag'] for _,_,r in jobs])
  while any(p.poll() is None for p,*_ in jobs):time.sleep(10)
  failed=[]
  for proc,f,rec in jobs:
   f.close();terminal=E/rec['tag']/'status.json';end=json.loads(terminal.read_text()).get('time') if terminal.exists() else None
   if end is None:end=datetime.datetime.now().isoformat()
   hours=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(rec['start'])).total_seconds()/3600
   rec.update(end=end,returncode=proc.returncode,GPU_hours=max(0,hours),status='completed' if proc.returncode==0 else 'failed');spent+=rec['GPU_hours'];dump(E/'commands'/(rec['tag']+'.json'),rec)
   if proc.returncode:failed.append(rec['tag'])
   else:done.append({k:rec[k] for k in ['direction','h','sign','output']})
  dump(E/'mg_fd_index.json',done)
  if failed:raise RuntimeError('FD failed, retain existing tasks/results; no next wave: '+str(failed))
 with (E/'logs/mg_fd_check.log').open('xb') as f:
  run=subprocess.run(['python','-u',str(R/'validation/workflows/response_design/reverse_response/e3_mg_fd_acceptance.py'),'--root',str(E)],stdout=f,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError('Mixed FD acceptance failed')
 status('E3_development_derivative_gates_PASS_formal_timing_pending',spent_GPU_hours=spent)
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3导数开发验收完成：Si及MgSiN2全模式、平移、3混合方向h/h2重优化FD与显存短测通过；正式N=1/6/24/54×3计时未提交，需之后确认hn48独占。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3导数门全部通过；hn48正式scaling待独占资源检查，不自动盲投。\n')
 print('E3 导数开发：全部验收 PASS。\n证据：si_acceptance、mg_mode_acceptance、mg_fd_acceptance。\n正式 hn48 timing 尚未提交；下一步先确认独占。',flush=True)
except Exception as exc:status('failed',error=str(exc));raise
