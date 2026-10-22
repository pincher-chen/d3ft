"""Gate-driven hn48 serial formal queue, no submission before acceptance."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
from run_identity import reserve_name,await_detached_record
import argparse,json,datetime,subprocess,os,time
import math,statistics
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry';F=E/'formal_timing_20261015';cfg=json.loads((F/'protocol.json').read_text());R=Path(cfg['runtime'])
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text());env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
ap=argparse.ArgumentParser();ap.add_argument('--resume-launch-record',type=Path);args=ap.parse_args()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):dump(F/'status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),jobid=250212,node='hn48',**kw));print('E3_FORMAL_QUEUE',stage,kw,flush=True)
def estimated_seconds(N):
 previous=[]
 for path in F.glob('*/result.json'):
  row=json.loads(path.read_text())
  if row.get('passed') and row.get('forward') and row.get('reverse'):previous.append(row)
 if not previous:return 4*3600.
 per_axis=max(x['seconds']/r['N'] for r in previous for x in r['forward'])
 reverse=max(x['seconds'] for r in previous for x in r['reverse'])
 warm=max(r['warmup_and_compilation_seconds'] for r in previous)
 return warm+3*(reverse+N*per_axis)
def exclusive_check():
 # This CPU-only check has no project path in its own command arguments.
 code="import os,subprocess,json; me=os.getpid(); parent=os.getppid(); p=os.environ['response_PROJECT_PATH']; alias=os.environ['response_PROJECT_ALIAS']; rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines(); matches=[r for r in rows if (p in r or alias in r) and int(r.split()[0]) not in [me,parent] and 'e3_queue_formal' not in r]; gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True); print(json.dumps(dict(repo_processes=matches,gpu_apps=gpu.stdout.strip(),gpu_query_rc=gpu.returncode,passed=not matches and not gpu.stdout.strip() and gpu.returncode==0)))"
 te={**os.environ,'response_PROJECT_PATH':str(P),'response_PROJECT_ALIAS':'${D3FT_ROOT}'}
 cmd=['srun','--jobid=250212','--overlap','--cpu-bind=none','--nodelist=hn48','-N1','-n1','--cpus-per-task=1','python','-c',code]
 r=subprocess.run(cmd,env=te,capture_output=True,text=True,timeout=60)
 if r.returncode:raise RuntimeError('hn48 allocation/probe unavailable '+r.stderr[-1000:])
 d=json.loads(r.stdout);d['time']=datetime.datetime.now().isoformat();dump(F/'last_exclusive_check.json',d);return d['passed']
try:
 status('waiting_complete_derivative_acceptance',submitted_GPU_steps=0)
 gates=['si_acceptance_refined.json','mg_mode_acceptance.json','mg_fd_acceptance.json']
 while True:
  ready=all((E/n).exists() and json.loads((E/n).read_text()).get('passed') is True for n in gates)
  if ready:break
  time.sleep(30)
 spent=0.;results=[];completed_N=[]
 if args.resume_launch_record:
  rec=json.loads(args.resume_launch_record.read_text());N=rec['N'];status('formal_timing_running',N=N,CPU_handover_only=True)
  rec=await_detached_record(args.resume_launch_record);spent=rec['GPU_hours'];out=Path(rec['command'][rec['command'].index('--output')+1]);ok=(out/'result.json').exists()
  rec['status']='completed' if ok else 'failed';dump(args.resume_launch_record,rec)
  if not ok:raise RuntimeError('Existing formal timing stage failed, partial repetitions retained')
  results.append(json.loads((out/'result.json').read_text()));completed_N.append(N)
 for N in [1,6,24,54]:
  if N in completed_N:continue
  status('waiting_hn48_exclusive',N=N,spent_GPU_hours=spent)
  while not exclusive_check():time.sleep(30)
  # Budget all development steps, including failed/slack upper bounds, before
  # adding this formal step. Solely observational CPU checks cost zero GPUh.
  development=0.
  for path in (E/'commands').glob('*.json'):
   rec=json.loads(path.read_text())
   if 'GPU_hours' in rec:development+=rec['GPU_hours']
   elif 'start' in rec and 'end' in rec:development+=max(0,(datetime.datetime.fromisoformat(rec['end'])-datetime.datetime.fromisoformat(rec['start'])).total_seconds()/3600)
  remaining=min(60-spent,120-development-spent-5)
  if remaining<=0:raise RuntimeError('E3 total/formal GPU budget exhausted before all Ns')
  estimate=estimated_seconds(N)
  if remaining*3600<estimate:raise RuntimeError('E3 estimated work exceeds remaining budget; no truncated measurement launched')
  safe_timeout=int(math.ceil(2.5*estimate/3600)*3600)
  tag=reserve_name(F,f'N{N}',output_parent=F,log_parent=F);tmp='/dev/shm/response_design_e3_formal_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
  indexpath=F/'run_index.json';index=json.loads(indexpath.read_text()) if indexpath.exists() else {};index[str(N)]=dict(tag=tag,output=str(F/tag));dump(indexpath,index)
  cmd=['srun','--jobid=250212','--overlap','--cpu-bind=none','--nodelist=hn48','-N1','-n1','--cpus-per-task=8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env','CUDA_VISIBLE_DEVICES=0','timeout',str(safe_timeout),cfg['python'],'-u',str(R/'validation/workflows/response_design/reverse_response/e3_formal_timing.py'),'--root',str(E),'--N',str(N),'--output',str(F/tag),'--budget-hours',str(remaining)]
  rec=dict(jobid=250212,node='hn48',gpu=0,N=N,command=cmd,start=datetime.datetime.now().isoformat(),status='running',exclusive_check=json.loads((F/'last_exclusive_check.json').read_text()),development_GPU_hours=development,estimated_seconds=estimate,timeout_seconds=safe_timeout,timeout_multiplier=safe_timeout/estimate,budget_stop_only_at_saved_axis_boundaries=True)
  dump(F/f'{tag}_launch.json',rec);status('formal_timing_running',N=N,spent_GPU_hours=spent)
  start=time.monotonic()
  with (F/f'{tag}.log').open('xb') as f:proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(F/f'{tag}_launch.json',rec);code=proc.wait()
  hours=(time.monotonic()-start)/3600;spent+=hours;rec.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=hours,total_GPU_hours=spent,status='completed' if code==0 else 'failed');dump(F/f'{tag}_launch.json',rec)
  if code==124:raise RuntimeError('Infrastructure timeout124; not a physical/timing-quality failure; partial repetitions retained')
  if code or not (F/tag/'result.json').exists():raise RuntimeError('Formal N stage interrupted; classify raw record before any failed-physics claim')
  results.append(json.loads((F/tag/'result.json').read_text()))
 dump(F/'acceptance.json',dict(passed=True,results=results,total_active_GPU_hours=spent,whole_node_reserved_GPU_hours=4*spent,scope='Complete prescribed Ns x3, one certified MgSiN2 state; no claims of universal scaling'))
 status('E3_formal_scaling_complete',passed=True,spent_GPU_hours=spent)
 path=P/'paper/response_design_EXPERIMENTS_20261014.md';s=path.read_text()
 # Replace only the E3 status cell, preserving other experiment rows.
 lines=s.splitlines();lines=[line.rsplit('|',2)[0]+'| **完成**：Si/Mg导数门及hn48独占N=1/6/24/54×3正式计时全部通过 |' if line.startswith('| E3 |') else line for line in lines];path.write_text('\n'.join(lines)+'\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3完成：完整导数门及hn48四种实际输入图×3正式重复通过；原始机时/编译/失败保留。\n')
 print('E3：完整导数与正式scaling PASS。\n证据：formal_timing_20261015/acceptance.json。\n活跃GPUh与四卡独占占用成本分别记录。',flush=True)
except Exception as exc:status('infrastructure_interrupted' if 'timeout124' in str(exc) or 'budget' in str(exc) else 'failed',error=str(exc),physical_failure_claim_allowed=False);raise
