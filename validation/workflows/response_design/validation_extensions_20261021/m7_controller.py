"""Detached M7 single-card job, fixed8GPUh budget, no automatic retries."""
import datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parents[3]
R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
D=Path(json.loads((R/'M7-A/current.json').read_text())['root'])
read=lambda f:json.loads(Path(f).read_text());p=read(D/'protocol.json');job=None
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
 f=Path(f);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def stage(name,**kw):
 dump(D/'status.json',dict(time=now(),stage=name,pid=os.getpid(),**kw));print('M7',name,kw,flush=True)
def ledger():
 used=0. if job is None else (job.get('end_epoch',time.time())-job['start_epoch'])/3600
 dump(D/'GPU_ledger.json',dict(time=now(),total_M7_GPU_hours=used,cap_GPU_hours=8,jobs=[] if job is None else [job]))
 return used
def snapshot():
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1',
      'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 for i in range(3):
  try:
   r=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=45)
   rows=[[int(float(v.strip())) for v in x.split(',')] for x in r.stdout.strip().splitlines()]
   assert len(rows)==8
   return dict(time=now(),node='hn35',jobid=250285,rows=rows,raw=r.stdout)
  except Exception:
   if i==2:raise
   time.sleep(5)
def main():
 global job
 a=read(D/'CPU_acceptance.json');assert a['passed']
 for file,h in a['source_SHA256'].items():assert sha(file)==h,('Frozen input/code changed',file)
 assert sha(D/'protocol.json')==read(D/'protocol_SHA256.json')['protocol.json']
 with (D/'manager_started.json').open('x') as f:json.dump(dict(time=now(),pid=os.getpid()),f)
 stage('waiting_GPU0_free_no_interruption')
 while True:
  snap=snapshot()
  if any(v[0]==0 and v[1:]==[0,0] for v in snap['rows']):break
  time.sleep(30)
 dump(D/'prelaunch_occupancy.json',snap)
 out=D/'run_strain6';assert not out.exists()
 rt=Path(p['runtime']['gamma']);env=os.environ.copy();env.update(p['environment']);tmp=f'/dev/shm/m7_strain6_{os.getpid()}'
 env.update(CUDA_VISIBLE_DEVICES='0',TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',
  PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['PYTHONPATH']=':'.join(map(str,[rt,rt/'validation/workflows/response_design',rt/'validation/workflows/response_design/reverse_response',rt/'validation/workflows/phase3',p['jrystal']]))
 for key in list(env):
  if key.startswith('D3FT_') and any(v in key for v in ['CHECKPOINT','RESUME','RESULTS_ROOT','YIELD']):env.pop(key)
 cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c4',
  'env','CUDA_VISIBLE_DEVICES=0','bash','-c',
  'mkdir -p "$TMPDIR"; printf "STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"',
  'worker','env','CUDA_VISIBLE_DEVICES=0','timeout','--signal=TERM','--kill-after=20s',str(p['timeout_seconds']),
  p['python'],str(W/'m7_strain_worker.py'),'--protocol',str(D/'protocol.json'),'--output',str(out)]
 job=dict(time=now(),start_epoch=time.time(),node='hn35',jobid=250285,gpu=0,gpu_count=1,
          timeout_seconds=p['timeout_seconds'],output=str(out),command=cmd)
 with (D/'worker.log').open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=P,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  job['pid']=proc.pid;dump(D/'launch.json',job);ledger();stage('single_GPU_strain6_design_running',worker_srun_pid=proc.pid)
  while proc.poll() is None:
   ledger();time.sleep(15)
 job.update(end_epoch=time.time(),returncode=proc.returncode);dump(D/'completion.json',job);used=ledger()
 from m7_compare import compare
 comparison=None
 if (out/'trajectory.json').exists():comparison=compare(p,out,used,D)
 if proc.returncode:raise RuntimeError(f'Native worker exit {proc.returncode}; inspect retained logs/partial trajectory; no retry')
 acceptance=read(out/'acceptance.json');assert acceptance['completed']
 assert used<=8
 return dict(completed=True,passed=bool(acceptance['target_reached']),
  outcome='target_reached' if acceptance['target_reached'] else 'completed_target_not_reached',
  acceptance=str(out/'acceptance.json'),comparison=str(D/'comparison.json'),
  trajectory_summary=acceptance['trajectory_summary'],comparison_summaries=comparison['summaries'])
if __name__=='__main__':
 try:summary=main()
 except Exception as ex:
  traceback.print_exc();summary=dict(completed=False,passed=False,outcome='stopped_failure_or_budget',error=repr(ex),
       traceback=traceback.format_exc(),no_physical_failure_claim_without_inspection=True)
 summary.update(time=now(),GPU_hours=ledger(),cap_GPU_hours=8)
 dump(D/'summary.json',summary);stage('terminal',**summary)
 dump(R/'M7-A/latest_summary.json',dict(root=str(D),summary=str(D/'summary.json')))
 line=f"\n- {now()} M7-A六应变公平对照终态：{summary['outcome']}，累计{summary['GPU_hours']:.5f}/8GPUh；最多5步、原2meV容差/0.003步长/0.03总量/物理门不变。记录`{D}/summary.json`与逐步acceptance/comparison.csv；不scancel、不改正式稿、失败/未达标保留。\n"
 if 'comparison_summaries' in summary:
  for x in summary['comparison_summaries']:
   line+=f"  {x['experiment']}: {x['accepted_updates']}步，gap={x['final_gap_eV']:.9f}eV，目标误差{x['final_target_error_meV']:+.6f}meV，GPUh={x['assigned_GPU_hours']}；\n"
 for name in ['paper/response_design_EXPERIMENTS_20261014.md','response_design_to_do_list.md']:
  with (P/name).open('a') as f:f.write(line)
 print('M7结束',summary['outcome'],'\nGPUh',summary['GPU_hours'],'/8\n记录',D,flush=True)
