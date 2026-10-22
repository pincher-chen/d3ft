"""One four-GPU synthetic layout gate only, no productionMg calculations."""
import datetime,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parents[3];R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root']);Q=Path(json.loads((R/'M5-A/current_sharded.json').read_text())['root']);H=Path(json.loads((Q/'precheck_hn48_current.json').read_text())['root']);p=json.loads((Q/'protocol.json').read_text());probe=json.loads((H/'protocol.json').read_text());read=lambda f:json.loads(Path(f).read_text());start=None;end=None

def now():return datetime.datetime.now().astimezone().isoformat()
def dump(f,d):
 f=Path(f);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(f)
def ledger():
 hours=0. if start is None else 4*((end or time.time())-start)/3600
 value={'time':now(),'GPU_hours':hours,'gpu_count':4,'node':'hn48','jobid':275669,'historical_M5_GPU_hours':p['historical_GPU_hours'],'total_M5_GPU_hours':p['historical_GPU_hours']+hours,'cap':35,'start_epoch':start,'end_epoch':end};dump(H/'GPU_ledger.json',value)
 # No M5productionisrunning: theCPUqueuewaitsM2. Keep an effective ledger
 # without modifying old experiment or failure cost records.
 effective=read(R/'GPU_ledger.json');c=Path(read(R/'M3-A/current_continuation.json')['root'])/'GPU_ledger.json';effective['GPU_hours']['M3-A']=read(c)['total_M3_GPU_hours'];effective['GPU_hours']['M5-A']=value['total_M5_GPU_hours'];effective['time']=now();effective['M5_hn48_probe_ledger']=str(H/'GPU_ledger.json');dump(R/'GPU_ledger_effective.json',effective);return value

def stage(name,**kw):dump(H/'status.json',{'time':now(),'stage':name,**kw});print('HN48_LAYOUT',name,kw,flush=True)
if __name__=='__main__':
 with (H/'manager_started.json').open('x') as f:json.dump({'pid':os.getpid(),'time':now()},f)
 ledger();stage('waiting_CPU_reference')
 try:
  cpu=H/'cpu_reference/acceptance.json';limit=time.monotonic()+300
  while not cpu.exists():
   if time.monotonic()>limit:raise RuntimeError('CPUreference notready within300s; noGPUlaunch')
   time.sleep(5)
  d=read(cpu)
  if not d['passed'] or d['backend']!='cpu' or d['devices']!=4 or d['nk']!=32:raise ValueError('CPU32k/4device gatefailed')
  if hashlib.sha256(Path(probe['script']).read_bytes()).hexdigest()!=probe['script_SHA256']:raise ValueError('Fixturechanged')
  for f,sha in read(Q/'CPU_acceptance.json')['source_SHA256'].items():
   if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=sha:raise ValueError('CPU-tested corechanged '+f)
  stage('waiting_hn48_assigned4idle')
  while True:
   try:
    a=subprocess.run(['srun','--jobid=275669','--overlap','--nodelist=hn48','--cpu-bind=none','-N1','-n1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45,check=True);rows=[[int(float(x.strip())) for x in row.split(',')] for row in a.stdout.strip().splitlines()];assert len(rows)==4
    if all(x[0] in range(4) and x[1:]==[0,0] for x in rows):break
   except Exception as ex:print('GPUqueryretry',repr(ex),flush=True)
   time.sleep(15)
  dump(H/'prelaunch_occupancy.json',{'time':now(),'node':'hn48','jobid':275669,'rows':rows})
  if p['historical_GPU_hours']+4*(probe['timeout_seconds']+25)/3600>35:raise RuntimeError('Shared35GPUhbudgetwouldexceed')
  env=os.environ.copy();env.update(p['environment']);rt=Path(p['runtime_path']);tmp=f'/dev/shm/m5_hn48probe_{os.getpid()}';env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',JAX_PLATFORMS='cuda',TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env['PYTHONPATH']=':'.join([str(rt),str(rt/'validation/workflows/response_design/reverse_response'),str(rt/'validation/workflows/response_design'),str(rt/'validation/workflows/phase3'),p['jrystal']]);env.pop('D3FT_RESPONSE_SINGLE_DEVICE',None)
  output=H/'gpu_probe';assert not output.exists();cmd=['srun','--jobid=275669','--overlap','--nodelist=hn48','--cpu-bind=none','--gpu-bind=none','-N1','-n1','-c8','env','CUDA_VISIBLE_DEVICES=0,1,2,3','bash','-c','mkdir -p "$TMPDIR"; printf "STEP=%s.%s NODE=%s GPU=%s\\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"; exec "$@"','worker','timeout','--signal=TERM','--kill-after=20s',str(probe['timeout_seconds']),p['python'],probe['script'],'--backend','gpu','--reference',H/'cpu_reference/reference.npz','--output',output];cmd=list(map(str,cmd));start=time.time();stage('actual_four_GPU_layout_probe')
  with (H/'GPU_probe.log').open('x') as log:
   proc=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);dump(H/'GPU_launch.json',{'time':now(),'pid':proc.pid,'command':cmd,'start_epoch':start,'gpu_count':4})
   while proc.poll() is None:time.sleep(5);ledger()
   end=time.time();ledger();dump(H/'GPU_completion.json',{'time':now(),'returncode':proc.returncode,'end_epoch':end})
  if proc.returncode:raise RuntimeError('ActualGPUlayoutprobe failed; retainfailure, nofull64kproduction')
  result=read(output/'acceptance.json')
  if not result['passed']:raise RuntimeError('ActualGPUacceptance false')
  certificate={'time':now(),'passed':True,'node':'hn48','jobid':275669,'devices':4,'nk':32,'k_per_device':8,'CPU_reference_SHA256':hashlib.sha256((H/'cpu_reference/reference.npz').read_bytes()).hexdigest(),'GPU_acceptance':str(output/'acceptance.json'),'GPU_acceptance_SHA256':hashlib.sha256((output/'acceptance.json').read_bytes()).hexdigest(),'fixture_script_SHA256':probe['script_SHA256'],'production_core_SHA256':read(Q/'CPU_acceptance.json')['source_SHA256'],'errors':result['errors'],'inner_comparison_count':result['inner_comparison_count'],'GPU_hours':ledger()['GPU_hours'],'scope':'Small32k syntheticfourGPUnotMg64k/nottiming; existing8CPU64kpass; fullMgfirst2PCG/primal/memorychecksremain'};dump(H/'certificate.json',certificate);stage('complete',passed=True,GPU_hours=certificate['GPU_hours']);outcome='passed'
 except Exception as ex:
  traceback.print_exc();dump(H/'failure.json',{'time':now(),'passed':False,'error':repr(ex),'no_full64k_started':True,'GPU_hours':ledger()['GPU_hours']});stage('failed',error=repr(ex));outcome='failed'
 line=f"\n- {now()} M5 hn48四卡小规模布局验收{outcome}：32合成k/27G/2占据、每卡8块，非Mg64k/非计时；本次{ledger()['GPU_hours']:.5f}GPUh计入35，旧5.34246保留；记录`{H}`。通过则M2结束+hn35全空直接64k，不重复8卡合成验收；失败不绕过门。\n"
 for name in ['response_design_to_do_list.md','paper/response_design_EXPERIMENTS_20261014.md']:
  with (P/name).open('a') as f:f.write(line)
 print('M5四卡布局',outcome,'\nGPUh',ledger()['GPU_hours'],'\n记录',H,flush=True)
