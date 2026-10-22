"""Independent new AlP FD processes; reuse validated subpoints, never a PID.

Only the registered a0/B observables are calculated. Each new missing electronic
subpoint still uses 3000 steps and the same physical/numerical acceptance gates.
An eight-hour timeout can checkpoint a point, then a fresh unique process can
reuse its completed subpoints. No failed whole point is called complete.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
import numpy as np
import e1_extend_kmesh as e1
from run_identity import reserve_name
from fd_acceptance import component
P=e1.P;E=e1.E;X=E/'extension_20261015';OLD=X/'formal_validation'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();T=a.root
cfg=json.loads((T/'protocol.json').read_text());fd=json.loads((OLD/'FD_protocol.json').read_text());row=json.loads((X/'response_row_k12.json').read_text());base=json.loads((Path(row['source'])/'summary.json').read_text());spent=cfg['prior_GPU_hours'];completed={};certificates=[]
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(Path(q).read_text())
def dump(q,r):
 q=Path(q);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def status(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,jobid=250285,gpus=[2,3],spent_GPU_hours=spent,controller_pid=os.getpid(),**kw))
def note(s):e1.note(s)
def idle():
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 rows={int(x[0]):int(x[1]) for l in r.stdout.splitlines() if len(x:=l.split(','))==2};dump(T/'GPU_probe.json',dict(time=now(),rows=rows));return all(rows[g]<=3 for g in [2,3])
def npoints(q):return len(read(Path(q)/'points.json')) if q and (Path(q)/'points.json').exists() else 0
def run_point(axis,sign):
 global spent
 key=str(axis)+','+str(sign);reuse=cfg.get('reuse_subpoints',{}).get(key);attempt=0
 while True:
  if spent+16>250:raise RuntimeError('E1 250 GPUh cap cannot reserve next unchanged 8h two-card process; no new process launched')
  while not idle():status('waiting_GPU2_3',axis=axis,sign=sign);time.sleep(20)
  attempt+=1;h=fd['steps'][axis];vector=np.zeros(7);vector[axis]=sign*h;initial=base['a0_A']+row['da0_dp'][axis]*sign*h
  tag=reserve_name(T,f'k12_axis{axis}_'+('minus' if sign<0 else 'plus'),output_parent=T/'runs',log_parent=T);out=T/'runs'/tag;tmp='/dev/shm/'+tag
  env={**cfg['environment'],'D3FT_PHASE3_RESULTS_ROOT':str(T/'runs'),'PYTHONPATH':str(T/'runtime')+':'+cfg['environment']['PYTHONPATH'].split(':')[-1],'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax'}
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E1-fresh','env','CUDA_VISIBLE_DEVICES=2,3','timeout','28800',e1.PY,'-u',str(T/'runtime/validation/workflows/phase3/worker.py'),'--material','AlP','--parameters','base','--tag',tag,'--mode','evaluate','--bulk-only','--grid','64','--cutoff','50','--k','12','--epochs','3000','--empty','0','--strain-step','0.005','--a0',str(initial),'--kpool','--fused-report','--fixed-mask',str(Path(row['source'])/'mask.npy'),'--vector',json.dumps(vector.tolist()),'--pressure-tol','1e-8']
  if reuse:cmd+=['--reuse-from',str(reuse)]
  rec=dict(start=now(),axis=axis,sign=sign,attempt=attempt,tag=tag,jobid=250285,gpus=[2,3],command=cmd,environment=env,output=str(out),reused_subpoints_from=reuse,timeout_seconds=28800,epochs=3000,independent_new_process=True,inherited_PID=None,status='running')
  f=(T/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(T/(tag+'_launch.json'),rec)
  while proc.poll() is None:status('new_FD_process_running',axis=axis,sign=sign,pid=proc.pid,attempt=attempt,elapsed_seconds=time.monotonic()-start);time.sleep(15)
  f.close();hours=2*(time.monotonic()-start)/3600;spent+=hours;rec.update(end=now(),returncode=proc.returncode,GPU_hours=hours,status='completed' if proc.returncode==0 else 'failed');dump(T/(tag+'_launch.json'),rec)
  if proc.returncode==124:
   previous=npoints(reuse);current=npoints(out)
   if current<=previous:raise RuntimeError('Timeout with no additional completed subpoint; stop rather than repeat without progress')
   note('E1 '+tag+'外部8h超时，已完成子点'+str(current)+'（前'+str(previous)+'）；旧FAIL保留，下一独立新名进程只复用严格匹配子点，步数/门不变。');reuse=str(out);continue
  if proc.returncode:raise RuntimeError('FD child '+tag+' failed rc='+str(proc.returncode)+'; immutable record retained')
  if ('DONE '+tag) not in (T/(tag+'.log')).read_text() or not (out/'summary.json').exists():raise RuntimeError('Missing DONE/summary despite child exit0')
  d=e1.derive(out);dump(out/'equilibrium_jacobian.json',d);summary=read(out/'summary.json');completed[key]=dict(folder=str(out),summary=summary);dump(T/'completed_points.json',completed);return summary
try:
 assert read(OLD/'QE_acceptance.json')['passed'];(T/'runs').mkdir(exist_ok=True)
 for axis,h in enumerate(fd['steps']):
  pair=[]
  for sign in [-1,1]:pair.append(run_point(axis,sign))
  checks=[component(key,float(g*2*h),pair[1][key]-pair[0][key],base['a0_A'],.02) for key,g in [('a0_A',row['da0_dp'][axis]),('B_GPa',row['dB_equilibrium_dp'][axis])]]
  roots=all(abs(r['equilibrium_pressure_GPa'])/(3*r['B_GPa'])<1e-7 for r in pair);certificate=dict(axis=axis,parameter=fd['parameter_order'][axis],step=h,rows=checks,root_passed=roots,passed=roots and all(c['joint_gate_passed'] for c in checks));certificates.append(certificate);dump(T/f'FD_axis{axis}.json',certificate)
  note('E1独立新进程 '+fd['parameter_order'][axis]+' a0/B原FD门 '+('PASS' if certificate['passed'] else 'FAIL')+'，不是四弹性量证书。')
  if not certificate['passed']:raise RuntimeError('Original FD gate failed axis '+str(axis))
 accepted=dict(row,accepted_for_calibration=True,independently_reoptimized_parameter_FD=True,parameter_FD_certificate=str(T/'acceptance.json'),QE_certificate=str(OLD/'QE_acceptance.json'));dump(T/'accepted_response_row_k12.json',accepted)
 result=dict(time=now(),passed=True,status='完成',response_row=str(T/'accepted_response_row_k12.json'),adopted_k=12,seven_independent_reoptimized_parameter_FD=certificates,total_GPU_hours_upper_bound=spent,old_SG15_rows_replaced=False,certified_observables=['a0_A','B_GPa']);dump(T/'acceptance.json',result);dump(E/'latest_acceptance.json',result);status('complete',passed=True)
except Exception as exc:status('failed',passed=False,error=str(exc));note('E1独立新进程队列停止：'+str(exc)+'；失败和子点全保留，不继承旧进程、不改门。');raise
finally:
 Q=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json';r=read(Q)
 for g in ['2','3']:
  if r.get(g,{}).get('status_file')==str(T/'status.json'):r.pop(g)
 dump(Q,r)
