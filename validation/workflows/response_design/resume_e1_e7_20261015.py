"""Timeout recovery and disjoint E7 fine controls, under existing budgets.

Phase 1: AlP FD on 2/3 and BN fine on 4/6 concurrently.
Phase 2: diamond fine on 2/3/4/6 (its registered grid needs four cards).
Phase 3: remaining AlP FD pairs on those four cards. No changes to physics.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse, datetime, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import e1_extend_kmesh as e1
import e7_ct_pilot as e7
from run_identity import reserve_name,await_detached_record
from fd_acceptance import component
P=e1.P; E=e1.E; EXT=E/'extension_20261015'; OLD=EXT/'formal_validation'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--E1-only-resume',type=Path);a=ap.parse_args();T=a.root
cfg=json.loads((T/'protocol.json').read_text());protocol=json.loads((OLD/'FD_protocol.json').read_text())
row=json.loads((EXT/'response_row_k12.json').read_text());baseline=json.loads((Path(row['source'])/'summary.json').read_text())
spent=cfg['E1_prior_GPU_hours'];launches=[];results={};certs=[]
e7.priority_pending=lambda:None
def now():return datetime.datetime.now().isoformat()
def load(q):return json.loads(Path(q).read_text())
def dump(q,r):
 q=Path(q);tmp=q.with_name(q.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def status(stage,**kw):
 dump(T/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,E1_GPU_hours=spent,**kw))
def note(s):e1.note(s)
def idle(gs):
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 rows={int(x[0]):int(x[1]) for line in r.stdout.splitlines() if len(x:=line.split(','))==2};return all(rows[g]<=3 for g in gs)
def await_cards(gs):
 while not idle(gs):status('waiting_owned_cards',gpus=gs);time.sleep(20)
def point(axis,sign,gs):
 global spent
 if spent+len(gs)*28800/3600>250:raise RuntimeError('E1 250 GPUh cap cannot reserve an unchanged eight-hour stage; no new stage launched')
 await_cards(gs);h=protocol['steps'][axis];vector=np.zeros(7);vector[axis]=sign*h
 initial=baseline['a0_A']+row['da0_dp'][axis]*sign*h
 tag=reserve_name(T,f'k12_axis{axis}_'+('minus' if sign<0 else 'plus'),output_parent=T/'runs',log_parent=T)
 record=load(OLD/'k12_axis0_minus_launch.json');env=dict(record['environment']);env['D3FT_PHASE3_RESULTS_ROOT']=str(T/'runs')
 tmp='/dev/shm/'+tag;env.update(TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax')
 runtime=Path(load(E/'protocol.json')['runtime']);mask=Path(row['source'])/'mask.npy'
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c'+str(8 if len(gs)==2 else 16),'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e1-retry','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gs)),'timeout','28800',e1.PY,'-u',str(runtime/'validation/workflows/phase3/worker.py'),'--material','AlP','--parameters','base','--tag',tag,'--mode','evaluate','--grid','64','--cutoff','50','--k','12','--epochs','3000','--empty','0','--strain-step','0.005','--a0',str(initial),'--kpool','--fused-report','--fixed-mask',str(mask),'--vector',json.dumps(vector.tolist()),'--pressure-tol','1e-8']
 rec=dict(start=now(),jobid=250285,gpus=gs,axis=axis,sign=sign,tag=tag,command=cmd,environment=env,status='running',timeout_seconds=28800,epochs=3000,physics_and_gates_unchanged=True)
 f=(T/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(T/(tag+'_launch.json'),rec)
 while proc.poll() is None:
  dump(T/'E1_progress.json',dict(time=now(),axis=axis,sign=sign,gpus=gs,pid=proc.pid,elapsed_seconds=time.monotonic()-start));time.sleep(15)
 f.close();hours=len(gs)*(time.monotonic()-start)/3600;spent+=hours;rec.update(end=now(),returncode=proc.returncode,GPU_hours=hours,status='completed' if proc.returncode==0 else 'failed');dump(T/(tag+'_launch.json'),rec);launches.append(rec)
 if proc.returncode:raise RuntimeError('E1 stage '+tag+' returncode '+str(proc.returncode)+'; retained, not reclassified as physical failure')
 folder=T/'runs'/tag;derived=e1.derive(folder);dump(folder/'equilibrium_jacobian.json',derived);summary=load(folder/'summary.json');results[(axis,sign)]=dict(summary=summary,folder=str(folder));dump(T/'FD_points_index.json',{str(k):v for k,v in results.items()});return folder
def fine(material,gs,seconds):
 coarse=e7.E/'runs'/e7.selected_tag(material,'cheap')
 if not e7.load(coarse/'native_acceptance.json').get('passed'):raise RuntimeError('E7 cheap native gate absent '+material)
 work=e7.run(material,'fine',gs,seconds,e7.load(coarse/'summary.json')['a0_A'],run_tag=material+'_fine_recovery')
 if work is not None and e7.load(work/'native_acceptance.json').get('passed'):e7.compare(material)
 e7.report();return str(work) if work else None
def reserve_handoff():
 # BN ended before diamond is dispatched; release E7's own reservations so
 # available() can check the now-idle four cards. Other owners are untouched.
 Q=e7.E/'gpu_reservations.json';r=e7.load(Q)
 for key in ('2','3','4','6'):
  if r.get(key,{}).get('status_file')==str(T/'status.json'):r.pop(key)
 e7.dump(Q,r)
try:
 assert load(OLD/'QE_acceptance.json')['passed']
 assert load(OLD/'k12_axis0_minus_launch.json')['returncode']==124
 (T/'runs').mkdir(exist_ok=True);status('E1_two_card_and_BN_two_card_running')
 if a.E1_only_resume:
  status('observing_unchanged_E1_GPU_child',record=str(a.E1_only_resume),gpus=[2,3])
  inherited=await_detached_record(a.E1_only_resume);spent+=inherited['GPU_hours'];dump(T/'inherited_E1_completion.json',inherited)
  folder=a.E1_only_resume.parent/'runs'/inherited['tag'];log=a.E1_only_resume.parent/(inherited['tag']+'.log')
  if not (folder/'summary.json').exists() or ('DONE '+inherited['tag']) not in log.read_text():raise RuntimeError('Inherited E1 child ended without worker DONE/summary; failure preserved')
  e1.derive(folder);results[(0,-1)]=dict(summary=load(folder/'summary.json'),folder=str(folder))
 else:
  with ThreadPoolExecutor(max_workers=2) as pool:
   first=pool.submit(point,0,-1,[2,3]);bn=pool.submit(fine,'BN',[4,6],21600)
   while not (first.done() and bn.done()):status('E1_and_BN_concurrent',E1_finished=first.done(),BN_finished=bn.done());time.sleep(15)
   initial_error=None
   try:first.result()
   except Exception as exc:initial_error=str(exc)
   try:bn_result=bn.result()
   except Exception as exc:bn_result=dict(error=str(exc))
   dump(T/'first_wave_results.json',dict(E1_error=initial_error,BN_result=bn_result))
  reserve_handoff();status('diamond_four_card_fine');fine('diamond',[2,3,4,6],28800)
  if initial_error:raise RuntimeError(initial_error)
 status('remaining_E1_FD_running')
 for axis,h in enumerate(protocol['steps']):
  for sign in [-1,1]:
   if (axis,sign) not in results:point(axis,sign,cfg.get('remaining_E1_gpus',[2,3,4,6]))
  pair=[results[(axis,sign)]['summary'] for sign in [-1,1]]
  checks=[component(key,float(gradient*2*h),pair[1][key]-pair[0][key],baseline['a0_A'],.02) for key,gradient in [('a0_A',row['da0_dp'][axis]),('B_GPa',row['dB_equilibrium_dp'][axis])]]
  root=all(abs(r['equilibrium_pressure_GPa'])/(3*r['B_GPa'])<1e-7 for r in pair)
  cert=dict(axis=axis,parameter=row['parameter_order'][axis],step=h,rows=checks,root_passed=root,passed=root and all(r['joint_gate_passed'] for r in checks));certs.append(cert);dump(T/f'FD_axis{axis}.json',cert)
  note('E1新超时分支参数'+row['parameter_order'][axis]+'原FD门 '+('PASS' if cert['passed'] else 'FAIL')+'，旧124记录保留。')
  if not cert['passed']:raise RuntimeError('Original FD numerical/physical gate failed axis '+str(axis))
 accepted=dict(row,accepted_for_calibration=True,independently_reoptimized_parameter_FD=True,parameter_FD_certificate=str(T/'acceptance.json'),QE_certificate=str(OLD/'QE_acceptance.json'))
 dump(T/'accepted_response_row_k12.json',accepted);result=dict(time=now(),passed=True,status='完成',adopted_k=12,response_row=str(T/'accepted_response_row_k12.json'),seven_independent_reoptimized_parameter_FD=certs,QE_certificate=str(OLD/'QE_acceptance.json'),total_GPU_hours_upper_bound=spent,old_SG15_rows_replaced=False)
 dump(T/'acceptance.json',result);dump(E/'latest_acceptance.json',result);status('complete',passed=True);note('E1超时恢复分支完整七参数FD和既有QE/k/力门PASS，新响应行独立保存。')
except Exception as exc:
 status('failed',error=str(exc),passed=False);note('E1/E7恢复队列停止：'+str(exc)+'；原超时和新失败均保留，不放宽门，不取消其他计算。');raise
