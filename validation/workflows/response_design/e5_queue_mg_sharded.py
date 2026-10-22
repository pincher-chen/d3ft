"""CPU deferred queue: E3 timing first, then hn35 k2 response, then E4.

No running GPU task is signalled or cancelled. E1's gate-only CPU observer can
be held by the user-authorized resource-order handoff, and is resumed on every
exit of this pilot (pass or failure). All steps use fixed allocation 250285.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import datetime,json,os,signal,subprocess,time
P=Path(__file__).resolve().parents[3];A=P/'validation/results/response_design_20261015_e5_distributed';p=json.loads((A/'mg_k2_protocol.json').read_text())
R=A/'runtime';PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python';script=R/'validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py'
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment']
env={**os.environ,**base};env['PYTHONPATH']=str(R)+':${D3FT_ROOT}/third_party/jrystal_dependency';spent=0.;history=[]
def now():return datetime.datetime.now().isoformat()
def dump(name,row):
 dest=A/name;tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(dest)
def status(stage,**kw):dump('queue_status.json',dict(time=now(),stage=stage,jobid=250285,spent_GPU_hours=spent,**kw))
def note(text):
 for doc in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
  with doc.open('a') as f:f.write('\n- '+now()+' '+text+'\n')
def gate(path):return Path(path).exists() and json.loads(Path(path).read_text()).get('passed') is True
def idle(gpus):
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits']
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError('GPU query failed')
 rows={int(line.split(',')[0]):int(line.split(',')[1]) for line in r.stdout.splitlines()}
 dump('latest_GPU_probe.json',dict(time=now(),rows=rows,selected=gpus));return all(rows[g]==0 for g in gpus)
def launch(tag,stage,gpus,seconds,output,k=None):
 if spent+len(gpus)*seconds/3600>48+1e-9:raise RuntimeError('E5 pilot hard budget would be exceeded')
 tmp='/dev/shm/response_design_e5_mg_'+tag;localenv={**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task='+str(16 if len(gpus)>1 else 4),'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'timeout',str(seconds),PY,'-u',str(script),'--protocol',str(A/'mg_k2_protocol.json'),'--stage',stage,'--output',str(output)]
 if k is not None:cmd+=['--k-index',str(k)]
 rec=dict(time=now(),command=cmd,environment={n:localenv[n] for n in base if n in localenv},jobid=250285,gpus=gpus,tag=tag,timeout_seconds=seconds,start_monotonic=time.monotonic(),output=str(output))
 log=(A/(tag+'.log')).open('xb');proc=subprocess.Popen(cmd,env=localenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(tag+'_launch.json',rec);return proc,log,rec
def finish(job):
 global spent
 proc,log,rec=job;code=proc.wait();log.close();used=len(rec['gpus'])*(time.monotonic()-rec['start_monotonic'])/3600;spent+=used
 rec.update(end=now(),returncode=code,GPU_hours=used);dump(rec['tag']+'_launch.json',rec);history.append(rec)
 if code:raise RuntimeError(rec['tag']+' failed; original output kept')
def resume_E1():
 f=A/'E1_CPU_observer_hold.json'
 if not f.exists():return
 r=json.loads(f.read_text());pid=r['pid'];cmdfile=Path(f'/proc/{pid}/cmdline')
 if cmdfile.exists() and b'e1_finish_extension.py' in cmdfile.read_bytes():
  os.kill(pid,signal.SIGCONT);r.update(resumed=now(),status='resumed');dump('E1_CPU_observer_hold.json',r)
  note('E5优先级队列释放hn35后，E1仅CPU门控观察器已恢复，后续QE/参数FD接续；从未取消或信号操作计算GPU任务。')
try:
 lockfd=os.open(A/'queue_controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(lockfd)
 status('waiting_E3_derivatives_and_formal_timing_priority',GPU_steps_submitted=0)
 while True:
  fg=Path(p['priority_gate']);formal=json.loads(fg.read_text()) if fg.exists() else {}
  if formal.get('stage')=='failed':raise RuntimeError('E3 formal priority queue failed; do not bypass')
  if all(gate(f) for f in p['required_acceptance_files']) and formal.get('stage') in p['priority_allowed_stages']:break
  time.sleep(20)
 status('waiting_eight_idle_hn35_cards',GPU_steps_submitted=0)
 while not idle(list(range(8))):time.sleep(20)
 status('fresh_k2_SCF_running');finish(launch('mg_k2_scf','scf',list(range(8)),7200,Path(p['fresh_scf_directory'])))
 note('E5新MgSiN2 k2 SCF结束，独立新态/物理门已保存，谱与耦合响应仍未完成。')
 status('eight_local_spectra_running')
 # Concurrent tasks each have a one-card, one-hour cap: combined wave <=8h.
 if spent+8+24>48:raise RuntimeError('Insufficient pilot budget for registered spectrum/response caps')
 jobs=[launch('mg_spectrum_k'+str(k),'spectrum',[k],3600,Path(p['spectrum_root'])/f'k{k}',k) for k in range(8)]
 failures=[]
 for job in jobs:
  try:finish(job)
  except Exception as exc:failures.append(str(exc))
 if failures:raise RuntimeError('; '.join(failures))
 status('coupled_adjoint_running');finish(launch('mg_k2_response','response',list(range(8)),10800,A/'production/coupled_response'))
 result=json.loads((A/'production/coupled_response/result.json').read_text())
 if not result['passed']:raise RuntimeError('Mg k2 response mode gate failed')
 status('Mg_k2_response_pilot_complete',passed=True,FD_and_design_complete=False,history=history)
 note('E5新MgSiN2 k2分片响应pilot完成：完整54梯度与三固定切向方向/平移门通过；尚未完成Mg混合FD、mesh设计与k4转移。')
 print('E5 MgSiN2 k2多卡响应pilot完成。\n真实新SCF/八点谱/公共密度耦合与方向门通过。\n设计、FD与k4验证仍单独待做。',flush=True)
 resume_E1()
 # E4 only uses GPU0, disjoint from E1 followup [2,3,4,6]. The controller's
 # physical dry-run proof is a required gate, never a paper-quality endpoint.
 e4=P/'validation/results/response_design_20261015_e4_joint_design';dry=e4/'si_cpu_dry_v1/result.json'
 if not dry.exists() or json.loads(dry.read_text()).get('completed_updates',0)<1:raise RuntimeError('E4 real CPU dry run missing')
 while not idle([0]):time.sleep(20)
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=8','env','CUDA_VISIBLE_DEVICES=0','timeout','36000',PY,'-u',str(R/'validation/workflows/response_design/joint_design_dft.py'),'--protocol',str(e4/'mg_protocol.json'),'--output',str(e4/'production_mg_v1')]
 rec=dict(start=now(),jobid=250285,gpus=[0],command=cmd,timeout_seconds=36000,E4_hard_budget_GPU_hours=100)
 with (e4/'production_mg_v1.log').open('xb') as log:
  start=time.monotonic();proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump_e4=e4/'production_mg_v1_launch.json';dump_e4.write_text(json.dumps(rec,indent=2)+'\n');code=proc.wait()
 rec.update(end=now(),returncode=code,GPU_hours=(time.monotonic()-start)/3600);dump_e4.write_text(json.dumps(rec,indent=2)+'\n')
 note('E4 Mg54联合设计后台结束，退出码'+str(code)+'；逐态/拒绝/实际更新记录保存，不以退出码代替设计成功。')
 if code:raise RuntimeError('E4 production driver failed; failure records retained')
 design=json.loads((e4/'production_mg_v1/result.json').read_text())
 acceptance=dict(time=now(),passed=design['completed_updates']>0 and design['target_reached'],result=design,
                 scope='Three-update registered Gamma joint design example; transfer/band comparison separate',GPU_hours=rec['GPU_hours'])
 (e4/'production_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
 note('E4 Mg54实际设计 '+('PASS（目标与全部逐态门）' if acceptance['passed'] else '未达目标/未完成')+'；不冒称结构已弛豫。')
 print('E4 Mg54设计后台结束：'+('PASS' if acceptance['passed'] else '未达完整目标')+'。\n真实轨迹与拒绝记录保留。\n结果见production_acceptance.json。',flush=True)
except Exception as exc:
 status('failed',passed=False,error=str(exc));note('E5/E4优先级队列停止追加：'+str(exc)+'；既有计算自然结束，失败保留。')
 raise
finally:resume_E1()
