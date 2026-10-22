"""Fixed GPU2..5 E5 pilot, after the current E3 FD wave; no hn48 dependency."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
from run_identity import reserve_name,unique_output_path,await_detached_record
import argparse,datetime,json,os,subprocess,time
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_distributed';A=E/'four_card_20261015';R=A/'runtime'
protocol=A/'mg_k2_protocol.json';cfg=json.loads(protocol.read_text())
PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
env=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment'];env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
ap=argparse.ArgumentParser();ap.add_argument('--resume-scf-record',type=Path);args=ap.parse_args()
spent=0
def now():return datetime.datetime.now().isoformat()
def dump(q,d):
 t=q.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(q)
def status(stage,**kw):
 d=dict(time=now(),stage=stage,jobid=250285,gpus=[2,3,4,5],spent_GPU_hours=spent,**kw);dump(A/'queue_status.json',d);dump(E/'queue_status.json',d)
def gate(q):return Path(q).exists() and json.loads(Path(q).read_text()).get('passed') is True
def idle(gs):
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 rows={int(s.split(',')[0]):int(s.split(',')[1]) for s in r.stdout.splitlines()};return all(rows[g]<=3 for g in gs)
def launch(tag,stage,gpus,seconds,out,k=None):
 logical_tag=tag;tag=reserve_name(A,tag,log_parent=A)
 if stage!='spectrum':
  out=unique_output_path(out)
  cfg['fresh_scf_directory' if stage=='scf' else 'response_directory']=str(out);dump(protocol,cfg)
 if spent+len(gpus)*seconds/3600>48:raise RuntimeError('E5 pilot 48 GPUh cap')
 tmp='/dev/shm/response_design_e5_four_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-n1','-c'+str(8 if len(gpus)>1 else 4),'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'timeout',str(seconds),PY,'-u',cfg.get('driver_script',str(R/'validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py')),'--protocol',str(protocol),'--stage',stage,'--output',str(out)]
 if k is not None:cmd+=['--k-index',str(k)]
 rec=dict(time=now(),jobid=250285,gpus=gpus,command=cmd,environment={k:te[k] for k in env},output=str(out),status='running',tag=tag,logical_tag=logical_tag)
 f=(A/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(A/(tag+'_launch.json'),rec);print('E5_START',tag,gpus,proc.pid,flush=True);return proc,f,rec,start
def finish(job):
 global spent
 proc,f,rec,start=job;code=proc.wait();f.close();hours=len(rec['gpus'])*(time.monotonic()-start)/3600;spent+=hours;rec.update(end=now(),returncode=code,GPU_hours=hours,status='completed' if not code else 'failed');dump(A/(rec['tag']+'_launch.json'),rec)
 return code
def note(s):
 for q in (P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md'):
  with q.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
try:
 status('waiting_mode_and_CPU_gate_for_independent_SCF',requires_hn48_start=False)
 while not all(gate(q) for q in cfg.get('scf_required_acceptance_files',cfg['required_acceptance_files'])):
  time.sleep(20)
 scf_gpus=cfg.get('scf_gpus',[2,3,4,5])
 if args.resume_scf_record:
  status('four_card_fresh_k2_SCF_running',CPU_handover_only=True)
  rec=await_detached_record(args.resume_scf_record);spent=rec['GPU_hours'];out=Path(rec['output']);cfg['fresh_scf_directory']=str(out)
  terminal=json.loads((out/'status.json').read_text());passed=terminal.get('stage')=='completed' and terminal.get('passed') is True
  rec['status']='completed' if passed else 'failed';dump(args.resume_scf_record,rec)
  if not passed:raise RuntimeError('Existing k2 SCF did not pass; no new response launched')
 else:
  while not idle(scf_gpus):time.sleep(20)
  status('four_card_fresh_k2_SCF_running')
  if finish(launch('mg_k2_scf','scf',scf_gpus,10800,Path(cfg['fresh_scf_directory']))):raise RuntimeError('Fresh Mg k2 SCF failed')
 dump(A/'scf_acceptance.json',dict(time=now(),passed=True,source=cfg['fresh_scf_directory'],scope='Fresh k2 SCF residual/ASR/charge only; spectrum/response still pending'))
 note('E5四卡新Mg k2 SCF物理门PASS，E6独立静水重优化可接续；并非谱/响应已完成。')
 status('waiting_complete_FD_gate_before_spectrum_response')
 while not all(gate(q) for q in cfg['required_acceptance_files']):
  fd=P/'validation/results/response_design_20261015_e3_geometry/mg_fd_queue_status.json'
  if fd.exists() and json.loads(fd.read_text()).get('stage')=='failed':raise RuntimeError('E3 FD gate failed; new SCF retained, no response claimed')
  time.sleep(20)
 while not idle([2,3,4,5]):time.sleep(20)
 status('four_card_local_spectra_running')
 cfg['spectrum_directories']=[str(unique_output_path(Path(cfg['spectrum_root'])/f'k{k}')) for k in range(8)];dump(protocol,cfg)
 for start in (0,4):
  if spent+4+12>48:raise RuntimeError('Insufficient pilot spectrum/response budget')
  jobs=[launch('spectrum_k'+str(k),'spectrum',[g],3600,Path(cfg['spectrum_directories'][k]),k) for k,g in zip(range(start,start+4),[2,3,4,5])]
  codes=[finish(j) for j in jobs]
  if any(codes):raise RuntimeError('Local spectrum failed, no response acceptance')
 status('four_card_coupled_adjoint_running')
 if finish(launch('mg_k2_response','response',[2,3,4,5],10800,A/'production/coupled_response')):raise RuntimeError('Coupled response failed')
 resultfile=Path(cfg['response_directory'])/'result.json';result=json.loads(resultfile.read_text());dump(A/'response_index.json',dict(result_file=str(resultfile)))
 if not result['passed']:raise RuntimeError('Coupled response mode gate failed')
 status('E5_four_card_pilot_complete_E4_deferred',passed=True)
 note('E5四卡Mg k2耦合响应pilot PASS；本轮E4按最新优先级暂不追加，E6固定GPU6/7接续。')
 print('E5 四卡k2 pilot PASS。\n新SCF/8点谱/耦合伴随及三个切向通过。\nMg mesh FD/k4和E4仍独立待验。',flush=True)
except Exception as exc:
 status('failed',passed=False,error=str(exc));note('E5四卡pilot失败：'+str(exc)+'；失败保留，不伪造下一阶段门。');raise
