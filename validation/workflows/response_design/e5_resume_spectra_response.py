"""Durable fixed-card continuation: accepted local spectra -> coupled response.

Every child is waited and recorded by this detached controller. Heartbeats
distinguish compilation, active children, waiting cards and terminal failures.
No GPU task is signalled. Completed spectra are reused only after hash gates.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse, datetime, hashlib, json, os, subprocess, time
from pathlib import Path
from run_identity import reserve_name
P=Path(__file__).resolve().parents[3]
E=P/'validation/results/response_design_20261015_e5_distributed'; A=E/'four_card_20261015'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();T=args.root
p=json.loads((T/'protocol.json').read_text());R=Path(p['runtime']);spent=p['prior_GPU_hours']
PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
env=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment'];env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(Path(q).read_text())
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
def dump(q,row):
 q=Path(q);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def status(stage,**kw):
 row=dict(time=now(),stage=stage,jobid=250285,gpus=[0,5],controller_pid=os.getpid(),spent_GPU_hours=spent,root=str(T),**kw)
 for f in [T/'status.json',E/'queue_status.json',A/'queue_status.json']:dump(f,row)
 print(json.dumps(row),flush=True)
def note(s):
 for f in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
  with f.open('a') as stream:stream.write('\n- '+now()+' '+s+'\n')
def idle(gs):
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 rows={int(x[0]):[int(y) for y in x[1:]] for line in r.stdout.splitlines() if len(x:=line.split(','))==3}
 dump(T/'GPU_probe.json',dict(time=now(),rows=rows,requested=gs));return all(rows[g][0]<=3 and rows[g][1]==0 for g in gs)
def await_cards(gs):
 deadline=time.monotonic()+21600
 while not idle(gs):
  status('waiting_fixed_cards',waiting_gpus=gs)
  if time.monotonic()>deadline:raise RuntimeError('Six-hour card wait limit; no GPU work cancelled')
  time.sleep(20)
def spectrum_gate(k):
 folder=Path(p['spectrum_directories'][k]);row=read(folder/'summary.json');end=read(folder/'status.json')
 if end.get('stage')!='completed' or not end.get('passed') or row['k_index']!=k or row['source_state_sha256']!=p['source_state_sha256'] or row['saved_orbitals_sha256']!=sha(folder/'orbitals.npy') or max(row['residuals_Ha'])>1e-9 or row['orthogonality_max']>1e-10:raise RuntimeError('Local spectrum acceptance failed k='+str(k))
 return dict(k=k,source=str(folder),residual_max_Ha=max(row['residuals_Ha']),orthogonality_max=row['orthogonality_max'])
def launch(stage,gs,k=None):
 logical='spectrum_k'+str(k) if stage=='spectrum' else 'coupled_adjoint_forward'
 tag=reserve_name(T,logical,output_parent=T,log_parent=T);out=T/tag
 if stage=='spectrum':p['spectrum_directories'][k]=str(out)
 else:p['response_directory']=str(out)
 dump(T/'protocol.json',p)
 seconds=3600 if stage=='spectrum' else 21600
 if spent+len(gs)*seconds/3600>p['pilot_GPU_hours_cap']:raise RuntimeError('Inclusive E5 pilot GPU-hour cap')
 tmp='/dev/shm/'+tag;te={**os.environ,**env,'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax','D3FT_TRACE_STAGES':'1'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c'+str(8 if len(gs)>1 else 4),'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5-resume','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gs)),'timeout',str(seconds),PY,'-u',str(R/'validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py'),'--protocol',str(T/'protocol.json'),'--stage',stage,'--output',str(out)]
 if k is not None:cmd+=['--k-index',str(k)]
 f=(T/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(time=now(),tag=tag,pid=proc.pid,jobid=250285,gpus=gs,command=cmd,environment={key:te[key] for key in [*env,'TMPDIR','PYTHONPYCACHEPREFIX','JAX_COMPILATION_CACHE_DIR','D3FT_TRACE_STAGES']},output=str(out),k_index=k,status='running',timeout_seconds=seconds)
 dump(T/(tag+'_launch.json'),rec)
 return dict(proc=proc,file=f,start=start,record=rec)
def wait_wave(jobs,stage):
 global spent
 pending=list(jobs);failures=[]
 while pending:
  status(stage,children=[dict(pid=j['proc'].pid,gpus=j['record']['gpus'],output=j['record']['output'],elapsed_seconds=time.monotonic()-j['start']) for j in pending])
  for job in list(pending):
   code=job['proc'].poll()
   if code is None:continue
   job['file'].close();rec=job['record'];hours=len(rec['gpus'])*(time.monotonic()-job['start'])/3600;spent+=hours
   rec.update(end=now(),returncode=code,GPU_hours=hours,status='completed' if code==0 else 'failed');dump(T/(rec['tag']+'_launch.json'),rec);pending.remove(job)
   if code:failures.append(rec['tag'])
  if pending:time.sleep(15)
 if failures:raise RuntimeError('Child failed, original outputs retained: '+','.join(failures))
try:
 src=Path(p['fresh_scf_directory']);point=read(src/'point.json')
 if sha(src/'state.pkl')!=p['source_state_sha256'] or not point['stationarity_pass'] or point['residual']>1e-8 or point['acoustic_sum_rule_Ha_Bohr']>5e-8 or abs(point['charge']-96)>1e-6:raise RuntimeError('SCF source/hash gate failed')
 for gate in p['required_acceptance_files']:
  if not read(gate).get('passed'):raise RuntimeError('Prior required gate failed')
 for k in p['adopted_k_indices']:spectrum_gate(k)
 todo=[k for k in range(8) if k not in p['adopted_k_indices']]
 for offset in range(0,len(todo),2):
  await_cards([0,5]);ks=todo[offset:offset+2]
  jobs=[launch('spectrum',[gpu],k) for k,gpu in zip(ks,[0,5])]
  wait_wave(jobs,'local_spectra_running')
  for k in ks:spectrum_gate(k)
  dump(T/'spectra_progress.json',dict(time=now(),completed=[spectrum_gate(k) for k in range(8) if (Path(p['spectrum_directories'][k])/'summary.json').exists()]))
 spectra=[spectrum_gate(k) for k in range(8)];dump(T/'spectra_acceptance.json',dict(passed=True,time=now(),spectra=spectra))
 note('E5八个局域谱全部PASS，自动接GPU0/5公共密度耦合adjoint与三个forward方向；不是FD/设计已完成。')
 await_cards([0,5]);wait_wave([launch('response',[0,5])],'coupled_adjoint_forward_running')
 result=read(Path(p['response_directory'])/'result.json')
 if not result.get('passed'):raise RuntimeError('Coupled response acceptance failed')
 index=dict(result_file=str(Path(p['response_directory'])/'result.json'),protocol=str(T/'protocol.json'),source_state_sha256=p['source_state_sha256'])
 dump(T/'response_index.json',index);dump(A/'response_index.json',index);dump(A/'mg_k2_protocol.json',p)
 status('coupled_modes_PASS_FD_design_pending',passed=True,FD_passed=False,design_completed=False,response_index=index)
 note('E5 k2耦合adjoint/三个forward及平移门PASS；独立重优化FD与k2设计仍待，不把模式一致当FD通过。')
except Exception as exc:
 status('failed',passed=False,error=str(exc));note('E5谱/响应接续失败：'+str(exc)+'；全部现场保留，未取消E6。');raise
finally:
 reservations=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json';rows=read(reservations)
 for key in ['0','5']:
  if rows.get(key,{}).get('status_file')==str(T/'status.json'):rows.pop(key)
 dump(reservations,rows)
