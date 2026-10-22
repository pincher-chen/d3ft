"""Fresh D3FT/QE baseline band path on GPU0 after E4 audit, no figures."""
import argparse,datetime as dt,fcntl,os,subprocess,time
from pathlib import Path
import numpy as np
from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from e1_three_parameter_queue import read,dump

def main(T):
 os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY));c=read(T/'protocol.json');P=Path(c['project']);Y=P/'validation/results/response_design_20261015_e7_ct_pilot';Q=Y/'gpu_reservations.json';child=None;GPU=c.get('fixed_gpu',0)
 def now():return dt.datetime.now().astimezone().isoformat()
 def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,jobid=250285,gpus=[GPU],controller_pid=os.getpid(),**kw))
 def note(s):
  for p in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
   with p.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
 def claim():
  if c.get('wait_for_E4',True):
   i=read(P/'validation/results/response_design_20261016_e4_endpoint_audit/active_audit.json');audit=Path(i['root']);s=read(audit/'status.json')
   if s.get('stage') not in ['complete','failed'] or not ((audit/'acceptance.json').exists() or list(audit.glob('*/reoptimized/readout.json'))):return False
  r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45)
  memory={int(v[0]):int(v[1]) for line in r.stdout.splitlines() if len(v:=line.split(','))==2}
  if memory.get(GPU,10**9)>3:return False
  with (Y/'gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   if str(GPU) in owners:return False
   owners[str(GPU)]=dict(owner='E8 D3FT/QE fresh baseline path',controller_pid=os.getpid(),status_file=str(T/'status.json'));dump(Q,owners)
  return True
 try:
  state('waiting_dependency_or_requested_GPU')
  while not claim():state('waiting_dependency_or_requested_GPU');time.sleep(30)
  if c['timeout_seconds']/3600>c['GPU_budget']:raise ValueError('E8 budget reservation failed')
  base=read(P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json')['environment'];tmp='/dev/shm/'+T.name;env={**os.environ,**base,'PYTHONPATH':c['runtime']+':'+base['PYTHONPATH'].split(':')[-1],'response_E8_RUNTIME':c['runtime'],'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax','PYTHONPYCACHEPREFIX':tmp+'/pycache','D3FT_TRACE_STAGES':'1'}
  out=T/'D3FT_start';cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E8-path','env','CUDA_VISIBLE_DEVICES='+str(GPU),'timeout',str(c['timeout_seconds']),c['python'],'-u',str(T/'path_worker.py'),str(T/'protocol.json'),'start',str(out),__import__('json').dumps(list(range(61)))]
  start=time.monotonic();log=(T/'D3FT_path.log').open('xb');child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record=dict(time=now(),pid=child.pid,jobid=250285,gpus=[GPU],command=cmd,status='running',output=str(out),timeout_seconds=c['timeout_seconds']);dump(T/'GPU_launch.json',record);note('E8独立D3FT 61点母相参照能带已接250285/GPU'+str(GPU)+'，复用本仓库合格起点密度，与新QE同几何路径比较；不是E6 k4生产，不改图。')
  while child.poll() is None:state('D3FT_path_running',pid=child.pid,completed_points=len(list(out.glob('k_*.json'))),elapsed_seconds=time.monotonic()-start);time.sleep(20)
  log.close();record.update(end=now(),returncode=child.returncode,GPU_hours=(time.monotonic()-start)/3600,status='completed' if child.returncode==0 else 'failed');dump(T/'GPU_launch.json',record)
  if child.returncode:raise RuntimeError('E8 path child failed/timeout; accepted partial k records retained')
  rows=[read(out/f'k_{i:04d}.json') for i in range(61)];dft=np.asarray([row['eigenvalues_Ha'] for row in rows]);qe=np.load(c['qe_bands'])
  import hashlib
  if hashlib.sha256(Path(c['qe_bands']).read_bytes()).hexdigest()!=c['qe_bands_sha256']:raise ValueError('QE path archive changed')
  if dft.shape!=qe['energies_Ha'].shape or np.max(abs(qe['k_fractional']-np.asarray(c['k_fractional'])))>1e-8:raise ValueError('Band path/order/shape mismatch')
  HA=27.211386245988;left=(dft-dft[:,47].max())*HA;right=(qe['energies_Ha']-qe['energies_Ha'][:,47].max())*HA;delta=left-right
  comparison=dict(time=now(),completed=True,all61_D3FT_spectra_completed=True,QE_SCF_NSFC_completed=True,points=61,bands=56,SCF_sampling=[1,1,1],not_k_convergence=True,
    path_gap_D3FT_eV=float((dft[:,48].min()-dft[:,47].max())*HA),path_gap_QE_eV=float((qe['energies_Ha'][:,48].min()-qe['energies_Ha'][:,47].max())*HA),
    VBM_aligned_all_bands_MAE_eV=float(abs(delta).mean()),VBM_aligned_all_bands_max_abs_eV=float(abs(delta).max()),VBM_aligned_frontier44_55_MAE_eV=float(abs(delta[:,44:56]).mean()),
    maximum_D3FT_ritz_Ha=max(max(row['residuals_Ha']) for row in rows),numerical_agreement_certified=False,agreement_threshold=None,
    caveat='Independent own VBM reference; QE |G+k| vs D3FT fixed |G| basis; prestrained P1 geometry with parent Pna21 path labels. No fitted offset beyond energy-origin alignment; no experimental claim.',GPU_hours=record['GPU_hours'])
  dump(T/'comparison.json',comparison);np.savez_compressed(T/'aligned_bands_comparison.npz',D3FT_eV=left,QE_eV=right,k_fractional=c['k_fractional'],difference_eV=delta)
  state('E8_path_comparison_complete',comparison= comparison);note('E8 D3FT/QE同几何61点能带数据与差异统计已完成；数值吻合阈值未预注册，不宣称跨代码验证PASS；尚未作图/修改论文。')
 except Exception as exc:state('failed',passed=False,error=str(exc));note('E8接续停止：'+repr(exc)+'；保留失败与已完成k点，不记完整比较。');raise
 finally:
  if child is not None and child.poll() is None:child.wait()
  with (Y/'gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   if owners.get(str(GPU),{}).get('status_file')==str(T/'status.json'):owners.pop(str(GPU));dump(Q,owners)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
