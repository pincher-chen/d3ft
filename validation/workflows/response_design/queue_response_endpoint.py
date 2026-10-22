"""Wait for reverse design and resource predecessors, then verify its endpoint mesh."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
P=Path('${D3FT_ROOT}')
D=P/'validation/results/response_design_20261004_response_design'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
 f=Path(f);t=f.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(f)
def matrix(v):return [[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]]
def run_mesh(root):
 cfg=read(root/'protocol.json');mesh=cfg['mesh'];job=cfg['jobid'];gpus=cfg['gpus']
 def stage(label,**kw):dump(root/'status.json',dict(stage=label,time=datetime.datetime.now().isoformat(),**kw))
 def run(cmd,name):
  dump(root/(name+'_command.json'),cmd)
  with (root/(name+'.log')).open('xb') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{name}: RC {result.returncode}')
 try:
  stage('waiting_dependencies')
  for _ in range(2160):
   rs=read(D/'reverse_status.json') if (D/'reverse_status.json').exists() else {}
   if rs.get('stage')=='failed' or rs.get('stage')=='completed' and not rs.get('converged'):
    stage('dependency_failed',reverse=rs);return
   if mesh==4:
    f=P/'validation/results/response_design_20261004_elastic_mode_k4/status.json';pred=read(f) if f.exists() else {}
    if pred.get('stage')=='failed':stage('dependency_failed',elastic=pred);return
    ready=pred.get('stage')=='completed'
   else:
    ready=all((D/'runs'/f'elastic_{sign}_half/point.json').exists() for sign in ['plus','minus'])
    if (D/'elastic_task_failure.json').exists():stage('dependency_failed',elastic=read(D/'elastic_task_failure.json'));return
   if rs.get('stage')=='completed' and rs.get('converged') and ready:break
   time.sleep(20)
  else:raise RuntimeError('Dependencies did not complete in registered 12-hour wait')
  base=Path(cfg['parent']);parent=read(base/'point.json');assert parent['stationarity_pass'] and parent['residual']<=1e-6
  theta=rs['theta'];source=root/'runs/endpoint';command=list(cfg['point_command']);command[command.index('--strain')+1]=json.dumps(matrix(theta))
  for _ in range(180):
   inv=subprocess.check_output(cfg['query']+['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
   usage={int(x.split(',')[0]):int(x.split(',')[1]) for x in inv.strip().splitlines()}
   if all(usage[g]<=100 for g in gpus):break
   time.sleep(20)
  else:raise RuntimeError('Expected GPUs remained occupied')
  stage('electronic_state',theta=theta);run(command,'ground')
  point=read(source/'point.json');assert point['stationarity_pass'] and point['residual']<=1e-6
  assert point['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(point['integrated_valence_density']-96)<=.00096
  assert len(point['k_fractional'])==mesh**3 and sha(source/'mask.npy')==sha(base/'mask.npy')
  assert read(source/'metadata.json')['code_sha256']==read(base/'metadata.json')['code_sha256']
  stage('spectra');workers=[]
  for s in cfg['shards']:
   with (root/(s['name']+'.log')).open('xb') as log:proc=subprocess.Popen(s['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   workers.append((s,proc))
  while any(proc.poll() is None for s,proc in workers):
   stage('spectra',points=sum(len(list(Path(s['output']).glob('k_*.json'))) for s,proc in workers),total=mesh**3)
   time.sleep(20)
  codes=[dict(name=s['name'],returncode=proc.returncode) for s,proc in workers];dump(root/'exit_codes.json',codes)
  if any(x['returncode'] for x in codes):raise RuntimeError('Endpoint spectrum worker failed')
  rows=[];density_sha=sha(source/'density.npy')
  for s,proc in workers:
   out=Path(s['output']);meta=read(out/'metadata.json');assert meta['indices']==s['indices'] and meta['density_sha256']==density_sha
   assert meta['solver']=='block' and meta['lightweight_reconstruction'] and len(meta['mesh'])==mesh**3
   view=Path(s['view']);assert sha(view/'metadata.json')==meta['source_metadata_sha256']
   bindings=read(view/'metadata.json')['generic_source_binding']
   for name in ['point.json','metadata.json','density.npy','mask.npy','input.xyz']:assert bindings[str(source/name)]==sha(source/name)
   for i in s['indices']:
    row=read(out/f'k_{i:04d}.json');assert row['index']==i and max(row['residuals_Ha'])<=1e-7
    assert np.isfinite(row['eigenvalues_Ha']).all() and np.isfinite(row['residuals_Ha']).all()
    assert np.allclose(row['k_frac'],point['k_fractional'][i],rtol=0,atol=1e-10);rows.append(row)
  assert sorted(x['index'] for x in rows)==list(range(mesh**3))
  v=max(rows,key=lambda x:x['vbm_Ha']);c=min(rows,key=lambda x:x['cbm_Ha']);gap=(c['cbm_Ha']-v['vbm_Ha'])*27.211386245988
  assert gap>2e-7*27.211386245988
  gamma=next(x for x in rows if max(abs(y) for y in x['k_frac'])<1e-10);gg=(gamma['cbm_Ha']-gamma['vbm_Ha'])*27.211386245988
  old=read(Path(cfg['start_spectrum']));target=read(D/'protocol.json')['target_gamma_eV']
  report=dict(time=datetime.datetime.now().isoformat(),mesh=[mesh]*3,theta=theta,source=str(source),
   density_sha256=density_sha,mask_sha256=sha(source/'mask.npy'),points=sorted(rows,key=lambda x:x['index']),
   full_mesh=True,all_registered_gates_passed=True,gap_eV=gap,gamma_gap_eV=gg,
   start_mesh_gap_eV=old['gap_eV'],mesh_gap_gain_eV=gap-old['gap_eV'],Gamma_gain_eV=gg-old['gamma_gap_eV'],
   fixed_original_target_eV=target,target_transfer_error_eV=gg-target,target_transfer_passed=abs(gg-target)<=.001,
   reverse_trajectory_Gamma_converged=True,max_eigen_residual_Ha=max(max(x['residuals_Ha']) for x in rows),
   scope='Same reverse-designed geometry, own k-mesh stationary density. Target not changed on transfer; no continuous-BZ or all-property certificate')
  dump(root/'result.json',report);stage('completed',mesh_gap_eV=gap,target_transfer_passed=report['target_transfer_passed'])
 except Exception as exc:stage('failed',error=str(exc));raise
def prepare(mesh):
 root=P/f'validation/results/response_design_20261004_reverse_endpoint_k{mesh}';root.mkdir(exist_ok=False)
 for folder in ['runs','spectra','views']:(root/folder).mkdir()
 shutil.copy2(Path(__file__),root/'controller.py')
 if mesh==4:
  family=P/'validation/results/response_design_20261004_corrected_k4_check';point=read(family/'plans/sequence.json')['experiments'][0]['command']
  spectra=read(family/'plans/corrected_endpoint_k4_spectra.json');parent=P/'validation/results/response_design_20261003_design_k4_check/runs/design_start_k4'
  start=P/'validation/results/response_design_20261003_design_k4_check/spectra/design_start_k4_complete.json';job='250285';count=8;gpus=list(range(8))
 else:
  family=P/'validation/results/response_design_20261003_design_k2_check';point=read(family/'plans/design_start_k2_sequence.json')['experiments'][0]['command']
  spectra=read(family/'plans/design_start_k2_spectra.json');parent=family/'runs/design_start_k2';start=family/'spectra/design_start_k2_complete.json';job='247061';count=4;gpus=[2,3]
 query=['srun','--jobid='+job,'--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32']
 allowed=json.loads(subprocess.check_output(query+['python3','-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True));physical=[x for x in allowed if x<64]
 wrapper=root/'private_tmp.sh';wrapper.write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261001_reverse_endpoint_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
mkdir -m 700 -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
 def configure(old,devices,tag,cores):
  b=old.index('env')+1;e=old.index('/bin/bash');settings={x.split('=',1)[0]:x.split('=',1)[1] for x in old[b:e]}
  tmp=f'/dev/shm/d3ft_response_design_20261001_reverse_endpoint_k{mesh}_{tag}'
  settings.update(CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/python_cache',
    JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',JAX_ENABLE_COMPILATION_CACHE='false',JAX_COMPILATION_CACHE_MAX_SIZE='0')
  mask=','.join(str(x) for c in cores for x in [c,c+64] if x in allowed)
  return query+[f'--gres=gpu:{count}','--gpu-bind=none','taskset','-c',mask,'timeout','--signal=TERM','--kill-after=30s','28800s','env']+[f'{k}={v}' for k,v in settings.items()]+['/bin/bash',str(wrapper)]+old[e+2:]
 cores=physical if mesh==4 else physical[4:8]
 cmd=configure(point,gpus,'ground',cores)
 for flag,value in [('--output',str(root/'runs/endpoint')),('--resume-from',str(parent))]:cmd[cmd.index(flag)+1]=value
 shards=[]
 for slot,gpu in enumerate(gpus):
  indices=list(range(slot,mesh**3,len(gpus)));s=spectra['shards'][0];view=root/'views'/f'gpu{gpu}';out=root/'spectra'/f'gpu{gpu}'
  cores=physical[slot::len(gpus)] if mesh==4 else physical[4+2*slot:6+2*slot]
  c=configure(s['command'],[gpu],f'spectrum_gpu{gpu}',cores)
  i=next(i for i,x in enumerate(c) if x.endswith('run_k4_spectrum_shard.py'))
  c=c[:i+1]+[str(root/'runs/endpoint'),str(view),str(out),json.dumps(indices)]
  shards.append(dict(name=f'spectrum_gpu{gpu}',command=c,indices=indices,view=str(view),output=str(out)))
 cfg=dict(mesh=mesh,jobid=job,gpus=gpus,parent=str(parent),start_spectrum=str(start),query=query,
  point_command=cmd,shards=shards,frozen_at=datetime.datetime.now().isoformat(),scope='Automatic verification of actual converged reverse endpoint; no old FD endpoint substituted')
 dump(root/'protocol.json',cfg)
 with (root/'controller.log').open('xb') as log:proc=subprocess.Popen([sys.executable,'-u',str(root/'controller.py'),'--execute',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 dump(root/'launch.json',dict(time=datetime.datetime.now().isoformat(),pid=proc.pid));print(str(root),proc.pid)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=int,choices=[2,4]);ap.add_argument('--execute',type=Path);a=ap.parse_args()
 run_mesh(a.execute) if a.execute else prepare(a.mesh)
