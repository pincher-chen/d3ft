"""One registered clamped-ion affine energy-curvature mode on the k4 mesh."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261004_elastic_mode_k4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def strain(v):return np.array([[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]])
def execute():
 cfg=read(R/'protocol.json');base=Path(cfg['base']);parent=read(base/'point.json');mask=sha(base/'mask.npy')
 def gate(folder):
  pt=read(folder/'point.json');assert pt['stationarity_pass'] and pt['residual']<=1e-6
  assert pt['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(pt['integrated_valence_density']-96)<=.00096
  assert len(pt['k_fractional'])==64 and sha(folder/'mask.npy')==mask
  meta=read(folder/'metadata.json');old=read(base/'metadata.json')
  assert meta['code_sha256']==old['code_sha256'] if 'code_sha256' in meta else meta['code']==old['code']
  return pt
 try:
  states={}
  for e in cfg['experiments']:
   dump(R/'status.json',dict(stage='running',task=e['name'],time=datetime.datetime.now().isoformat()))
   with (R/(e['name']+'.log')).open('xb') as log:result=subprocess.run(e['command'],stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError(f"{e['name']} RC {result.returncode}; original log preserved")
   states[e['name']]=gate(Path(e['output']))
  h=cfg['h'];direction=strain(np.asarray(cfg['direction']));theta=strain(np.asarray(cfg['theta0']));volume=parent['volume_bohr3']
  def slope(pt,t):
   pull=direction@np.linalg.inv(np.eye(3)+theta+t*direction)
   return pt['volume_bohr3']*float(np.sum(np.asarray(pt['stress_Ha_Bohr3'])*pull))
  ep,em=states['plus'],states['minus'];energy_curvature=(ep['energy_Ha']-2*parent['energy_Ha']+em['energy_Ha'])/h**2
  stress_curvature=(slope(ep,h)-slope(em,-h))/(2*h)
  ce=energy_curvature/volume*29421.015697;cs=stress_curvature/volume*29421.015697
  passed=bool(np.isfinite(ce) and np.isfinite(cs) and abs(ce-cs)<=cfg['energy_stress_tolerance_GPa'])
  dump(R/'result.json',dict(passed=passed,energy_curvature_modulus_GPa=ce,stress_slope_modulus_GPa=cs,
    difference_GPa=abs(ce-cs),h=h,direction=cfg['direction'],base=str(base),
    states={n:{k:s[k] for k in ['energy_Ha','residual','acoustic_sum_rule_Ha_Bohr','volume_bohr3']} for n,s in states.items()},
    scope='One clamped-ion affine directional E second derivative divided by reference volume; pre-stress included, not a full elastic tensor or analytic k4 DFPT'))
  dump(R/'status.json',dict(stage='completed',passed=passed,time=datetime.datetime.now().isoformat()))
 except Exception as exc:dump(R/'status.json',dict(stage='failed',error=str(exc),time=datetime.datetime.now().isoformat()));raise
def prepare():
 R.mkdir(exist_ok=False);shutil.copy2(Path(__file__),R/'controller.py')
 old=read(P/'validation/results/response_design_20261004_corrected_k4_check/plans/sequence.json')['experiments'][0]['command']
 cfg0=read(P/'validation/results/response_design_20261002_next_phase/protocol.json');base=P/'validation/results/response_design_20261003_design_k4_check/runs/design_start_k4'
 query=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32']
 inventory=subprocess.check_output(query+['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 assert all(int(x.split(',')[1])<=100 for x in inventory.strip().splitlines()),inventory
 allowed=json.loads(subprocess.check_output(query+['python3','-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True))
 wrapper=R/'private_tmp.sh';wrapper.write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261001_elastic_k4_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
mkdir -m 700 -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
 experiments=[];h=.001
 for name,sign in [('plus',1),('minus',-1)]:
  cmd=list(old);tmp=f'/dev/shm/d3ft_response_design_20261001_elastic_k4_{name}'
  cmd=[f'TMPDIR={tmp}' if x.startswith('TMPDIR=') else f'PYTHONPYCACHEPREFIX={tmp}/python_cache' if x.startswith('PYTHONPYCACHEPREFIX=') else f'JAX_COMPILATION_CACHE_DIR={tmp}/jax_cache' if x.startswith('JAX_COMPILATION_CACHE_DIR=') else x for x in cmd]
  cmd[cmd.index('/bin/bash')+1]=str(wrapper);cmd.insert(cmd.index('/bin/bash'),'JAX_COMPILATION_CACHE_MAX_SIZE=0')
  cmd[cmd.index('-c')+1]=','.join(map(str,allowed))
  out=R/'runs'/name;theta=np.asarray(cfg0['theta0'])+sign*h*np.asarray(cfg0['direction'])
  for flag,value in [('--output',str(out)),('--resume-from',str(base)),('--strain',json.dumps(strain(theta).tolist()))]:cmd[cmd.index(flag)+1]=value
  experiments.append(dict(name=name,output=str(out),command=cmd))
 cfg=dict(base=str(base),h=h,theta0=cfg0['theta0'],direction=cfg0['direction'],jobid=250285,experiments=experiments,
  energy_stress_tolerance_GPa=.1,frozen_at=datetime.datetime.now().isoformat(),base_hashes={n:sha(base/n) for n in ['point.json','metadata.json','mask.npy']},
  scope='Original reference geometry, fixed fractional coordinates/Miller mask/PP/core; central h=0.001 energy and conjugate-stress curvature')
 dump(R/'protocol.json',cfg)
 with (R/'controller.log').open('xb') as log:proc=subprocess.Popen([sys.executable,'-u',str(R/'controller.py'),'--execute'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 dump(R/'launch.json',dict(time=datetime.datetime.now().isoformat(),pid=proc.pid,jobid=250285));print(str(R),proc.pid)
if __name__=='__main__':execute() if '--execute' in sys.argv else prepare()
