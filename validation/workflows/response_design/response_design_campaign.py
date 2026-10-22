"""Matched fixed-path reverse/tangent trajectories and one Gamma elastic check."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261004_response_design'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
 f=Path(f);q=f.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(f)
def matrix(v):return [[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]]
def run(command,tag):
 dump(R/'execution'/(tag+'_command.json'),command)
 with (R/'execution'/(tag+'.log')).open('xb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{tag} RC {result.returncode}; original log retained')
def source_gate(cfg,source):
 pt=read(source/'point.json');assert pt['stationarity_pass'] and pt['residual']<=1e-6
 assert pt['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(pt['integrated_valence_density']-96)<=.00096
 assert sha(source/'mask.npy')==cfg['parent_mask_sha256'] and len(pt['k_fractional'])==1
 parent=read(Path(cfg['base_source'])/'metadata.json');meta=read(source/'metadata.json')
 assert meta['code_sha256']==parent['code_sha256'] and meta['physical']['pp']==parent['physical']['pp']
 return pt
def prefix(cfg,gpu,tag):
 base=list(cfg['prefixes'][str(gpu)]);tmp=f'/dev/shm/d3ft_response_design_20261001_response_design_{tag}'
 for i,x in enumerate(base):
  if x.startswith('TMPDIR='):base[i]='TMPDIR='+tmp
  elif x.startswith('JAX_COMPILATION_CACHE_DIR='):base[i]='JAX_COMPILATION_CACHE_DIR='+tmp+'/jax_cache'
  elif x.startswith('PYTHONPYCACHEPREFIX='):base[i]='PYTHONPYCACHEPREFIX='+tmp+'/python_cache'
 return base
def point(cfg,t,tag,gpu,parent):
 source_gate(cfg,Path(parent));out=R/'runs'/tag;theta=np.asarray(cfg['theta0'])+t*np.asarray(cfg['direction'])
 command=prefix(cfg,gpu,tag+'_ground')+list(cfg['point_tail'])
 for flag,value in [('--output',str(out)),('--resume-from',str(parent)),('--strain',json.dumps(matrix(theta))),('--epochs',str(cfg['design_epochs']))]:command[command.index(flag)+1]=value
 run(command,tag+'_ground');source_gate(cfg,out);return out,theta
def orbitals(cfg,source,tag,gpu):
 output=R/'orbitals'/tag
 run(prefix(cfg,gpu,tag+'_orbitals')+[str(R/'save_response_orbitals.py'),str(source),str(output)],tag+'_orbitals')
 bands=read(output/'summary.json');assert max(bands['residuals_Ha'])<=1e-7 and bands['orthogonality_error']<=1e-8
 assert min(bands['vbm_adjacent_separation_Ha'],bands['cbm_adjacent_separation_Ha'])>1e-5
 return output,bands
def response(cfg,source,orb,theta,tag,gpu,mode):
 protocol=dict(read(R/'base_response_protocol.json'),source=str(source),orbitals=str(orb),theta0=list(theta))
 f=R/'execution'/(tag+'_response_protocol.json');dump(f,protocol);out=R/'responses'/tag
 if mode=='reverse':cmd=prefix(cfg,gpu,tag+'_response')+[str(R/'gamma_response.py'),str(f),'--mode','reverse','--output',str(out)]
 else:cmd=prefix(cfg,gpu,tag+'_response')+[str(R/'directional_response.py'),str(f),'--mode','forward','--output',str(out)]
 run(cmd,tag+'_response');result=read(out/'result.json')
 if mode=='reverse':
  assert result['results']['response_record']['converged'];derivative=float(np.dot(result['results']['gap_gradient_eV'],cfg['direction']))
 else:assert result['response_record']['converged'];derivative=result['directional_gap_derivative_eV']
 assert np.isfinite(derivative) and abs(derivative)>1e-6
 return derivative,result
def trajectory(mode):
 cfg=read(R/'protocol.json');gpu=0 if mode=='reverse' else 1;target=cfg['target_gamma_eV'];base=Path(cfg['base_source'])
 accepted=read(P/'validation/results/response_design_20261004_reverse_response_v5/acceptance.json');assert accepted['passed']
 gradient=accepted['reverse_gradient_eV'] if mode=='reverse' else accepted['same_program_forward_gradient_eV']
 derivative=float(np.dot(gradient,cfg['direction']));t=0.;gap=cfg['reference_gamma_eV'];parent=base
 history=[dict(iteration=0,t=0.,gap_eV=gap,error_eV=gap-target,directional_derivative_eV=derivative,source=str(parent),reused_accepted_response=True)]
 def state(stage,**kw):dump(R/(mode+'_status.json'),dict(stage=stage,time=datetime.datetime.now().isoformat(),**kw))
 try:
  for i in range(1,cfg['design_max_updates']+1):
   step=np.clip((target-gap)/derivative,-cfg['design_max_step'],cfg['design_max_step']);trial=float(np.clip(t+step,-cfg['design_bound'],cfg['design_bound']))
   if abs(trial-t)<1e-8:raise RuntimeError('Bounded path stalled')
   tag=f'{mode}_iter{i}';state('ground_state',iteration=i,t=trial)
   source,theta=point(cfg,trial,tag,gpu,parent);state('readout',iteration=i,t=trial)
   orb,bands=orbitals(cfg,source,tag,gpu);new_gap=bands['gap_eV'];better=abs(new_gap-target)<abs(gap-target)
   history.append(dict(iteration=i,t=trial,gap_eV=new_gap,error_eV=new_gap-target,actual_loss_accepted=better,source=str(source),orbitals=str(orb),theta=theta.tolist()))
   dump(R/(mode+'_history.json'),history)
   if not better:state('completed',converged=False,reason='actual objective did not improve');return
   t,gap,parent=trial,new_gap,source
   if abs(gap-target)<=cfg['design_tolerance_eV']:state('completed',converged=True,updates=i,t=t,source=str(source),theta=theta.tolist());return
   if i==cfg['design_max_updates']:state('completed',converged=False,reason='registered update limit');return
   state('electronic_response',iteration=i,t=t);derivative,record=response(cfg,source,orb,theta,tag,gpu,mode)
   history[-1]['directional_derivative_eV']=derivative;dump(R/(mode+'_history.json'),history)
 except Exception as exc:state('failed',error=str(exc));raise
def elastic():
 cfg=read(R/'protocol.json');protocol=read(R/'base_response_protocol.json');protocol.update(fd_h=.001,
  fd_plus=str(P/'validation/results/response_design_20261002_resource_fill/runs/mgsin2_gamma_direction_plus_fs_r3'),
  fd_minus=str(P/'validation/results/response_design_20261002_resource_fill/runs/mgsin2_gamma_direction_minus_fs_r3'),
  elastic_absolute_tolerance_GPa=.2,elastic_relative_tolerance=.02)
 f=R/'execution/elastic_response_protocol.json';dump(f,protocol)
 try:
  run(prefix(cfg,2,'elastic_analytic')+[str(R/'directional_response.py'),str(f),'--mode','elastic','--output',str(R/'elastic_response')],'elastic_analytic')
  # A second difference is a new observable: one smaller step checks its truncation.
  point(cfg,-.0005,'elastic_minus_half',2,cfg['base_source'])
 except Exception as exc:dump(R/'elastic_task_failure.json',dict(error=str(exc)));raise
def half_plus():
 cfg=read(R/'protocol.json');point(cfg,.0005,'elastic_plus_half',3,cfg['base_source'])
def prepare():
 R.mkdir(exist_ok=False)
 for name in ['execution','runs','orbitals','responses']:(R/name).mkdir()
 shutil.copy2(Path(__file__),R/'controller.py')
 for f in (P/'validation/workflows/response_design/reverse_response').glob('*.py'):shutil.copy2(f,R/f.name)
 old=P/'validation/results/response_design_20261002_next_phase';cfg=read(old/'protocol.json');cfg['design_epochs']=3000;cfg['jobid']=247061
 c=read(old/'gpu_template.json')['experiments'][0]['command'];e=c.index('/bin/bash');template=c[:e+3]
 query=['srun','--jobid=247061','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32']
 inv=subprocess.check_output(query+['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 assert all(int(x.split(',')[1])<=100 for x in inv.strip().splitlines()),inv
 allowed=json.loads(subprocess.check_output(query+['python3','-c','import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True));physical=[x for x in allowed if x<64]
 wrapper=R/'private_tmp.sh';wrapper.write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261001_response_design_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
mkdir -m 700 -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
 begin=c.index('env')+1;settings={x.split('=',1)[0]:x.split('=',1)[1] for x in c[begin:e]}
 settings.update(JAX_ENABLE_COMPILATION_CACHE='false',JAX_COMPILATION_CACHE_MAX_SIZE='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 settings['PYTHONPATH']=str(Path(cfg['runtime']))+':'+str(Path(settings['D3FT_REFERENCE_ROOT'])/'software/jrystal')
 prefixes={}
 for gpu in range(4):
  group=physical[2*gpu:2*gpu+2];mask=','.join(str(x) for core in group for x in [core,core+64] if x in allowed)
  env=dict(settings,CUDA_VISIBLE_DEVICES=str(gpu))
  prefixes[str(gpu)]=query+['--gres=gpu:4','--gpu-bind=none','taskset','-c',mask,'timeout','--signal=TERM','--kill-after=30s','21600s','env']+[f'{k}={v}' for k,v in env.items()]+['/bin/bash',str(wrapper),c[e+2]]
 cfg.update(prefixes=prefixes,point_tail=c[e+3:],frozen_at=datetime.datetime.now().isoformat(),
  source_hashes={f.name:sha(f) for f in R.glob('*.py')},native_CPU_mask=','.join(str(x) for core in physical[8:] for x in [core,core+64] if x in allowed),
  scope='Two independent matched 1D Gamma trajectories: actual electronic adjoint versus one path tangent; native AD-DFPT run separately. Same target, bounds, step and source, no per-trajectory speed claim from R1 six-input cost.')
 dump(R/'protocol.json',cfg);shutil.copy2(P/'validation/results/response_design_20261004_reverse_response_v5/protocol.json',R/'base_response_protocol.json')
 launches=[]
 for task in ['reverse','forward','elastic','half_plus']:
  with (R/(task+'_controller.log')).open('xb') as log:proc=subprocess.Popen([sys.executable,'-u',str(R/'controller.py'),'--task',task],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launches.append(dict(task=task,pid=proc.pid))
 dump(R/'launch.json',dict(time=datetime.datetime.now().isoformat(),jobid=247061,tasks=launches));print(str(R),launches)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['reverse','forward','elastic','half_plus']);a=ap.parse_args()
 if a.task in ['reverse','forward']:trajectory(a.task)
 elif a.task=='elastic':elastic()
 elif a.task=='half_plus':half_plus()
 else:prepare()
