from pathlib import Path
import json,shutil,os,subprocess,datetime,hashlib,sys
P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261004_compact_training_v2';R.mkdir(exist_ok=False)
runtime=R/'runtime';runtime.mkdir()
shutil.copytree(P/'d3ft',runtime/'d3ft',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(P/'validation/config',runtime/'validation/config',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(P/'validation/workflows/phase3',runtime/'validation/workflows/phase3',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(P/'validation/workflows/response_design/compact_training.py',R/'compact_training.py')
base=json.loads((P/'validation/results/response_design_20261002_next_phase/gpu_template.json').read_text())['experiments'][0]['command']
begin=base.index('env')+1;end=base.index('/bin/bash')
env={x.split('=',1)[0]:x.split('=',1)[1] for x in base[begin:end]}
env.update(PYTHONPATH=f'{runtime}:{P}/validation/results/response_design_20261002_resource_fill:{env["D3FT_REFERENCE_ROOT"]}/software/jrystal',
           D3FT_PHASE3_RESULTS_ROOT=str(R/'training/runs'),D3FT_TRACE_STAGES='1',OMP_NUM_THREADS='4')
# New isolated CPU cores; no shared 16-thread masks.
worker=['srun','--jobid=250212','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=12','--gres=gpu:4','--gpu-bind=none','/usr/bin/timeout','--signal=TERM','--kill-after=30s','43200s','env']+[f'{k}={v}' for k,v in env.items()]+['/bin/bash',str(P/'validation/results/response_design_20261002_resource_fill/comparison_runtime/private_tmp.sh'),str(Path(env['D3FT_REFERENCE_ROOT']).parent.parent.parent.parent/'unused')]
# Use the exact interpreter already proven by the parent task.
worker[-1]=base[base.index('/bin/bash')+2]
manifest=json.loads((P/'validation/results/phase3_20260908/production_manifest_v3.json').read_text())
matsettings={'Si':(50,60,10,1,.005),'BN':(100,60,10,4,.0025),'SiC':(50,48,12,8,.0025)}
materials=json.loads((runtime/'validation/workflows/phase3/materials.json').read_text())
settings={}
for mat,(cut,grid,k,factor,step) in matsettings.items():
 old=json.loads((Path(manifest[mat])/'summary.json').read_text());m=materials[mat]
 settings[mat]=dict(cutoff=cut,grid=grid,k=k,xc_grid_factor=factor,strain_step=step,initial_a0=old['a0_A'],
     references={'a0_A':m['a_ref'],'B_GPa':(m['C11']+2*m['C12'])/3},relative_weight=m['sigma_rel'],
     reference_source='Existing materials.json and paper_rewriting_output/final_paper/response-baseline.csv; historical temperature/covariance qualifications remain',
     historical_production=manifest[mat])
cfg=dict(output=str(R/'training'),runtime=str(runtime),jobid=250212,train_materials=['Si','BN'],observables=['a0_A','B_GPa'],
 holdout_material='SiC',training_gpu_map={'Si':[0,1],'BN':[2,3]},holdout_gpus=[0,1,2,3],worker_command_prefix=worker,
 materials=settings,epochs=3000,point_timeout_seconds=45000,maximum_trials=3,minimum_parameter_step=1e-7,
 scales=[.1,.02],absolute_delta_bounds=[.0804,.02195149727645171],initial_scaled_radius=.2,maximum_scaled_radius=.4,
 initial_damping=1.,minimum_acceptance_ratio=.1,
 scope='Fixed small experimental-property calibration pilot; no old earlier development all-matrix precondition. Fresh baseline and candidates run same frozen source/settings; individual parameter derivative validation and final numerical-resolution claim remain separate.',
 frozen_at=datetime.datetime.now().isoformat())
(R/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
paths=[R/'compact_training.py',R/'protocol.json']+list((runtime/'d3ft').rglob('*.py'))+list((runtime/'validation/workflows/phase3').glob('*.py'))
(R/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
# Test proposal, rejection, bounding before consuming GPU resources.
sys.path.insert(0,str(R));from compact_training import proposal,decision
import numpy as np
r=np.array([1.,-2.,.3]);j=np.array([[2.,1.],[1.,-1.],[.2,.4]])
x,pred=proposal(r,j,np.zeros(2),[.1,.02],.2,1.,[.0804,.02195])
assert pred>0 and np.linalg.norm(x/np.array([.1,.02]))<=.2+1e-12
assert not decision(1.,1.1,.2)['accepted']
assert not decision(1.,.9,0.)['accepted']
assert decision(1.,.8,.25)['accepted']
assert np.all(abs(x)<=np.array([.0804,.02195]))
(R/'workflow_checks.json').write_text(json.dumps({'bounded_scaled_proposal':True,'reject_actual_increase':True,'reject_nonpositive_prediction':True,'accept_actual_decrease_with_ratio':True,'holdout_not_in_training':True},indent=2)+'\n')
controller=['srun','--jobid=250212','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32','/usr/bin/timeout','--signal=TERM','--kill-after=30s','172800s',base[base.index('/bin/bash')+2],'-u',str(R/'compact_training.py'),str(R/'protocol.json')]
with (R/'controller.log').open('xb') as logfile:
 process=subprocess.Popen(controller,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(runtime))
(R/'launch.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),pid=process.pid,command=controller,log=str(R/'controller.log')),indent=2)+'\n')
print(json.dumps({'directory':str(R),'pid':process.pid,'jobid':250212,'training':['Si','BN'],'holdout':'SiC','checks':'passed'},indent=2))
