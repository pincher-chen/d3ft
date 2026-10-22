from pathlib import Path
import json,shutil,hashlib,subprocess,datetime
P=Path('${D3FT_ROOT}');R=P/'validation/results/response_design_20261004_reverse_response_v5';R.mkdir(exist_ok=False)
source=P/'validation/workflows/response_design/reverse_response'
for f in source.glob('*.py'):shutil.copy2(f,R/f.name)
runtime=P/'validation/results/response_design_20261002_resource_fill/comparison_runtime'
template=json.loads((P/'validation/results/response_design_20261002_next_phase/gamma_orbitals_strain_ref_plan.json').read_text())['experiments'][0]['command']
base=template[:template.index('/bin/bash')+3]
envstart=base.index('env')+1;envend=base.index('/bin/bash');settings={x.split('=',1)[0]:x.split('=',1)[1] for x in base[envstart:envend]}
settings.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
# The numerical parent is the original frozen comparison runtime, not current editable core code.
settings['PYTHONPATH']=f'{runtime}:{settings["D3FT_REFERENCE_ROOT"]}/software/jrystal'
cfg=dict(source=str(P/'validation/results/response_design_20261002_resource_fill/runs/mgsin2_gamma_strain_ref_fs_r3'),
 orbitals=str(P/'validation/results/response_design_20261002_next_phase/orbitals/gamma_orbitals_strain_ref'),runtime=str(runtime),
 native_gradient=str(P/'validation/results/response_design_20261003_dftk_export_repair/gradient_assembled.json'),
 theta0=[.003,-.002,.001,.002,-.001,.0015],direction=[.38263936589669045,.7652787317933809,-.38263936589669045,.19131968294834523,-.11479180976900713,.2678475561276833],
 component_relative_tolerance=.01,component_absolute_floor_eV=1e-4,
 residual_relative_tolerance=1e-8,residual_absolute_tolerance=1e-11,max_iterations=200,
 jobid=247061,source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in R.glob('*.py')},
 electronic_operator='Projected Omega+K on Q†dQ=0, real inner product, exact discrete Hartree/XC second derivative',
 scope='New real response implementation; finite nonlinear model checked before launch; all material gates pending until actual run and independent merge')
(R/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
(R/'algorithm_checks.json').write_text(json.dumps(dict(complex_real_linear_PCG=True,reject_negative_curvature=True,
 nonlinear_selfconsistent_model=True,independent_reoptimized_AD_FD_error=1.2214951272682129e-11,
 tangent_adjoint_error=0.,scope='Finite-model tests; not a DFT result'),indent=2)+'\n')
allocated=json.loads(subprocess.check_output(['srun','--jobid=247061','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32','/usr/bin/python3','-c','import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True))
physical=[x for x in allocated if x<64]
if len(physical)<8:raise RuntimeError('Insufficient allocated CPU cores for four isolated workers')
groups=[physical[i::4] for i in range(4)]
cpumasks=[','.join(str(x) for c in group for x in [c,c+64] if x in allocated) for group in groups]
launches=[]
for gpu,(name,mode,axes) in enumerate([('reverse','reverse',[]),('forward01','forward',[0,1]),('forward23','forward',[2,3]),('forward45','forward',[4,5])]):
 env=dict(settings);env.update(CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=f'/dev/shm/d3ft_response_design_20261001_response_{name}',
 JAX_COMPILATION_CACHE_DIR=f'/dev/shm/d3ft_response_design_20261001_response_{name}/jax_cache',PYTHONPYCACHEPREFIX=f'/dev/shm/d3ft_response_design_20261001_response_{name}/python_cache')
 cmd=['srun','--jobid=247061','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32','--gres=gpu:4','--gpu-bind=none',
 '/usr/bin/taskset','-c',cpumasks[gpu],
 '/usr/bin/timeout','--signal=TERM','--kill-after=30s','43200s','env']+[f'{k}={v}' for k,v in env.items()]+base[envend:]
 cmd += [str(R/'gamma_response.py'),str(R/'protocol.json'),'--mode',mode,'--output',str(R/name)]
 if axes:cmd+=['--axes']+list(map(str,axes))
 (R/(name+'_command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (R/(name+'.log')).open('xb') as log:process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(runtime))
 launches.append(dict(worker=name,gpu=gpu,pid=process.pid,command=cmd))
watch=['srun','--jobid=247061','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32','/usr/bin/taskset','-c',str(physical[-1]),'/usr/bin/timeout','43200s',base[-1],'-u',str(R/'merge_response.py'),str(R)]
with (R/'merge.log').open('xb') as log:process=subprocess.Popen(watch,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(R/'launch.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),workers=launches,merge=dict(pid=process.pid,command=watch)),indent=2)+'\n')
print(json.dumps({'root':str(R),'jobid':247061,'worker_pids':[x['pid'] for x in launches],'merge_pid':process.pid},indent=2))

reverse_command=json.loads((R/'reverse_command.json').read_text())
position=reverse_command.index(str(R/'gamma_response.py'))
cost_command=reverse_command[:position]+[str(R/'warmed_comparison.py'),str(R/'protocol.json'),'--output',str(R/'warmed_cost')]
(R/'warmed_cost_command.json').write_text(json.dumps(cost_command,indent=2)+'\n')
queue=watch[:watch.index(str(R/'merge_response.py'))]+[str(R/'queue_warmed_comparison.py'),str(R)]
with (R/'warmed_cost_queue.log').open('xb') as log:
    process=subprocess.Popen(queue,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(R/'warmed_cost_queue_launch.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),pid=process.pid,command=queue,dependency='material accuracy PASS'),indent=2)+'\n')
