"""Free fixed-allocation GPUs: gated pilot, parallel paths, then publication builder."""
import concurrent.futures,datetime,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];R=P/'validation/results';O=R/'response_design_20261005_band_structure'
read=lambda p:json.loads(Path(p).read_text())
def dump(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
protocol=read(O/'path_protocol.json');old=read(R/'response_design_20261004_reverse_endpoint_k4/protocol.json');c=old['shards'][0]['command'];env={x.split('=',1)[0]:x.split('=',1)[1] for x in c[c.index('env')+1:c.index('/bin/bash')]}
python=c[c.index('/bin/bash')+2]
wrapper=O/'private_tmp.sh'
wrapper.write_text('#!/bin/bash\nset -eu\ncase "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261005_bands_*) ;; *) exit 90;; esac\n[ ! -e "$TMPDIR" ] || exit 91\nmkdir -m 700 "$TMPDIR"\ntrap \'rm -rf -- "$TMPDIR"\' EXIT\nulimit -c 0\n"$@"\n')
allocations=[('250285',8),('250212',4)]
slots=[]
for job,n in allocations:
 cmd=['srun',f'--jobid={job}','--overlap','--cpu-bind=none','-N1','-n1',f'--cpus-per-task={4*n}',f'--gres=gpu:{n}','--gpu-bind=none']
 result=subprocess.run(cmd+['python3','-c','import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))'],capture_output=True,text=True,timeout=30,check=True)
 cpus=json.loads(result.stdout.strip().splitlines()[-1]);
 for gpu in range(n):slots.append((job,n,gpu,cpus[gpu::n]))
def execute(state,indices,slot,phase):
 job,n,gpu,cpus=slot;name=f'{phase}_{state}_{job}_gpu{gpu}';out=O/'shards'/name;tmp=f'/dev/shm/d3ft_response_design_20261005_bands_{name}'
 e=env.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/python_cache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache')
 cmd=['srun',f'--jobid={job}','--overlap','--cpu-bind=none','-N1','-n1',f'--cpus-per-task={4*n}',f'--gres=gpu:{n}','--gpu-bind=none','taskset','-c',','.join(map(str,cpus)),'timeout','--signal=TERM','--kill-after=30s','43200s','env']+[f'{k}={v}' for k,v in e.items()]+['/bin/bash',str(wrapper),python,'-u',str(P/'validation/workflows/response_design/band_path_worker.py'),str(O/'path_protocol.json'),state,str(out),json.dumps(indices)]
 dump(O/(name+'_command.json'),cmd)
 with (O/(name+'.log')).open('x') as f:run=subprocess.run(cmd,cwd=str(R/'response_design_20261002_resource_fill'),stdout=f,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError(f'{name} failed RC{run.returncode}; inspect log')
 return dict(state=state,indices=indices,path=str(out))
def stage(stage,**kw):dump(O/'status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw));print(stage,kw,flush=True)
try:
 O.joinpath('shards').mkdir(exist_ok=True)
 stage('pilot_running',slots=slots)
 edge=list(protocol['edge_indices'].values())
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  pilot=list(pool.map(lambda args:execute(*args),[(s,[i],slots[j],'pilot') for j,(s,i) in enumerate((s,i) for s in protocol['states'] for i in edge)]))
 dump(O/'pilot_results.json',pilot)
 rows=[read(Path(r['path'])/f'k_{i:04d}.json') for r in pilot for i in r['indices']]
 if any(r['existing_spectrum_max_difference_eV'] is None for r in rows):raise ValueError('Missing old-spectrum pilot comparison')
 stage('path_running',pilot_passed=True,pilot_max_difference_eV=max(r['existing_spectrum_max_difference_eV'] for r in rows),pilot_seconds=[r['seconds'] for r in rows])
 tasks=[]
 for si,s in enumerate(protocol['states']):
  left=[i for i in range(len(protocol['k_fractional'])) if i not in edge]
  for g in range(4):tasks.append((s,left[g::4],slots[si*4+g],'path'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:full=list(pool.map(lambda args:execute(*args),tasks))
 results=pilot+full;dump(O/'all_shards.json',results)
 merged={}
 for state in protocol['states']:
  rr=[read(Path(s['path'])/f'k_{i:04d}.json') for s in results if s['state']==state for i in s['indices']];rr.sort(key=lambda r:r['index'])
  if [r['index'] for r in rr]!=list(range(len(protocol['k_fractional']))):raise ValueError('Missing or duplicate points')
  merged[state]=dict(source=protocol['states'][state],points=rr,full_path=True,occupied_bands=48,max_eigen_residual_Ha=max(max(r['residuals_Ha']) for r in rr),scope='Mapped parent path only; no full-BZ extremum certificate')
 dump(O/'bands_complete.json',merged);stage('spectra_complete',states=list(merged))
 builder=P/'paper_rewriting_output/final_paper/plot_electronic_postprocessing.py'
 base='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/bin/python'
 with (O/'publication.log').open('x') as f:subprocess.run([base,str(builder),'--complete'],stdout=f,stderr=subprocess.STDOUT,check=True)
 stage('completed',publication_builder=True)
except Exception as exc:stage('failed',error=str(exc));raise
