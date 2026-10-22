"""Preserve coarse FD, preregister a smaller h/h2 pair at unchanged tolerance."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import json,datetime,os,subprocess,time
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry'
cfg=json.loads((E/'development_protocol.json').read_text());R=Path(cfg['runtime']);S=cfg['Si']
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())
env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
os.close(os.open(E/'si_refinement.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def dump(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):
 dump(E/'si_refinement_status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),jobid=250285,**kw));print('SI_FD_REFINEMENT',stage,kw,flush=True)
def launch(gpu,tag,args):
 tmp='/dev/shm/response_design_e3_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','1800',cfg['python'],'-u',*map(str,args)]
 rec=dict(tag=tag,jobid=250285,gpu=gpu,command=cmd,environment={k:te[k] for k in env},start=datetime.datetime.now().isoformat(),status='running')
 dump(E/'commands'/f'{tag}.json',rec);f=(E/'logs'/f'{tag}.log').open('xb');proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec['pid']=proc.pid;dump(E/'commands'/f'{tag}.json',rec);return proc,f,rec
def join(jobs):
 while any(p.poll() is None for p,*_ in jobs):time.sleep(5)
 failures=[]
 for proc,f,rec in jobs:
  f.close();rec.update(end=datetime.datetime.now().isoformat(),returncode=proc.returncode,status='completed' if proc.returncode==0 else 'failed')
  dump(E/'commands'/(rec['tag']+'.json'),rec)
  if proc.returncode:failures.append(rec['tag'])
 if failures:raise RuntimeError('Refinement failed '+str(failures))
try:
 directions=json.loads((E/'si_fd_directions.json').read_text())['directions'];steps=[.0005,.00025]
 dump(E/'si_refined_fd_directions.json',dict(time=datetime.datetime.now().isoformat(),directions=directions,
  steps_Bohr=steps,frozen_before_refined_calculations=True,thresholds=S['thresholds'],
  reason='Original h=.002 pair has FD truncation error .06349; do not change derivative/mode tolerance. Test h/4 and h/8 and preserve coarse h/h2.'))
 status('waiting_coarse_direction0_h_half')
 while not (E/'si_fd_d0_h0.001_s+1_bands/summary.json').exists():time.sleep(10)
 g=np.asarray(json.loads((E/'si_mode_acceptance.json').read_text())['reverse_gradient']);pred=float(g@np.asarray(directions[0]));errors=[]
 for h in [.002,.001]:
  vals=[json.loads((E/f'si_fd_d0_h{h:g}_s{sign:+d}_bands/summary.json').read_text())['gap_eV'] for sign in [-1,1]]
  fd=(vals[1]-vals[0])/(2*h);errors.append(abs(fd-pred))
 dump(E/'si_coarse_fd_truncation.json',dict(prediction=pred,steps=[.002,.001],errors=errors,error_ratio=errors[0]/errors[1],threshold_unchanged=True))
 if not errors[1]<.4*errors[0]:raise RuntimeError('Coarse FD not approximately second-order; cannot assume smaller h resolves it')
 tasks=[(i,h,sign) for i in range(2) for h in steps for sign in [-1,1]];done=[]
 for start in range(0,8,3):
  batch=tasks[start:start+3];status('waiting_free_refinement_gpus',batch=start//3)
  while True:
   q=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
   result=subprocess.run(q,capture_output=True,text=True,timeout=45)
   if result.returncode:raise RuntimeError('GPU query failed')
   free=[]
   for row in result.stdout.splitlines():
    i,m,u=map(int,row.split(','))
    if i in [1,5,7] and m<=3 and u==0:free.append(i)
   if len(free)>=len(batch):break
   time.sleep(10)
  jobs=[];records=[]
  for gpu,(i,h,sign) in zip(free,batch):
   d=np.asarray(directions[i]);theta=np.asarray(S['theta0'])+sign*h*d
   x,y,z,yz,xz,xy=theta[:6];strain=[[x,xy/2,xz/2],[xy/2,y,yz/2],[xz/2,yz/2,z]]
   tag=f'si_refined_fd_d{i}_h{h:g}_s{sign:+d}';out=E/tag
   cli=[R/'validation/workflows/response_design/generic_point.py','--structure',E/'inputs/si_structure.json','--pp-dir',E/'inputs/si_pp','--cutoff','20','--grid','40','--k','1','--epochs','4000','--local-radial-quadrature','simpson','--require-stationary','--output',out,'--strain',json.dumps(strain),'--displacements',json.dumps((theta[6:].reshape(2,3)/1.8897259886).tolist()),'--resume-from',E/'si_base','--resume-geometry-continuation']
   jobs.append(launch(gpu,tag,cli));records.append(dict(direction=i,h=h,sign=sign,point=str(out),bands=str(E/(tag+'_bands')),gpu=gpu,tag=tag))
  status('refined_electronic_points_running',tags=[r['tag'] for r in records]);join(jobs)
  jobs=[launch(r['gpu'],r['tag']+'_bands',[R/'validation/workflows/response_design/reverse_response/e3_primal_spectrum.py','--source',r['point'],'--output',r['bands']]) for r in records]
  join(jobs);done+=records;dump(E/'si_refined_fd_index.json',done)
 # Let original coarse tests conclude and retain their failed gate before release.
 status('waiting_original_coarse_chain_to_finish')
 while not (E/'si_acceptance.json').exists():time.sleep(10)
 # Earlier gate controllers poll every 20 seconds. Let them exit naturally
 # before launching the refined attempt, so they cannot overwrite new status.
 for launchfile in ['mg_controller_retry1_launch.json','mg_fd_controller_launch.json']:
  launchrecord=json.loads((E/launchfile).read_text());oldpid=launchrecord['pid']
  while True:
   try:
    process_state=Path(f'/proc/{oldpid}/stat').read_text().split(') ',1)[1].split()[0]
   except FileNotFoundError:break
   if process_state=='Z':break
   time.sleep(5)
 with (E/'logs/si_refined_acceptance.log').open('xb') as f:
  result=subprocess.run(['python','-u',str(R/'validation/workflows/response_design/reverse_response/e3_si_acceptance.py'),'--root',str(E),'--stage','full','--refined'],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Smaller-step FD still failed; no Mg submission')
 status('Si_derivative_substep_PASS_with_FD_refinement')
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3 Si坐标导数子步骤完成：12维mode/平移及两方向h=.0005/.00025 Bohr重优化FD通过，阈值不变；原h=.002/.001截断误差不合格保留于si_acceptance.json及si_coarse_fd_truncation.json。Mg及正式计时仍未验收。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3 Si坐标导数PASS，较粗FD失败保留，细步长门不变；Mg自动接续。\n')
 print('E3 Si：mode、平移与细步长 h/h2 重优化 FD PASS。\n粗步长失败完整保留，验收阈值未放宽。\n自动接续 MgSiN2，正式计时不提交。',flush=True)
 # Earlier waiting controllers exited on coarse FD failure; new attempt uses
 # explicitly refined gate, without rewriting a failed report into a PASS.
 for label,script,extra in [('mg','e3_queue_mg.py',['--attempt','refined','--si-gate','si_acceptance_refined.json']),('mg_fd','e3_queue_mg_fd.py',['--attempt','refined'])]:
  if label=='mg_fd':time.sleep(3)
  cmd=['python','-u',str(P/'validation/workflows/response_design'/script),*extra]
  with (E/f'{label}_controller_refined.log').open('xb') as f:c=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(E/f'{label}_controller_refined_launch.json',dict(pid=c.pid,jobid=250285,command=cmd,time=datetime.datetime.now().isoformat(),formal_timing=False))
except Exception as exc:status('failed',error=str(exc));raise
