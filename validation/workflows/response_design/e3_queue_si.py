"""Durable E3 Si development queue; no hn48 timing and no cancellation."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import datetime,json,os,subprocess,time,hashlib,argparse
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[3]
E=P/'validation/results/response_design_20261015_e3_geometry'
protocol=json.loads((E/'development_protocol.json').read_text());R=Path(protocol['runtime'])
base=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())
env=base['environment'].copy();env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
env['D3FT_PHASE3_RESULTS_ROOT']=str(E/'unused_legacy_sink')
ap=argparse.ArgumentParser();ap.add_argument('--resume-response',action='store_true');ap.add_argument('--attempt',default='initial');args=ap.parse_args()
os.close(os.open(E/f'si_queue_{args.attempt}.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def dump(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def status(stage,**kw):
 dump(E/'si_queue_status.json',dict(time=datetime.datetime.now().isoformat(),stage=stage,jobid=250285,formal_timing=False,**kw))
 print('E3_SI_QUEUE',stage,kw,flush=True)
spent=0.;gpu=None
def run(tag,args,limit=2700):
 global spent
 if spent+limit/3600>20:raise RuntimeError('Si development hard budget exhausted; no next submission')
 temp='/dev/shm/response_design_e3_'+tag
 taskenv={**os.environ,**env,'TMPDIR':temp,'JAX_COMPILATION_CACHE_DIR':temp+'/jax_cache','PYTHONPYCACHEPREFIX':temp+'/pycache'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4',
  'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e3','env',f'CUDA_VISIBLE_DEVICES={gpu}',
  'timeout',str(limit),protocol['python'],'-u',*map(str,args)]
 rec=dict(jobid=250285,gpu=gpu,tag=tag,command=cmd,environment={k:taskenv[k] for k in env},
  start=datetime.datetime.now().isoformat(),status='submitted',log=str(E/'logs'/f'{tag}.log'))
 dump(E/'commands'/f'{tag}.json',rec);status('running',tag=tag,gpu=gpu,spent_GPU_hours=spent)
 start=time.monotonic()
 with (E/'logs'/f'{tag}.log').open('xb') as f:
  child=subprocess.Popen(cmd,env=taskenv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  rec['pid']=child.pid;dump(E/'commands'/f'{tag}.json',rec);code=child.wait()
 elapsed=time.monotonic()-start;spent+=elapsed/3600
 rec.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=elapsed/3600,total_GPU_hours=spent,
  status='completed' if code==0 else 'failed');dump(E/'commands'/f'{tag}.json',rec)
 if code:raise RuntimeError(f'{tag} exit {code}; records retained, downstream not launched')
def geom(theta):
 x,y,z,yz,xz,xy=theta[:6];strain=[[x,xy/2,xz/2],[xy/2,y,yz/2],[xz/2,yz/2,z]]
 return ['--strain',json.dumps(strain),'--displacements',json.dumps((np.asarray(theta[6:]).reshape(2,3)/1.8897259886).tolist())]
S=protocol['Si'];theta=np.asarray(S['theta0']);point=R/'validation/workflows/response_design/generic_point.py'
bands=R/'validation/workflows/response_design/reverse_response/e3_primal_spectrum.py'
response=R/'validation/workflows/response_design/reverse_response/gamma_response.py'
common=['--structure',E/'inputs/si_structure.json','--pp-dir',E/'inputs/si_pp',
 '--cutoff',str(S['cutoff_Ha']),'--grid',str(S['grid']),'--k','1','--epochs',str(S['epochs']),
 '--local-radial-quadrature','simpson','--require-stationary']
try:
 status('waiting_free_gpu')
 while gpu is None:
  query=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=1',
   'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
  result=subprocess.run(query,capture_output=True,text=True,timeout=45)
  if result.returncode:raise RuntimeError('Allocation/GPU query failed: '+result.stderr[-1000:])
  for line in result.stdout.splitlines():
   index,mem,util=map(int,line.split(','))
   if index in protocol['gpu_candidates'] and mem<=3 and util==0:gpu=index;break
  if gpu is None:time.sleep(20)
 status('free_gpu_selected',gpu=gpu)
 basepoint=E/'si_base';basebands=E/'si_base_bands'
 if not args.resume_response:
  run('si_base',[point,*common,'--output',basepoint,*geom(theta)])
  run('si_base_bands',[bands,'--source',basepoint,'--output',basebands])
 else:
  m=json.loads((basebands/'metadata.json').read_text())
  for name,h in m['source_hashes'].items():
   assert hashlib.sha256((basepoint/name).read_bytes()).hexdigest()==h
  previous=json.loads((E/'si_response_protocol.json').read_text())
  dump(E/f'si_response_protocol_before_{args.attempt}.json',previous)
 rp=dict(source=str(basepoint),orbitals=str(basebands),runtime=str(R),theta0=theta.tolist(),
  include_coordinates=True,displacement_unit='bohr',source_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()
   for p in (R/'validation/workflows/response_design/reverse_response').glob('*.py')})
 dump(E/'si_response_protocol.json',rp)
 outputs={mode:str(E/f'si_{mode}_{args.attempt}') for mode in ['reverse','forward']}
 dump(E/'si_response_outputs.json',outputs)
 for mode in ['reverse','forward']:
  tag=f'si_{mode}_{args.attempt}'
  run(tag,[response,E/'si_response_protocol.json','--mode',mode,'--output',outputs[mode]])
 gate=R/'validation/workflows/response_design/reverse_response/e3_si_acceptance.py'
 run('si_mode_check_'+args.attempt,[gate,'--root',E,'--stage','mode'])
 d1=np.zeros(12);d1[6]=-1/np.sqrt(2);d1[9]=1/np.sqrt(2)
 d2=np.zeros(12);v=np.array([.3,.5,.7]);v/=np.linalg.norm(v)*np.sqrt(2);d2[6:9]=-v;d2[9:12]=v
 fd=[]
 dump(E/'si_fd_directions.json',dict(frozen_before_FD=True,directions=[d1.tolist(),d2.tolist()],steps_Bohr=S['FD_steps_Bohr']))
 for index,d in enumerate([d1,d2]):
  for h in S['FD_steps_Bohr']:
   for sign in [-1,1]:
    tag=f'si_fd_d{index}_h{h:g}_s{sign:+d}';dest=E/tag
    run(tag,[point,*common,'--output',dest,*geom(theta+sign*h*d),'--resume-from',basepoint,'--resume-geometry-continuation'])
    run(tag+'_bands',[bands,'--source',dest,'--output',E/(tag+'_bands')])
    fd.append(dict(direction=index,h=h,sign=sign,point=str(dest),bands=str(E/(tag+'_bands'))))
    dump(E/'si_fd_index.json',fd)
 run('si_full_acceptance_'+args.attempt,[gate,'--root',E,'--stage','full'])
 status('Si_derivative_substep_PASS',gpu=gpu,spent_GPU_hours=spent)
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3子步骤Si坐标导数：12分量forward/reverse、h/h2重优化FD和整体平移门全部通过；证据 validation/results/response_design_20261015_e3_geometry/si_acceptance.json。MgSiN2全54验收及hn48正式计时仍待做。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E3 Si坐标响应子步骤PASS；MgSiN2及正式计时未完成。\n')
 print('E3 子步骤 Si：全部导数验收 PASS。\n证据：response_design_20261015_e3_geometry/si_acceptance.json。\n下一步：MgSiN2 54 分量；正式计时尚未提交。',flush=True)
except Exception as exc:
 status('failed',error=str(exc),gpu=gpu,spent_GPU_hours=spent)
 raise
