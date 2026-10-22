"""Observe existing E5 steps, then response and FD; only idle 250285 GPUs."""
from pathlib import Path
import json,datetime,os,time,subprocess
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_kmesh';cfg=json.loads((E/'protocol.json').read_text());R=Path(cfg['runtime'])
base=json.loads((E/'commands/si_k2_base.json').read_text());env=base['environment'].copy()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):dump(E/'status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),jobid=250285,**kw));print('E5_QUEUE',stage,kw,flush=True)
spent=0.
def run(tag,gpu,cli):
 global spent
 if spent+1>12:raise RuntimeError('E5 pilot GPU subbudget exhausted, no next step')
 tmp='/dev/shm/response_design_e5_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'JAX_COMPILATION_CACHE_DIR':tmp+'/jax_cache','PYTHONPYCACHEPREFIX':tmp+'/pycache'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','3600',cfg['python'],'-u',*map(str,cli)]
 rec=dict(tag=tag,jobid=250285,gpu=gpu,command=cmd,start=datetime.datetime.now().isoformat(),status='running',environment={k:te[k] for k in env})
 dump(E/'commands'/f'{tag}.json',rec);start=time.monotonic()
 with (E/'logs'/f'{tag}.log').open('xb') as f:
  proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(E/'commands'/f'{tag}.json',rec);code=proc.wait()
 hours=(time.monotonic()-start)/3600;spent+=hours;rec.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=hours,status='completed' if code==0 else 'failed');dump(E/'commands'/f'{tag}.json',rec)
 if code:raise RuntimeError(tag+' failed; results retained')
def wait_existing(folder,terminal):
 while not terminal.exists():
  st=folder/'status.json'
  if st.exists() and json.loads(st.read_text()).get('stage')=='failed':raise RuntimeError('Existing stage failed '+str(folder))
  time.sleep(10)
try:
 status('waiting_Gamma_and_full_k2_spectrum')
 wait_existing(E/'si_gamma_equivalence',E/'si_gamma_equivalence/result.json')
 if not json.loads((E/'si_gamma_equivalence/result.json').read_text())['passed']:raise RuntimeError('Gamma reduction failed')
 wait_existing(E/'si_k2_spectrum',E/'si_k2_spectrum/summary.json')
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5子步骤Γ退化PASS：新KMeshResponse与GammaResponse的完整12维梯度、Hessian及外部扰动一致；k2独立全8点谱已生成，耦合/FD仍待完成。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5 Γ退化验收PASS；Si k2耦合与FD进行。\n')
 print('E5 子步骤：Γ退化 PASS。\n证据：si_gamma_equivalence/result.json。\n继续 k2公共密度耦合、两模式及FD。',flush=True)
 status('k2_full_mode_running')
 run('si_k2_response',2,[R/'validation/workflows/response_design/reverse_response/e5_response_check.py','--protocol',E/'protocol.json','--source',E/'si_k2_base','--orbitals',E/'si_k2_spectrum','--output',E/'si_k2_response'])
 d0=np.zeros(12);d0[0]=1.;d1=np.array([.1,-.07,.05,.08,-.03,.06,-.3,.2,-.1,.3,-.2,.1]);d1/=np.linalg.norm(d1)
 dirs=[d0.tolist(),d1.tolist()];steps=[.0001,.00005]
 dump(E/'fd_directions.json',dict(directions=dirs,steps=steps,frozen_before_FD=True,definition='d0 strain xx checks dk/dstrain; d1 strain + balanced Cartesian Bohr, fixed radial/Miller program'))
 records=[]
 for i,d in enumerate(dirs):
  for h in steps:
   for sign in [-1,1]:
    tag=f'si_k2_fd_d{i}_h{h:g}_s{sign:+d}';out=E/tag;theta=np.asarray(cfg['theta0'])+sign*h*np.asarray(d)
    x,y,z,yz,xz,xy=theta[:6];strain=[[x,xy/2,xz/2],[xy/2,y,yz/2],[xz/2,yz/2,z]]
    status('k2_FD_running',tag=tag,completed=len(records),spent_GPU_hours=spent)
    run(tag,2,[R/'validation/workflows/response_design/reverse_response/e5_fd_point.py','--radial-parent',E/'si_k2_base',
     '--structure',E/'inputs/si_structure.json','--pp-dir',E/'inputs/si_pp','--output',out,'--cutoff','20','--grid','40','--k','2','--epochs','4000',
     '--strain',json.dumps(strain),'--displacements',json.dumps((theta[6:].reshape(2,3)/1.8897259886).tolist()),'--local-radial-quadrature','simpson','--require-stationary','--resume-from',E/'si_k2_base','--resume-geometry-continuation'])
    spectrum=E/(tag+'_spectrum');run(tag+'_spectrum',2,[R/'validation/workflows/response_design/reverse_response/e5_mesh_spectrum.py','--source',out,'--output',spectrum])
    records.append(dict(direction=i,h=h,sign=sign,point=str(out),spectrum=str(spectrum)));dump(E/'fd_index.json',records)
 with (E/'logs/si_prototype_acceptance.log').open('xb') as f:r=subprocess.run(['python','-u',str(P/'validation/workflows/response_design/e5_accept_si.py'),str(E)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Si k2 independent h/h2 FD failed')
 status('Si_k2_coupled_prototype_PASS_production_pending',spent_GPU_hours=spent)
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5 Si k2原型PASS：12分量mode、公共密度非零跨k耦合、Γ退化及两方向h/h2重优化FD通过；单卡，仅原型，不等于MgSiN2/k4生产完成。证据E5/si_prototype_acceptance.json。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5 Si k2耦合原型及FD通过；后续分片和MgSiN2尚待做。\n')
 print('E5 Si k2原型：全部门 PASS。\n证据：si_prototype_acceptance.json。\n多卡MgSiN2设计/k4验证仍待开发，不宣称生产完成。',flush=True)
except Exception as exc:status('failed',error=str(exc));raise
