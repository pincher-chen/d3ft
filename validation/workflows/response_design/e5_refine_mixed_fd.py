"""Two smaller mixed-FD points; retain original failed gate and tolerances."""
from pathlib import Path
import json,datetime,os,subprocess,time
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_kmesh';cfg=json.loads((E/'protocol.json').read_text());R=Path(cfg['runtime']);base=json.loads((E/'commands/si_k2_base.json').read_text());env=base['environment'].copy()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):dump(E/'refinement_status.json',dict(stage=stage,time=datetime.datetime.now().isoformat(),jobid=250285,**kw));print('E5_REFINEMENT',stage,kw,flush=True)
def run(tag,cli):
 temp='/dev/shm/response_design_e5_'+tag;te={**os.environ,**env,'TMPDIR':temp,'JAX_COMPILATION_CACHE_DIR':temp+'/jax_cache','PYTHONPYCACHEPREFIX':temp+'/pycache'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5','env','CUDA_VISIBLE_DEVICES=2','timeout','1800',cfg['python'],'-u',*map(str,cli)]
 rec=dict(tag=tag,jobid=250285,gpu=2,start=datetime.datetime.now().isoformat(),command=cmd,status='running');dump(E/'commands'/f'{tag}.json',rec);start=time.monotonic()
 with (E/'logs'/f'{tag}.log').open('xb') as f:child=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=child.pid;dump(E/'commands'/f'{tag}.json',rec);code=child.wait()
 rec.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=(time.monotonic()-start)/3600,status='completed' if code==0 else 'failed');dump(E/'commands'/f'{tag}.json',rec)
 if code:raise RuntimeError('Refined step failed '+tag)
try:
 failed=json.loads((E/'si_prototype_acceptance.json').read_text());mixed=[r for r in failed['FD'] if r['direction']==1]
 if not mixed[1]['error']<.4*mixed[0]['error']:raise RuntimeError('Not resolved second-order FD trend, do not assume step refinement fixes it')
 old=json.loads((E/'fd_directions.json').read_text());dirs=old['directions']
 dump(E/'fd_directions_refined.json',dict(time=datetime.datetime.now().isoformat(),directions=dirs,steps_per_direction=[[.0001,.00005],[.00005,.000025]],
  refinement_preregistered=True,original_gate_preserved=True,thresholds_unchanged=True,reason='Mixed h/h2 truncation ratio3.985; CBM separation1.5165e-5Ha. Same target, branch, solver, masks, tables; smaller steps only'))
 index=[r for r in json.loads((E/'fd_index.json').read_text()) if r['direction']==0 or r['h']==.00005]
 for sign in [-1,1]:
  h=.000025;tag=f'si_k2_refined_fd_d1_h{h:g}_s{sign:+d}';out=E/tag;theta=np.array(cfg['theta0'])+sign*h*np.array(dirs[1]);x,y,z,yz,xz,xy=theta[:6];strain=[[x,xy/2,xz/2],[xy/2,y,yz/2],[xz/2,yz/2,z]]
  status('running',tag=tag)
  run(tag,[R/'validation/workflows/response_design/reverse_response/e5_fd_point.py','--radial-parent',E/'si_k2_base','--structure',E/'inputs/si_structure.json','--pp-dir',E/'inputs/si_pp','--output',out,'--cutoff','20','--grid','40','--k','2','--epochs','4000','--strain',json.dumps(strain),'--displacements',json.dumps((theta[6:].reshape(2,3)/1.8897259886).tolist()),'--local-radial-quadrature','simpson','--require-stationary','--resume-from',E/'si_k2_base','--resume-geometry-continuation'])
  spec=E/(tag+'_spectrum');run(tag+'_spectrum',[R/'validation/workflows/response_design/reverse_response/e5_mesh_spectrum.py','--source',out,'--output',spec]);index.append(dict(direction=1,h=h,sign=sign,point=str(out),spectrum=str(spec)));dump(E/'fd_index_refined.json',index)
 with (E/'logs/si_refined_acceptance.log').open('xb') as f:r=subprocess.run(['python','-u',str(P/'validation/workflows/response_design/e5_accept_si.py'),str(E),'--refined'],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Refined mixed FD still fails; do not certify prototype')
 status('Si_k2_prototype_PASS',original_FD_failure_preserved=True)
 with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5 Si k2原型完整PASS：Γ退化、12维mode、共享密度跨k作用、纯应变h/h2及混合h/2/h/4重优化FD全部通过；初始混合粗步长失败保留，原阈值不变。证据E5/si_prototype_acceptance_refined.json。仍是单卡原型，多卡MgSiN2优化/k4终点未做。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E5 Si k2耦合原型完整验收PASS，粗FD失败保留；多卡/材料设计待推进。\n')
 print('E5 Si k2原型：Γ退化、mode、跨k及细FD全部PASS。\n证据：si_prototype_acceptance_refined.json；初始失败保留。\n尚未完成多卡MgSiN2及k4生产，论文未改。',flush=True)
except Exception as exc:status('failed',error=str(exc));raise
