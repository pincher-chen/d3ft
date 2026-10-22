"""GPU0: memory/two-step gate -> full modes -> independent hydro FD -> design.

Existing E6 hydro states are preferred. If E6 terminally failed with incomplete
states, compute a new quartet locally; never overwrite its failed evidence.
All new electronic states and spectra have immutable unique names and strict
native gates. The k4 transfer remains an independent unfinished experiment.
"""
import argparse,datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from run_identity import reserve_name,await_detached_record
from joint_design import run_design,deform,matrix,nearest_distance
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_distributed';A=E/'four_card_20261015'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();T=args.root
p=json.loads((T/'smoke_protocol.json').read_text());spent=p['prior_GPU_hours'];R=T/'pipeline_runtime';GPU=int(p.get('workflow_gpu',0))
PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(Path(q).read_text())
def dump(q,r):
 q=Path(q);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def sha(q):return __import__('hashlib').sha256(Path(q).read_bytes()).hexdigest()
def status(stage,**kw):
 row=dict(time=now(),stage=stage,jobid=250285,gpus=[GPU],spent_GPU_hours=spent,root=str(T),**kw)
 for q in [T/'status.json',E/'queue_status.json',A/'queue_status.json']:dump(q,row)
def note(s):
 for q in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
  with q.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
def run(script,argv,logical,seconds):
 global spent
 if spent+seconds/3600>300:raise RuntimeError('E5 300 GPUh hard cap; no next calculation submitted')
 tag=reserve_name(T,logical,log_parent=T);tmp='/dev/shm/'+tag;env={**os.environ,**read(T/'smoke_launch.json')['environment']}
 env.update(PYTHONPATH=str(R)+':'+env['PYTHONPATH'].split(':')[-1],TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',D3FT_RESPONSE_SINGLE_DEVICE='1')
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E5-single-pipeline','env','CUDA_VISIBLE_DEVICES='+str(GPU),'timeout',str(seconds),PY,'-u',str(R/'validation/workflows/response_design/reverse_response'/script),*argv]
 f=(T/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(time=now(),pid=proc.pid,jobid=250285,gpus=[GPU],command=cmd,environment={k:env[k] for k in read(T/'smoke_launch.json')['environment']},timeout_seconds=seconds,status='running');dump(T/(tag+'_launch.json'),rec)
 while proc.poll() is None:status(logical,pid=proc.pid,elapsed_seconds=time.monotonic()-start);time.sleep(15)
 f.close();h=(time.monotonic()-start)/3600;spent+=h;rec.update(end=now(),returncode=proc.returncode,GPU_hours=h,status='completed' if not proc.returncode else 'failed');dump(T/(tag+'_launch.json'),rec)
 if proc.returncode:raise RuntimeError(logical+' failed; immutable log '+tag+'.log')
def unique(logical):return T/reserve_name(T,logical,output_parent=T)
def response(source,spectra,theta,logical,tangents):
 out=unique(logical);cfg=dict(p,source=str(source),fresh_scf_directory=str(source),spectrum_directories=[str(x) for x in spectra],theta0=np.asarray(theta).tolist(),pcg_smoke_only=False,tangent_directions=tangents)
 proto=T/(out.name+'_protocol.json');dump(proto,cfg)
 run('e5_mg_sharded_driver.py',['--protocol',str(proto),'--stage','response','--output',str(out)],logical,43200)
 result=read(out/'result.json')
 if not result['passed'] or not result['memory']['passed']:raise RuntimeError('Single-device full response mode/memory gate failed')
 return out,result
def spectra(source,theta,logical):
 dirs=[unique(logical+'_k'+str(k)) for k in range(8)];cfg=dict(p,source=str(source),fresh_scf_directory=str(source),spectrum_directories=[str(x) for x in dirs],theta0=np.asarray(theta).tolist(),pcg_smoke_only=False,spectrum_restart={},spectrum_maxiter=1200)
 proto=T/(logical+'_spectrum_protocol.json');dump(proto,cfg)
 for k,out in enumerate(dirs):
  run('e5_mg_sharded_driver.py',['--protocol',str(proto),'--stage','spectrum','--k-index',str(k),'--output',str(out)],logical+'_k'+str(k),5400)
  s=read(out/'summary.json')
  if s['source_state_sha256']!=sha(Path(source)/'state.pkl') or max(s['residuals_Ha'])>1e-9:raise RuntimeError('Local spectrum hash/Ritz gate failed')
 return dirs
def target(dirs):
 sys.path.insert(0,str(R/'validation/workflows/response_design/reverse_response'))
 from mesh_target import select_mesh_target
 rows=[read(Path(d)/'summary.json') for d in dirs];_,info=select_mesh_target(np.array([r['eigenvalues_Ha'] for r in rows]),48)
 return rows,info
def native(source):
 pt=read(Path(source)/'point.json')
 if not pt['stationarity_pass'] or pt['residual']>1e-8 or pt['acoustic_sum_rule_Ha_Bohr']>5e-8 or abs(pt['charge']-96)>96e-8:raise RuntimeError('Fresh native physical gate failed')
 return pt
def hydro_FD(result):
 H=Path(read(P/'validation/results/response_design_20261015_e6_k2_hydro/active_corrected_queue.json')['root']);vals=[]
 while True:
  f=H/'hydro_values.json';vals=read(f) if f.exists() else []
  if len(vals)==4 and all(v.get('passed') for v in vals):break
  state=read(H/'status.json')
  if state['stage']=='failed':break
  status('waiting_E6_independent_hydro_quartet',E6_root=str(H));time.sleep(30)
 if len(vals)!=4 or not all(v.get('passed') for v in vals):
  vals=[]
  for h in [.002,.001]:
   for sign in [-1,1]:
    out=unique('FD_nu'+str(sign*h))
    run('e6_k2_hydro_point.py',['--source',p['fresh_scf_directory'],'--nu',str(sign*h),'--epochs','4000','--devices','1','--stream-k','--output',str(out)],'FD_hydro_SCF',14400);native(out)
    dirs=spectra(out,p['theta0'],out.name);_,info=target(dirs)
    vals.append(dict(h=h,sign=sign,source=str(out),passed=True,fundamental_eV=info['hard_gap_Ha']*27.211386245988))
  dump(T/'new_hydro_values.json',vals)
 else:
  for v in vals:
   d=Path(v['source']);native(d)
   if read(d/'metadata.json')['source']!=p['fresh_scf_directory']:raise RuntimeError('E6 FD source differs from response source')
 theta=np.asarray(p['theta0']);direction=np.r_[(1+theta[:3])/3,theta[3:6]/3,theta[6:]/3];ad=float(np.asarray(result['gradient_eV'])@direction);checks=[]
 for h in [.002,.001]:
  pair={v['sign']:v for v in vals if v['h']==h};fd=(pair[1]['fundamental_eV']-pair[-1]['fundamental_eV'])/(2*h);tol=3e-5+.002*abs(ad);checks.append(dict(h=h,FD_eV=fd,adjoint_eV=ad,error_eV=abs(fd-ad),tolerance_eV=tol,passed=abs(fd-ad)<=tol))
 cert=dict(time=now(),passed=all(c['passed'] for c in checks),checks=checks,points=vals,scope='Independent clamped-fractional hydro lnV h/h2 FD, not all 54 directions or k4 convergence',E6_root=str(H));dump(T/'FD_hydro_acceptance.json',cert)
 if not cert['passed']:raise RuntimeError('Original hydro FD gate failed; design not submitted')
class Backend:
 kind='dft'
 def __init__(self,full):
  self.full=full;self.theta=np.asarray(p['theta0']);m=read(Path(p['fresh_scf_directory'])/'metadata.json');phys=m['physical'];f=np.eye(3)+matrix(self.theta[:6]);self.basecell=np.asarray(phys['cell_angstrom'])*1.8897259886@np.linalg.inv(f).T;self.basepos=(np.asarray(phys['positions_angstrom'])*1.8897259886-self.theta[6:].reshape(-1,3))@np.linalg.inv(f).T
 def geometry(self,theta):return deform(self.basecell,self.basepos,theta)
 def evaluate(self,theta,folder,parent):
  folder=Path(folder);folder.mkdir();theta=np.asarray(theta)
  if parent is None:
   source=Path(p['fresh_scf_directory']);dirs=[Path(x) for x in p['spectrum_directories']]
  else:
   source=folder/'SCF';dump(folder/'theta.json',theta.tolist());dump(folder/'parent_theta.json',parent['theta'])
   run('e6_k2_hydro_point.py',['--source',parent['state_directory'],'--theta-json',str(folder/'theta.json'),'--parent-theta-json',str(folder/'parent_theta.json'),'--epochs','4000','--devices','1','--stream-k','--output',str(source)],'joint_design_SCF',14400);dirs=spectra(source,theta,folder.name+'_'+folder.parent.name)
  point=native(source);rows,info=target(dirs);cell,pos=self.geometry(theta)
  out=dict(energy_Ha=point['energy_Ha'],gap_eV=info['hard_gap_Ha']*27.211386245988,residual_Ha=point['residual'],forces_Ha_Bohr=point['forces_Ha_Bohr'],ASR_Ha_Bohr=point['acoustic_sum_rule_Ha_Bohr'],charge=point['charge'],electrons=96.,ritz_max_Ha=max(max(r['residuals_Ha']) for r in rows),nearest_distance_bohr=nearest_distance(cell,pos),cell_determinant_bohr3=float(np.linalg.det(cell)),branch_margin_Ha=min(x for x in info['margins'].values() if x is not None),electronic_reoptimized=True,state_directory=str(source),spectrum_directories=[str(x) for x in dirs],theta=theta.tolist(),reused_accepted_initial_state=parent is None)
  dump(folder/'readout.json',out);return out
 def gradient(self,theta,current,folder):
  folder=Path(folder);folder.mkdir()
  if current['state_directory']==p['fresh_scf_directory']:r=self.full
  else:_,r=response(current['state_directory'],current['spectrum_directories'],theta,'design_adjoint',[])
  dump(folder/'response.json',r);return np.asarray(r['gradient_eV'])
try:
 if p.get('resume_full_response'):
  result=read(Path(p['resume_full_response']))
  if not result['passed'] or not result['memory']['passed'] or sha(Path(p['fresh_scf_directory'])/'state.pkl')!=p['resume_source_state_sha256']:raise ValueError('Accepted full response/source binding changed')
  for gate in p['required_acceptance_files']:
   if not read(Path(gate))['passed']:raise ValueError('Required resume gate failed: '+gate)
  status('accepted_full_response_reused_independent_FD_pending')
 elif p.get('resume_probe_evidence'):
  out=Path(p['resume_probe_evidence']);gate=read(out/'smoke_acceptance.json')
  iterations=gate['iterations']
  if gate['inner_comparisons'] or len(iterations)!=2 or not gate['memory']['passed'] or gate['memory']['peak_GiB']>=78 or abs(gate['charge']-96)>96e-8 or gate['primal_residual_Ha']>1e-7 or not (0<iterations[1]['relative_residual']<iterations[0]['relative_residual']<1):raise RuntimeError('Prior physical/memory probe is not the registered logging-only failure')
  cert=dict(passed=True,source=str(out),source_sha256=sha(out/'smoke_acceptance.json'),scope='Previously measured memory/primal/charge/two completed PCG updates ONLY; missing CPU inner gate must pass online before iteration3',old_smoke_marked_passed=False,online_gate_required=True)
  dump(T/'prior_probe_physical_memory_acceptance.json',cert);p['required_acceptance_files'].append(str(T/'prior_probe_physical_memory_acceptance.json'));p['validate_first_two_inner']=True
  status('direct_full_response_with_online_first_two_gate')
 else:
  status('single_GPU_memory_and_two_iterations_running')
  rec=await_detached_record(T/'smoke_launch.json');spent+=rec['GPU_hours'];dump(T/'smoke_observed_completion.json',rec);out=Path(rec['output'])
  gate=read(out/'smoke_acceptance.json');end=read(out/'status.json')
  if not gate['passed'] or not gate['memory']['passed'] or end.get('stage')!='completed':raise RuntimeError('Single-GPU smoke/memory gate failed; full response not submitted')
  p['required_acceptance_files'].append(str(out/'smoke_acceptance.json'))
 shutil.copytree(T/'runtime',R)
 if not p.get('resume_full_response'):
  for rel in ['validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py','validation/workflows/response_design/reverse_response/e6_k2_hydro_point.py']:shutil.copy2(P/rel,R/rel)
 p.update(runtime=str(R),pcg_smoke_only=False,response_gpus=[GPU]);dump(T/'protocol.json',p)
 if p.get('resume_full_response'):
  full=Path(p['resume_full_response']).parent;result=read(full/'result.json')
 else:
  full,result=response(p['fresh_scf_directory'],p['spectrum_directories'],p['theta0'],'full_single_adjoint_forward',p['tangent_directions'])
  dump(A/'response_index.json',dict(result_file=str(full/'result.json'),protocol=str(T/'protocol.json'),source_state_sha256=sha(Path(p['fresh_scf_directory'])/'state.pkl')))
 note('E5单卡流式完整adjoint/3 forward/平移及78GiB显存门PASS；接独立静水h/h2 FD，尚非设计成功。')
 hydro_FD(result);note('E5单卡流式原静水h/h2 FD门PASS，自动接有界54维k2设计，目标相对起点+0.02eV、最多2更新，沿用E4物理/距离/力/步长门。')
 limits=read(P/'validation/results/response_design_20261015_e4_joint_design/mg_protocol.json')['limits'];design=unique('bounded_k2_design')
 result=run_design(design,p['theta0'],lambda initial:initial['gap_eV']+.02,limits,Backend(result),max_updates=2,max_backtracks=3)
 status('single_GPU_k2_pipeline_finished_k4_transfer_pending',design_result=result,design_directory=str(design),k4_transfer_completed=False)
 note('E5单卡k2设计结束：接受更新'+str(result['completed_updates'])+'，目标达成='+str(result['target_reached'])+'；只报告真实接受结果，k4转移仍待。')
except Exception as exc:status('failed',passed=False,error=str(exc));note('E5单卡管线停止：'+str(exc)+'；全部失败保留，不绕过显存/驻点/FD门。');raise
finally:
 Q=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json';r=read(Q)
 if r.get(str(GPU),{}).get('status_file')==str(T/'status.json'):r.pop(str(GPU))
 dump(Q,r)
