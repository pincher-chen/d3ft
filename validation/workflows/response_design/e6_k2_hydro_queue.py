"""Fixed GPU6/7 fresh hydro h/h2, after E5's independently accepted k2 SCF."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import argparse
from run_identity import reserve_name,unique_output_path
import datetime,json,os,subprocess,time
import numpy as np
P=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=P/'validation/results/response_design_20261015_e6_k2_hydro');args=ap.parse_args();E=args.root
p=json.loads((E/'protocol.json').read_text());R=Path(p['runtime']);base=Path(p['base_source']);E5=P/'validation/results/response_design_20261015_e5_distributed/four_card_20261015'
PY='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
env=json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment'];env['PYTHONPATH']=str(R)+':'+env['PYTHONPATH'].split(':')[-1]
GPUS=p['gpus'];spent=0.;rows=[]
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(q.read_text()) if q.exists() else {}
def dump(q,d):
 t=q.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(q)
def status(stage,**kw):
 d=dict(time=now(),stage=stage,jobid=250285,gpus=GPUS,spent_GPU_hours=spent,**kw);dump(E/'status.json',d)
 if p.get('status_mirror'):dump(Path(p['status_mirror']),d)
def note(s):
 for q in (P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md'):
  with q.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
def idle():
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 d={int(s.split(',')[0]):int(s.split(',')[1]) for s in r.stdout.splitlines()};return all(d[g]<=3 for g in GPUS)
def launch(tag,gs,seconds,script,args):
 logical_tag=tag;tag=reserve_name(E,tag,log_parent=E)
 if spent+len(gs)*seconds/3600>60:raise RuntimeError('E6 60 GPUh cap')
 tmp='/dev/shm/response_design_e6_'+tag;te={**os.environ,**env,'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax'}
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e6','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gs)),'timeout',str(seconds),PY,'-u',str(R/'validation/workflows/response_design/reverse_response'/script),*args]
 rec=dict(time=now(),tag=tag,logical_tag=logical_tag,jobid=250285,gpus=gs,command=cmd,environment={k:te[k] for k in env},status='running');f=(E/(tag+'.log')).open('xb');t=time.monotonic();proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(E/(tag+'_launch.json'),rec);return proc,f,t,rec
def finish(job):
 global spent
 proc,f,t,rec=job;c=proc.wait();f.close();h=len(rec['gpus'])*(time.monotonic()-t)/3600;spent+=h;rec.update(end=now(),returncode=c,GPU_hours=h,status='completed' if not c else 'failed');dump(E/(rec['tag']+'_launch.json'),rec);return c
def gaps(values):
 v=np.asarray(values);n=48;vb=int(np.argmax(v[:,n-1]));cb=int(np.argmin(v[:,n]));k=np.asarray(read(base/'point.json')['k_fractional']);gamma=int(np.argmin(np.linalg.norm(k-np.rint(k),axis=1)))
 return dict(fundamental_eV=float((v[cb,n]-v[vb,n-1])*27.211386245988),Gamma_eV=float((v[gamma,n]-v[gamma,n-1])*27.211386245988),vbm_k=vb,cbm_k=cb)
try:
 status('waiting_new_E5_k2_SCF_and_fixed_cards',formal_E6_complete=False)
 for gate in p.get('extra_acceptance_files',[]):
  while not Path(gate).exists():time.sleep(20)
  if not read(Path(gate)).get('passed'):raise RuntimeError('Single-GPU streaming CPU equivalence gate failed')
 while not read(Path(p['depends_on'])).get('passed'):
  if read(E5/'queue_status.json').get('stage')=='failed':raise RuntimeError('E5 parent SCF unavailable; no E6 GPU task submitted')
  time.sleep(20)
 base=Path(read(Path(p['depends_on'])).get('source',p['base_source']))
 for h in p['hydro_steps']:
  for sign in (-1,1):
   while not idle():time.sleep(20)
   tag=reserve_name(E,f'nu{sign*h:+.6f}',output_parent=E);point=E/tag
   status('hydro_k2_reoptimization',nu=sign*h)
   args=['--source',str(base),'--nu',str(sign*h),'--epochs','4000','--devices',str(len(GPUS)),'--output',str(point)]
   if p.get('stream_k'):args+=['--stream-k']
   if finish(launch(tag+'_SCF',GPUS,p.get('SCF_timeout_seconds',5400),'e6_k2_hydro_point.py',args)):
    rows.append(dict(h=h,sign=sign,passed=False,stage='SCF'));continue
   specroot=E/(tag+'_spectrum');specroot.mkdir()
   cfg=read(E5/'mg_k2_protocol.json');cfg.update(source=str(base),fresh_scf_directory=str(point),spectrum_root=str(specroot),spectrum_directories=[str(unique_output_path(specroot/f'k{k}')) for k in range(8)])
   specproto=specroot/'protocol.json';dump(specproto,cfg);codes=[]
   status('hydro_local_spectra',nu=sign*h)
   for k in range(0,8,len(GPUS)):
    if spent+len(GPUS)>60:raise RuntimeError('E6 spectrum wave exceeds budget')
    jobs=[launch(tag+'_k'+str(ik),[gpu],3600,'e5_mg_sharded_driver.py',['--protocol',str(specproto),'--stage','spectrum','--k-index',str(ik),'--output',cfg['spectrum_directories'][ik]]) for ik,gpu in zip(range(k,min(k+len(GPUS),8)),GPUS)]
    codes.extend(finish(j) for j in jobs)
   if any(codes):rows.append(dict(h=h,sign=sign,passed=False,stage='spectrum'));continue
   values=[read(Path(cfg['spectrum_directories'][ik])/'summary.json')['eigenvalues_Ha'] for ik in range(8)]
   record=dict(h=h,sign=sign,nu=sign*h,passed=True,source=str(point),**gaps(values));rows.append(record);dump(E/'hydro_values.json',rows)
   note(f'E6 k2 lnV={sign*h:+g}新态与8点谱完成，固定分数坐标；不是k4或文献验收。')
 if len(rows)!=4 or not all(x['passed'] for x in rows):raise RuntimeError('Hydro quartet incomplete; retain native failures')
 derivatives=[]
 for h in p['hydro_steps']:
  pair={r['sign']:r for r in rows if r['h']==h}
  derivatives.append(dict(h=h,**{key:(pair[1][key]-pair[-1][key])/(2*h) for key in ('fundamental_eV','Gamma_eV')}))
 dump(E/'FD_h_h2.json',dict(time=now(),derivatives=derivatives,points=rows,ionic_condition=p['ionic_condition'],not_k4_certificate=True,not_literature_PASS=True))
 # Response may run concurrently on disjoint cards. Do not invent its result.
 while not Path(p['base_response']).exists() and not read(E5/'response_index.json').get('result_file') and read(E5/'queue_status.json').get('stage')!='failed':time.sleep(20)
 p['base_response']=read(E5/'response_index.json').get('result_file',p['base_response'])
 response=read(Path(p['base_response']));theta=np.asarray(p['theta0']);direction=np.r_[(1+theta[:3])/3,theta[3:6]/3,theta[6:]/3]
 adjoint=float(np.asarray(response['gradient_eV'])@direction) if response.get('passed') else None
 checks=[]
 if adjoint is not None:
  checks=[dict(h=d['h'],error_eV=abs(d['fundamental_eV']-adjoint),tolerance_eV=3e-5+.002*abs(adjoint),passed=abs(d['fundamental_eV']-adjoint)<=3e-5+.002*abs(adjoint)) for d in derivatives]
 result=dict(time=now(),k2_pilot_passed=len(checks)==2 and all(c['passed'] for c in checks),checks=checks,adjoint_fundamental_A_eV=adjoint,FD=derivatives,
  spent_GPU_hours=spent,formal_E6_complete=False,not_literature_PASS=True,not_k4_certificate=True,definition='dEg/dlnV, clamped fractional positions; engineering prestrain hydro direction',ionic_condition=p['ionic_condition'])
 dump(E/'k2_hydro_acceptance.json',result);status('k2_hydro_pilot_finished_k4_literature_pending',k2_pilot_passed=result['k2_pilot_passed'],formal_E6_complete=False)
 note('E6新k2静水h/h2及伴随对照'+('PASS' if result['k2_pilot_passed'] else '未通过/响应未完成')+'；k4、轴映射、文献离子协议仍未验，E6不记正式完成。')
 print('E6 k2静水四点与独立谱流程结束。\n伴随/FD门 '+('PASS' if result['k2_pilot_passed'] else '未通过')+'。\nk4与文献协议仍待验收。',flush=True)
except Exception as exc:
 status('failed',error=str(exc),formal_E6_complete=False);dump(E/'hydro_values.json',rows);note('E6 k2 pilot失败/阻塞：'+str(exc)+'；失败保留。');raise
