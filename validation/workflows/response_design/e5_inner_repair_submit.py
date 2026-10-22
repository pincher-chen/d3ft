"""CPU gate before a fresh two-GPU E5 response attempt; no spectrum rerun."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e5_distributed'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();T=a.root
def now():return datetime.datetime.now().isoformat()
def read(q):return json.loads(Path(q).read_text())
def dump(q,r):
 q=Path(q);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(q)
def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,jobid=250285,gpus=[0,5],**kw))
try:
 gate=T/'cpu4/acceptance.json';deadline=time.monotonic()+540
 while not gate.exists():
  if time.monotonic()>deadline:raise RuntimeError('CPU gate timed out; GPU not submitted')
  state('waiting_CPU_four_device_gate');time.sleep(5)
 if not read(gate)['passed']:raise RuntimeError('CPU gate FAIL; GPU not submitted')
 old=Path(read(T/'previous_attempt.json')['root']);p=read(old/'protocol.json');R=T/'runtime'
 shutil.copytree(Path(p['runtime']),R)
 rel=Path('validation/workflows/response_design/reverse_response/projected_solver.py');shutil.copy2(P/rel,R/rel)
 dump(T/'runtime_patch_sha256.json',dict(old_solver=str(Path(p['runtime'])/rel),old_sha256=hashlib.sha256((Path(p['runtime'])/rel).read_bytes()).hexdigest(),new_solver=str(R/rel),new_sha256=hashlib.sha256((R/rel).read_bytes()).hexdigest(),CPU_gate=str(gate)))
 p.update(runtime=str(R),adopted_k_indices=list(range(8)),prior_GPU_hours=read(old/'status.json')['spent_GPU_hours'])
 p['required_acceptance_files']=[*p['required_acceptance_files'],str(gate)]
 dump(T/'protocol.json',p)
 cmd=['python','-u',str(P/'validation/workflows/response_design/e5_resume_spectra_response.py'),'--root',str(T)]
 f=(T/'controller.log').open('xb');pr=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 dump(T/'controller_launch.json',dict(time=now(),pid=pr.pid,command=cmd,jobid=250285,gpus=[0,5],CPU_gate_passed=True))
 dump(E/'active_followup.json',dict(root=str(T),previous_root=str(old)))
 print('CPU gate PASS; detached E5 controller '+str(pr.pid)+' submitted on GPU0/5.',flush=True)
except Exception as exc:state('failed',passed=False,error=str(exc),GPU_submitted=False);raise
