"""CPU-only migration/adoption regression and remote env boundary check."""
import copy,datetime,hashlib,importlib.util,json,subprocess,tempfile
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parents[3]
f=W/'controller_hn48.py';spec=importlib.util.spec_from_file_location('migrated',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);root=m.ROOT
hashfile=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
checks={};old=m.read(root/'M3-A/protocol.json');new=m.PRO['M3-A']
assert new['node']=='hn48' and new['jobid']==275669 and new['gpus']==[0,1,2,3]
for key,value in old.items():
 if key not in ['node','jobid','gpus']:assert new[key]==value
checks['M3_only_node_allocation_card_numbers_change']=True
prior=[r for r in m.JOBS if r['experiment']=='M3-A' and r['label'].endswith('_r2')]
def pointkey(r):
 a=r['command'];return(int(a[a.index('--direction')+1]),float(a[a.index('--h')+1]),int(a[a.index('--sign')+1]))
keys={pointkey(r) for r in prior};allkeys={(i,h,s) for i in range(5) for h in old['steps'] for s in [-1,1]};assert len(keys)==4 and len(allkeys-keys)==16 and len(keys|allkeys-keys)==20
checks['adopt4_old_hn35_points_plus16_new_hn48_no_duplication']=True
s=f.read_text();assert "for j in ['M4-A','M5-A']" in s and "for j in ['M4-A','M3-A','M5-A']" not in s;assert "def snapshot(node='hn35',jobid=250285)" in s;assert "snap=snapshot()" in s;assert "exclusive_violation.json" in s
checks['M2_only_M4_M5_and_all8_hn35_idle_no_hn48_dependency']=True
for k in ['M4-A','M5-A','M2-A']:assert m.PRO[k]['jobid']==250285 and m.PRO[k]['node']=='hn35'
checks['no_other_experiment_migrates_or_times_onhn48']=True
checks['history_retained_shared25GPUh_budget']=old['GPU_hours_cap']==25 and len(m.JOBS)==len(m.read(root/'ledger_before_hn48_migration.json')['jobs'])
# Native final artifacts drive adoption after old waitpid parent exits. Never kill GPU workers.
realroot=m.ROOT;real_ledger=m.ledger
with tempfile.TemporaryDirectory() as d:
 m.ROOT=Path(d);m.ledger=lambda:None
 for okay in [True,False]:
  out=Path(d)/str(okay);out.mkdir();(out/'status.json').write_text(json.dumps({'stage':'completed' if okay else 'failed'}));r={'pid':99999999,'label':'test_'+str(okay),'experiment':'M3-A','node':'hn35','gpu':1,'output':str(out)}
  got=m.adopt_wait(r);assert got['returncode']==(0 if okay else 1) and got['raw_srun_returncode'] is None
 m.ROOT=realroot;m.ledger=real_ledger
checks['adopt_completed_vs_failed_CPU_simulation_without_GPU_restart']=True
cmd=['srun','--jobid=275669','--overlap','--nodelist=hn48','--cpu-bind=none','--gpu-bind=none','-N1','-n1','env','CUDA_VISIBLE_DEVICES=2','/usr/bin/python3','-c','import os,json;assert os.environ["CUDA_VISIBLE_DEVICES"]=="2";assert os.environ["SLURM_JOB_ID"]=="275669";assert os.environ["SLURMD_NODENAME"]=="hn48";print(json.dumps({k:os.environ[k] for k in ["CUDA_VISIBLE_DEVICES","SLURM_JOB_ID","SLURMD_NODENAME"]}))']
x=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=45);checks['remote_hn48_env_binding_CPU_only']=json.loads(x.stdout)
for k in m.PRO:assert hashfile(root/k/'protocol.json')==m.read(root/'CPU_acceptance.json')['protocol_SHA256'][k]
checks['all_original_protocol_SHA_unchanged']=True
result={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'controller_SHA256':hashfile(f),'amendment_SHA256':hashfile(root/'resource_amendment_hn48_20261021.json')}
(root/'hn48_migration_CPU_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
