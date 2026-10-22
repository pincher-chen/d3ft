"""Background queues for the existing response Gamma states, with bounded audits."""
import concurrent.futures,datetime,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];A=P/'validation/results/response_design_20261014_e2_audit';E=P/'validation/results/response_design_20261014_e0_migration';env0=json.loads((E/'commands/alp_new_retry1.json').read_text())['environment'];py='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
fd=os.open(A/'queue.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
inventory=json.loads((A/'inventory.json').read_text());pending=[];seen=set()
for r in inventory['states']:
 pair=(r['source'],r['orbitals'])
 if r['passed'] or not r['independent_low_orbitals_exist'] or pair in seen:continue
 seen.add(pair);pending.append(r)
manifest=dict(start=datetime.datetime.now().isoformat(),jobid=250285,gpus=[1,3],states=pending,status='running',per_audit_timeout_seconds=1800,max_GPU_hours=len(pending)*.5)
(A/'queue_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def run_lane(gpu,states):
 for row in states:
  name=row['label'];out=A/name;log=A/(name+'.log');env=env0.copy();tmp='/dev/shm/response_design_e2_'+name;env.update(TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',PYTHONPYCACHEPREFIX=tmp+'/pycache');env['PYTHONPATH']+=':'+str(P/'validation/workflows/response_design')
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e2','env',f'CUDA_VISIBLE_DEVICES={gpu}','timeout','1800',py,'-u',str(P/'validation/workflows/response_design/e2_gamma_audit.py'),'--source',row['source'],'--orbitals',row['orbitals'],'--output',str(out)]
  record=dict(time=datetime.datetime.now().isoformat(),jobid=250285,gpu=gpu,command=cmd,environment=env,log=str(log),status='running');record_path=A/(name+'_launch.json');record_path.write_text(json.dumps(record,indent=2)+'\n');print('E2_START',name,'GPU',gpu,flush=True)
  with log.open('xb') as f:
   start=time.monotonic();proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=proc.pid;record_path.write_text(json.dumps(record,indent=2)+'\n');code=proc.wait()
  audit=out/'audit.json';accept=json.loads(audit.read_text())['acceptance'] if audit.exists() else dict(passed=False,reason='No completed audit');record.update(end=datetime.datetime.now().isoformat(),returncode=code,status='completed' if code==0 else 'failed',acceptance=accept,GPU_hours=(time.monotonic()-start)/3600);record_path.write_text(json.dumps(record,indent=2)+'\n');print('E2_END',name,'passed',accept['passed'],flush=True)
  with (P/'response_design_to_do_list.md').open('a') as f:f.write(f'\n- {record["end"]} E2子项 {name}: '+('PASS' if code==0 and accept['passed'] else 'FAIL')+f'；证据 {audit.relative_to(P)}，原始态未改。\n')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:run_lane(*x),[(1,pending[::2]),(3,pending[1::2])]))
manifest.update(end=datetime.datetime.now().isoformat(),status='completed_queue');(A/'queue_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
current=json.loads((A/'inventory.json').read_text())
for r in current['states']:
 f=A/r['label']/'audit.json'
 if f.exists():r.update(status='完成' if json.loads(f.read_text())['acceptance']['passed'] else '失败',passed=json.loads(f.read_text())['acceptance']['passed'],audit=str(f))
(A/'inventory.json').write_text(json.dumps(current,indent=2)+'\n')
print('E2 Gamma队列结束；校准态缺轨道仍未验，E2整体不标完成。',flush=True)
