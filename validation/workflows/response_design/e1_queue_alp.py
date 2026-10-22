"""E1 standalone NLCC AlP K4 then K10 lattice/response, guarded by E0."""
from pathlib import Path
import json,os,datetime,subprocess,time
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261014_e0_migration';A=P/'validation/results/response_design_20261014_e1_alp';protocol=json.loads((A/'protocol.json').read_text());gate=json.loads(Path(protocol['formal_production_gate']).read_text())
if not gate['passed']:raise RuntimeError('E0 physical gates not all passed')
fd=os.open(A/'queue.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
base=json.loads((E/'commands/alp_new_qe_tail.json').read_text());env=base['environment'].copy();R=Path(protocol['runtime']);env['PYTHONPATH']=env['PYTHONPATH'].replace(str(E/'runtime_tail'),str(R));env['D3FT_PHASE3_RESULTS_ROOT']=str(A/'runs');(A/'runs').mkdir(exist_ok=True);spent=0.;a0=5.48;previous=None
for k in protocol['k_queue']:
 tag=f'AlP_NLCC_k{k}';tmp=f'/dev/shm/response_design_e1_{tag}';env.update(TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',PYTHONPYCACHEPREFIX=tmp+'/pycache');cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=16','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e1','env','CUDA_VISIBLE_DEVICES=2,3,4,6','timeout','14400','/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python','-u',str(R/'validation/workflows/phase3/worker.py'),'--material','AlP','--parameters','base','--tag',tag,'--mode','evaluate','--grid','64','--cutoff','50','--k',str(k),'--epochs','3000','--empty','0','--strain-step','0.005','--a0',str(a0),'--kpool','--fused-report']
 if previous:cmd+=['--fixed-mask',str(previous/'mask.npy')]
 log=A/(tag+'.log');rec=dict(start=datetime.datetime.now().isoformat(),jobid=250285,gpus=protocol['gpus'],command=cmd,environment=env,log=str(log),status='running');rp=A/(tag+'_launch.json');rp.write_text(json.dumps(rec,indent=2)+'\n');print('E1_START',tag,'job250285 GPUs2,3,4,6',flush=True)
 with log.open('xb') as f:
  start=time.monotonic();proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;rp.write_text(json.dumps(rec,indent=2)+'\n');code=proc.wait()
 used=4*(time.monotonic()-start)/3600;spent+=used;folder=A/'runs'/tag;summary=folder/'summary.json';rec.update(end=datetime.datetime.now().isoformat(),returncode=code,GPU_hours=used,total_GPU_hours=spent,status='completed_stage' if code==0 and summary.exists() else 'failed');rp.write_text(json.dumps(rec,indent=2)+'\n')
 if code!=0 or not summary.exists():print('E1_FAIL',tag,'records retained; no next stage',flush=True);break
 report=json.loads(summary.read_text());a0=report['a0_A'];previous=folder
 print('E1_STAGE_COMPLETE',tag,'a0',a0,'GPUh',used,flush=True)
 if spent+16>protocol['hard_budget_GPU_hours']:print('E1_BUDGET_STOP',flush=True);break
else:
 (A/'status.json').write_text(json.dumps(dict(stage='K4_K10_computations_complete',time=datetime.datetime.now().isoformat(),total_GPU_hours=spent,passed=False,scope='Physical/numerical response-row acceptance and matching QE still require collection; do not mark E1 complete just from exit0'),indent=2)+'\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write(f'\n- {datetime.datetime.now().isoformat()} E1 AlP NLCC K4/K10计算已结束，待新基线与响应行正式验收；旧SG15行未替换。\n')
