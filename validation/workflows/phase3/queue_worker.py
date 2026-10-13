"""Queue one Phase 3 worker behind a known Slurm step, on explicit idle GPUs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--jobid',required=True)
dependency=ap.add_mutually_exclusive_group(required=True)
dependency.add_argument('--after-step')
dependency.add_argument('--after-queue',type=Path)
ap.add_argument('--gpus',required=True);ap.add_argument('--record',type=Path,required=True)
ap.add_argument('--resume',action='store_true',help='Resume a waiting queue after its previous process has been confirmed stopped')
ap.add_argument('worker_args',nargs=argparse.REMAINDER)
a=ap.parse_args()
gpu_ids=list(map(int,a.gpus.split(',')))
if (a.after_step and not a.after_step.startswith(a.jobid+'.')) or len(set(gpu_ids))!=len(gpu_ids):
    raise ValueError('Invalid step or GPU assignment')
arguments=a.worker_args[1:] if a.worker_args[:1]==['--'] else a.worker_args
if not arguments:raise ValueError('Missing worker arguments')
a.record.mkdir(parents=True,exist_ok=a.resume)
command=['srun','--jobid='+a.jobid,'--overlap','-N1','-n1','--cpus-per-task='+str(8*len(gpu_ids)),
    'env','CUDA_VISIBLE_DEVICES='+a.gpus,'JAX_PLATFORMS=cuda',
    'bash',str(ROOT/'validation/phase3_audit_20260908/run_env.sh'),sys.executable,'-u',
    str(Path(__file__).with_name('worker.py')),*arguments]
if a.resume:
    old=json.loads((a.record/'status.json').read_text())
    if old['state'] not in ['waiting_for_step','waiting_for_queue','waiting_for_idle_gpus','scheduler_query_retry']:
        raise ValueError('Only an unstarted waiting queue may be resumed')
    if json.loads((a.record/'command.json').read_text())!=command:
        raise ValueError('Resume command differs from the recorded command')
    tag=arguments[arguments.index('--tag')+1]
    if (ROOT/'validation/results/phase3_20260908'/tag).exists():
        raise ValueError('Worker output already exists; verify its live process separately')
else:
    (a.record/'command.json').write_text(json.dumps(command,indent=2))
last=None
def status(state,**fields):
    global last
    value=dict(state=state,time=datetime.datetime.now().isoformat(),**fields)
    temporary=a.record/'status.tmp';temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(a.record/'status.json')
    if state!=last:
        with (a.record/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    last=state

def query(command):
    try:
        return subprocess.check_output(command,text=True,stderr=subprocess.STDOUT,timeout=30)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        status('scheduler_query_retry',query=command,error=str(error))
        time.sleep(15)
        return None

while True:
    answer=query(['squeue','-h','-j',a.jobid,'-o','%T'])
    if answer is None:continue
    jobs=answer.strip()
    if jobs!='RUNNING':
        detail=query(['scontrol','show','job','-o',a.jobid])
        if detail is None:continue
        if 'JobState=RUNNING' not in detail:
            status('allocation_unavailable',job_state=detail);raise SystemExit(1)
    if a.after_step:
        answer=query(['squeue','-h','--steps','-j',a.jobid,'-o','%i'])
        if answer is None:continue
        steps=answer.split()
        if a.after_step in steps:
            status('waiting_for_step',step=a.after_step);time.sleep(30);continue
    else:
        previous=a.after_queue/'status.json'
        prior=json.loads(previous.read_text()) if previous.exists() else {}
        if prior.get('state')=='allocation_unavailable':
            status('allocation_unavailable');raise SystemExit(1)
        if prior.get('state')!='worker_exited':
            status('waiting_for_queue',dependency=str(a.after_queue));time.sleep(30);continue
    answer=query(['srun','--jobid='+a.jobid,'--overlap','-N1','-n1',
        '--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'])
    if answer is None:continue
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in answer.strip().splitlines()}
    if any(i not in memory for i in gpu_ids):raise ValueError('Requested GPU is absent')
    if any(memory[i]>100 for i in gpu_ids):
        status('waiting_for_idle_gpus',memory_MiB=memory);time.sleep(30);continue
    break
status('running')
with (a.record/'worker.log').open('x') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
status('worker_exited',returncode=result.returncode)
raise SystemExit(result.returncode)
