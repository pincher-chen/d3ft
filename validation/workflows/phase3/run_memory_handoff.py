"""One-off recorded diagnostic handoff; preserves the running worker.

Pause only the identified Python sequence controller, wait for its current
worker to exit, run the memory probes, then resume the controller in finally.
Uses the existing allocation through srun, never squeue.
"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

R=Path('validation/results/phase3_20260908')
record=R/'sequences/priority_memory_handoff'
record.mkdir(exist_ok=False)
controller=1884532
worker=1456047

def status(state,**kw):
    data=dict(state=state,time=datetime.datetime.now().isoformat(),**kw)
    tmp=record/'status.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(record/'status.json')
    print(json.dumps(data),flush=True)

def remote(code):
    return subprocess.check_output(['srun','--jobid=233252','--overlap','-N1','-n1','--cpus-per-task=1','python','-c',code],text=True,stderr=subprocess.STDOUT,timeout=30)

cmdline=Path(f'/proc/{controller}/cmdline').read_bytes().replace(b'\0',b' ').decode()
if 'run_gpu_sequence.py' not in cmdline or 'refinement_4gpu_0908_plan.json' not in cmdline:
    raise RuntimeError('Controller identity changed')
current=json.loads((R/'sequences/refinement_4gpu_0908/status.json').read_text())
if current.get('experiment')!='evaluate_diamond_g80_c100_h005' or current['state']!='running':
    raise RuntimeError('Unexpected active experiment; refusing handoff')
results=[]
try:
    os.kill(controller,signal.SIGSTOP)
    status('waiting_for_current_worker',controller=controller,worker=worker)
    while True:
        try:
            data=json.loads(remote(f"import json,pathlib,subprocess; p=pathlib.Path('/proc/{worker}/cmdline'); print(json.dumps(dict(cmdline=p.read_bytes().replace(bytes([0]),b' ').decode() if p.exists() else '',gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True))))"))
            if data['cmdline'] and 'worker.py' not in data['cmdline']:
                raise RuntimeError('Worker PID reused')
            memory=[int(row.split(',')[1]) for row in data['gpu'].splitlines()]
            if not data['cmdline'] and len(memory)==4 and max(memory)<=100:break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as e:
            status('query_retry',error=str(e))
        time.sleep(30)
    for source,suffix in [('Ge_memory_stage_probe_plan.json','original'),('Ge_memory_fused_density_probe_plan.json','fused')]:
        plan=json.loads((R/source).read_text());command=plan['experiments'][0]['command'][:]
        tag='sample_Ge_priority_memory_'+suffix
        command[command.index('--tag')+1]=tag
        (record/(suffix+'_command.json')).write_text(json.dumps(command,indent=2))
        status('running_probe',tag=tag)
        with (record/(suffix+'.log')).open('x') as log:
            try:
                out=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                result=dict(tag=tag,returncode=out.returncode)
            except subprocess.TimeoutExpired:
                result=dict(tag=tag,timeout_seconds=1800)
        results.append(result)
        (record/'results.json').write_text(json.dumps(results,indent=2))
        if 'timeout_seconds' in result:
            break  # Do not start another probe after an unverified timeout.
finally:
    os.kill(controller,signal.SIGCONT)
    status('controller_resumed',controller=controller,results=results)
