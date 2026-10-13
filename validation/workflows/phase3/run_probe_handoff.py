"""Run one recorded GPU probe between experiments in an existing sequence.

The identified sequence controller is paused, while its already-running worker
is left untouched. Once that worker exits and all requested GPUs are idle, the
probe runs through the existing allocation. The controller is resumed in a
``finally`` block. This helper never submits a job or queries the Slurm queue.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--controller',type=int,required=True)
    ap.add_argument('--worker',type=int,required=True)
    ap.add_argument('--controller-token',required=True)
    ap.add_argument('--status',type=Path,required=True)
    ap.add_argument('--expected-experiment',required=True)
    ap.add_argument('--command',type=Path,required=True,
                    help='JSON array containing the exact probe command')
    ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--jobid',required=True)
    ap.add_argument('--gpus',default='0,1,2,3')
    ap.add_argument('--timeout',type=int,default=1800)
    a=ap.parse_args()
    a.record.mkdir(parents=True,exist_ok=False)
    command=json.loads(a.command.read_text())
    requested=[int(x) for x in a.gpus.split(',')]

    def write_status(state,**fields):
        value=dict(state=state,time=datetime.datetime.now().isoformat(),**fields)
        tmp=a.record/'status.tmp';tmp.write_text(json.dumps(value,indent=2))
        tmp.replace(a.record/'status.json')
        print(json.dumps(value),flush=True)

    def remote(code):
        return subprocess.check_output([
            'srun','--jobid='+a.jobid,'--overlap','-N1','-n1',
            '--cpus-per-task=1','python','-c',code],text=True,
            stderr=subprocess.STDOUT,timeout=30)

    controller_cmd=Path(f'/proc/{a.controller}/cmdline')
    if not controller_cmd.exists() or a.controller_token.encode() not in controller_cmd.read_bytes():
        raise RuntimeError('Controller identity changed')
    active=json.loads(a.status.read_text())
    if active.get('state')!='running' or active.get('experiment')!=a.expected_experiment:
        raise RuntimeError('Unexpected active experiment; refusing handoff')
    (a.record/'command.json').write_text(json.dumps(command,indent=2))
    result={}
    try:
        os.kill(a.controller,signal.SIGSTOP)
        write_status('waiting_for_current_worker',controller=a.controller,worker=a.worker)
        while True:
            try:
                code=("import json,pathlib,subprocess; "
                      f"p=pathlib.Path('/proc/{a.worker}/cmdline'); "
                      "print(json.dumps(dict(cmdline=p.read_bytes().replace(bytes([0]),b' ').decode() if p.exists() else '',"
                      "gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',"
                      "'--format=csv,noheader,nounits'],text=True))))")
                data=json.loads(remote(code))
                if data['cmdline'] and b'worker.py' not in data['cmdline'].encode():
                    raise RuntimeError('Worker PID reused')
                memory={int(row.split(',')[0]):int(row.split(',')[1])
                        for row in data['gpu'].splitlines()}
                if not data['cmdline'] and all(memory.get(g,10**9)<=100 for g in requested):
                    break
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,
                    json.JSONDecodeError) as exc:
                write_status('query_retry',error=str(exc))
            time.sleep(30)
        write_status('running_probe')
        with (a.record/'probe.log').open('x') as log:
            try:
                done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=a.timeout)
                result={'returncode':done.returncode}
            except subprocess.TimeoutExpired:
                result={'timeout_seconds':a.timeout}
        (a.record/'result.json').write_text(json.dumps(result,indent=2))
    finally:
        if controller_cmd.exists() and a.controller_token.encode() in controller_cmd.read_bytes():
            os.kill(a.controller,signal.SIGCONT)
        write_status('controller_resumed',controller=a.controller,result=result)


if __name__=='__main__':main()
