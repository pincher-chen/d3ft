"""Execute a recorded serial experiment plan in an existing GPU allocation.

Checks devices through srun; never queries squeue or submits a new allocation.
Independent experiments continue after a failed worker, preserving exit codes.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan',type=Path)
    ap.add_argument('--record',type=Path,required=True)
    a=ap.parse_args()
    plan=json.loads(a.plan.read_text())
    a.record.mkdir(parents=True,exist_ok=False)
    (a.record/'plan.json').write_text(json.dumps(plan,indent=2))
    def status(state,**fields):
        d=dict(state=state,time=datetime.datetime.now().isoformat(),**fields)
        p=a.record/'status.tmp';p.write_text(json.dumps(d,indent=2));p.replace(a.record/'status.json')
        print(json.dumps(d),flush=True)
    if plan.get('after_record'):
        status('waiting_for_sequence',dependency=plan['after_record'])
        while True:
            previous=Path(plan['after_record'])/'status.json'
            if previous.exists():
                prior = json.loads(previous.read_text())
                if prior.get('state') == 'sequence_exited':
                    rows = prior.get('results', [])
                    if not rows or prior.get('stopped_after_failure') or any(row.get('returncode') != 0 for row in rows):
                        status('dependency_failed', dependency=plan['after_record'], results=rows)
                        raise RuntimeError('Preceding sequence did not complete successfully')
                    break
            try:os.kill(plan['after_pid'],0)
            except ProcessLookupError:
                status('dependency_controller_missing')
                raise RuntimeError('Preceding controller disappeared without a terminal record')
            time.sleep(30)
    results=[]
    for experiment in plan['experiments']:
        command=list(experiment['command']);name=experiment['name']
        # A nested step must not inherit the controller step's CPU binding.
        if not any(x.startswith('--cpu-bind') for x in command):
            command.insert(3, '--cpu-bind=none')
        if command[:3]!=['srun','--jobid='+plan['jobid'],'--overlap']:
            raise ValueError('Command does not use the intended existing allocation')
        while True:
            try:
                output=subprocess.check_output(['srun','--jobid='+plan['jobid'],'--overlap',
                    '--cpu-bind=none','-N1','-n1','--cpus-per-task=1','nvidia-smi',
                    '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                    text=True,stderr=subprocess.STDOUT,timeout=30)
                memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in output.strip().splitlines()}
                if all(g in memory and memory[g]<=100 for g in plan['gpus']):break
                status('waiting_for_idle_gpus',experiment=name,memory_MiB=memory)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
                status('node_query_retry',experiment=name,error=str(e),output=getattr(e,'output',None))
            time.sleep(30)
        status('running',experiment=name)
        with (a.record/(name+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=name,returncode=result.returncode))
        (a.record/'results.json').write_text(json.dumps(results,indent=2))
        status('experiment_exited',experiment=name,returncode=result.returncode)
        if result.returncode and plan.get('stop_on_failure',False):
            status('sequence_exited',results=results,stopped_after_failure=name)
            return
    status('sequence_exited',results=results)


if __name__=='__main__':main()
