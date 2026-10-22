"""Run recorded CPU baseline steps after immutable dependency preparation."""
import argparse
import datetime
import json
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
    (a.record/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def status(state,**kw):
        obj=dict(state=state,time=datetime.datetime.now().isoformat(),**kw)
        tmp=a.record/'status.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n')
        tmp.replace(a.record/'status.json')
        print(json.dumps(obj),flush=True)
    status('waiting_for_artifacts')
    deadline=time.monotonic()+3600
    marker=plan.get('artifact_marker','response_design_BASELINE_ARTIFACTS_READY')
    while marker not in Path(plan['artifact_log']).read_text():
        if time.monotonic()>deadline:
            status('failed',reason='artifact initialization exceeded one hour')
            return
        time.sleep(15)
    results=[]
    for ex in plan['experiments']:
        cmd=ex['command']
        if cmd[:3]!=['srun',f"--jobid={plan['jobid']}",'--overlap']:
            raise ValueError('baseline must use the recorded allocation')
        status('running',experiment=ex['name'])
        with (a.record/(ex['name']+'.log')).open('x') as log:
            done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=ex['name'],returncode=done.returncode))
        (a.record/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if done.returncode:
            status('failed',experiment=ex['name'],results=results)
            return
    status('completed',results=results)


if __name__=='__main__':
    main()
