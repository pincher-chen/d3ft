"""Update only the idle E7 CPU dispatcher; leave all GPU stages untouched."""
from pathlib import Path
import datetime,json,os,signal,subprocess,time
from e7_ct_pilot import E,P,load,dump,priority_pending

old = load(E/'controller_launch.json')['pid']
def children(pid):
    return [int(v) for q in (Path('/proc')/str(pid)/'task').glob('*/children')
            for v in q.read_text().split()]
while True:
    if not Path(f'/proc/{old}').exists():
        dump(E/'idle_handover_status.json',dict(stage='old_dispatcher_already_exited',time=datetime.datetime.now().isoformat()))
        break
    # A passed E3 gate blocks the old CPU dispatcher between GPU stages.
    # Never stop the dispatcher while it has an active srun child.
    # Independent GPU0/GPU6 controllers own their own timing records. They
    # need not finish before replacing this gate-only parent with no children.
    if priority_pending() is None or children(old):
        time.sleep(10)
        continue
    os.kill(old,signal.SIGSTOP)
    if children(old):
        os.kill(old,signal.SIGCONT)
        time.sleep(10)
        continue
    receipt=dict(time=datetime.datetime.now().isoformat(),old_CPU_dispatcher_pid=old,
        verified_children=[],GPU_processes_signalled=[],reason='Correct fine-control wall-time allowance and E4 priority between GPU stages')
    os.kill(old,signal.SIGTERM)
    os.kill(old,signal.SIGCONT)
    script=P/'validation/workflows/response_design/e7_ct_pilot.py'
    with (E/'controller_resumed.log').open('xb') as f:
        proc=subprocess.Popen(['python','-u',str(script),'--resume-idle-handover'],stdout=f,
                              stderr=subprocess.STDOUT,start_new_session=True)
    receipt['new_CPU_dispatcher_pid']=proc.pid
    dump(E/'idle_handover_status.json',receipt)
    dump(E/'controller_resumed_launch.json',dict(**receipt,command=['python','-u',str(script),'--resume-idle-handover']))
    print('E7 idle CPU dispatcher handed over; no GPU process signalled.',flush=True)
    break
