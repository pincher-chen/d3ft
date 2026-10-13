"""Rebind owned calculation threads within their existing Slurm job cpuset."""
import argparse
import datetime
import json
import os
from pathlib import Path

def cpu_set(text):
    result=set()
    for part in text.strip().split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));result.update(range(lo,hi+1))
        elif part:result.add(int(part))
    return result

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--jobid',required=True)
ap.add_argument('--bindings',required=True,help='JSON mapping root PID to Linux CPU-list string')
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
if a.out.exists():raise FileExistsError(a.out)
job=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope')/('job_'+a.jobid)
allowed=cpu_set((job/'cpuset.cpus.effective').read_text())
bindings={int(pid):cpu_set(cpus) for pid,cpus in json.loads(a.bindings).items()}
selected=set()
for cpus in bindings.values():
    if not cpus or not cpus<=allowed or selected&cpus:raise ValueError('CPU sets must be allocated and disjoint')
    selected|=cpus
records=[]
for pid,cpus in bindings.items():
    process=Path('/proc')/str(pid)
    if process.stat().st_uid!=os.getuid():raise PermissionError('Process belongs to another user')
    cgroup=(process/'cgroup').read_text().strip().split(':',2)[2]
    if '/job_'+a.jobid+'/' not in cgroup:raise ValueError('Process belongs to another job')
    task_allowed=cpu_set((Path('/sys/fs/cgroup')/cgroup.lstrip('/')/'cpuset.cpus.effective').read_text())
    if not cpus<=task_allowed:raise ValueError(f'PID {pid}: CPUs are outside the task cpuset; no affinities changed')
for root,cpus in bindings.items():
    process=Path('/proc')/str(root)
    if process.stat().st_uid!=os.getuid():raise PermissionError('Process belongs to another user')
    if '/job_'+a.jobid+'/' not in (process/'cgroup').read_text():raise ValueError('Process belongs to another job')
    # Bind the parent first so future children inherit the intended affinity.
    pending=[root];done=set()
    while pending:
        pid=pending.pop(0)
        if pid in done:continue
        done.add(pid);p=Path('/proc')/str(pid)
        if not p.exists():continue
        if p.stat().st_uid!=os.getuid() or '/job_'+a.jobid+'/' not in (p/'cgroup').read_text():
            raise ValueError('Unexpected child ownership or allocation')
        row=dict(pid=pid,root=root,command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),threads=[])
        for thread in p.joinpath('task').iterdir():
            tid=int(thread.name)
            try:
                before=sorted(os.sched_getaffinity(tid));os.sched_setaffinity(tid,cpus)
                after=sorted(os.sched_getaffinity(tid))
                if set(after)!=cpus:raise ValueError('Kernel did not apply the requested affinity')
                row['threads'].append(dict(tid=tid,before=before,after=after))
                pending.extend(map(int,(thread/'children').read_text().split()))
            except ProcessLookupError:continue
        records.append(row)
result=dict(time=datetime.datetime.now().isoformat(),job=a.jobid,allocated_linux_cpus=sorted(allowed),processes=records,
            scope='Scheduling affinity only; no numerical parameters or cgroup allocations changed')
with a.out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps([dict(pid=r['pid'],threads=len(r['threads']),cpus=sorted(bindings[r['root']])) for r in records]))
