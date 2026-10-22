"""Observe only existing AlP children; never submit/retry/accept parameter FD."""
import argparse,datetime as dt,time,fcntl
from pathlib import Path
from e1_three_parameter_queue import read,dump,alive
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();T=a.root;c=read(T/'protocol.json');P=Path(c['project']);Q=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json';children=c['active_children'];finished=[]
while children:
 for rec in children[:]:
  if alive(rec['pid']):continue
  children.remove(rec);end=dt.datetime.now().astimezone();rec.update(observed_end=end.isoformat(),returncode=None,waitpid_unavailable=True,GPU_hours_upper=len(rec['gpus'])*(end-dt.datetime.fromisoformat(rec['start']).astimezone()).total_seconds()/3600,no_new_submission=True,FD_pair_not_certified=True);finished.append(rec)
  dump(T/(rec['tag']+'_natural_end.json'),rec)
  with Q.with_name('gpu_reservations.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
   for g in rec['gpus']:
    if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
   dump(Q,owners)
 dump(T/'status.json',dict(time=dt.datetime.now().astimezone().isoformat(),stage='complete_limited_scope_draining_existing_children' if children else 'complete_limited_scope_all_children_ended',controller_pid=__import__('os').getpid(),active=[dict(pid=r['pid'],gpus=r['gpus']) for r in children],completed=len(finished),new_GPU_submissions=0))
 if children:time.sleep(10)
dump(T/'drain_complete.json',dict(completed=True,children=finished,scope_certificate=str(T/'acceptance.json'),no_retries=True))
