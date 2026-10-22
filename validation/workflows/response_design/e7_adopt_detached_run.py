"""Finish accounting for an unchanged GPU child after its CPU dispatcher retires.

No new GPU work is launched. A DONE marker plus a full native numerical gate
certifies scientific completion. The unavailable inherited waitpid exit code is
recorded explicitly; observed wall time is a conservative GPU-budget bound.
"""
from pathlib import Path
import argparse,datetime,json,time
import e7_ct_pilot as p
ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True);a=ap.parse_args()
row=p.load(a.record);pid=row['pid'];tag=row['tag'];work=p.E/'runs'/tag
while True:
    q=Path(f'/proc/{pid}/stat')
    if not q.exists() or q.read_text().split(') ',1)[1].split()[0]=='Z':break
    time.sleep(1)
end=datetime.datetime.now();start=datetime.datetime.fromisoformat(row['time'])
log=p.E/(tag+'.log');text=log.read_text(errors='replace')
done=(work/'summary.json').exists() and ('DONE '+tag) in text
row.update(end=end.isoformat(),status='completed' if done else 'failed',returncode=None,
           completion_evidence='worker DONE and summary' if done else 'GPU child ended without worker DONE/summary',
           inherited_waitpid_exit_code_unavailable=True,
           GPU_hours=max(0,(end-start).total_seconds())*len(row['gpus'])/3600,
           GPU_hours_scope='conservative observed end upper bound; one-second polling; includes CPU handover latency',
           no_next_task_submitted=True)
p.dump(a.record,row)
if done:
    try:
        p.dump(work/'equilibrium_jacobian.json',p.derive(work));gate=dict(passed=True,scope='native gates on unchanged legacy run, not chosen over requested new fine run')
    except Exception as exc:gate=dict(passed=False,error=str(exc))
    p.dump(work/'native_acceptance.json',gate)
p.note('E7旧运行'+tag+'自然结束，数据和保守机时已由CPU观察器登记；无GPU信号、无自动AlAs重试。')
