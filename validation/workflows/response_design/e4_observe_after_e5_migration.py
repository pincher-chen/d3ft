"""Observe the unchanged E4 GPU child after moving E5 to GPU7.

No E5 CPU is resumed on GPU0. E4 physics, output and timeout remain unchanged.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from e1_three_parameter_queue import alive, read, dump


def main(root):
    cfg = read(root/'protocol.json')
    old = Path(cfg['E4_root'])
    record = read(old/'GPU_launch.json')
    pid = record['pid']
    while not (root/'handover_ready.json').exists(): time.sleep(1)
    peak = read(old/'status.json').get('peak_sampled_MiB', 0)
    while alive(pid):
        sample = None
        try:
            r = subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1',
                'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=45)
            vals = [v for l in r.stdout.splitlines() if len(v:=l.split(',')) == 3 and int(v[0]) == 0]
            if r.returncode == 0 and len(vals) == 1:
                sample = dict(memory_MiB=int(vals[0][1]), utilization_percent=int(vals[0][2]))
                peak = max(peak, sample['memory_MiB'])
        except Exception as exc:
            dump(root/'probe_retry.json',dict(error=str(exc)))
        backend = old/'design/backend_status.json'
        now = dt.datetime.now().astimezone()
        dump(old/'status.json',dict(time=now.isoformat(),stage='Gamma_joint_design_running',
            controller_pid=os.getpid(),jobid=250285,gpus=[0],pid=pid,
            GPU_hours=(now-dt.datetime.fromisoformat(record['time'])).total_seconds()/3600,
            peak_sampled_MiB=peak,backend=read(backend) if backend.exists() else None,
            GPU=sample,E5_migrated_to_GPU7=True,E4_GPU_child_unchanged=True))
        time.sleep(15)
    end = dt.datetime.now().astimezone()
    done = (old/'design/result.json').exists()
    result = read(old/'design/result.json') if done else None
    record.update(end=end.isoformat(),returncode=None,inherited_waitpid_exit_code_unavailable=True,
        GPU_hours=(end-dt.datetime.fromisoformat(record['time'])).total_seconds()/3600,
        status='completed' if done else 'failed',completion_evidence='design/result.json exists' if done else 'missing result',
        peak_sampled_MiB=peak)
    dump(root/'observed_GPU_completion.json',record)
    dump(old/'GPU_launch.json',record)
    dump(old/'status.json',dict(time=end.isoformat(),stage='complete' if done else 'failed',
        controller_pid=os.getpid(),jobid=250285,gpus=[0],result=result,
        error=None if done else 'Unchanged E4 child ended without design result',
        GPU_hours=record['GPU_hours'],E5_migrated_to_GPU7=True,endpoint_audit_pending=True))
    Q = Path(cfg['project'])/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    with Q.with_name('gpu_reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        owners=read(Q)
        if owners.get('0',{}).get('status_file') == str(old/'status.json'):
            owners.pop('0');dump(Q,owners)
    for path in [Path(cfg['project'])/'paper/response_design_EXPERIMENTS_20261014.md',Path(cfg['project'])/'response_design_to_do_list.md']:
        with path.open('a') as f:f.write('\n- '+end.isoformat()+' E4原GPU0轨迹自然结束：'+('完成' if done else '失败')+'；E5已迁GPU7，不恢复旧CPU，端点审计另验。\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
