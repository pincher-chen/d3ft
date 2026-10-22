"""Recover the paused E5 CPU queue if E4's detached supervisor dies.

Never signal a GPU process. E5 is resumed only after E4's child has exited and
GPU0 is actually idle. The normal E4 supervisor performs the same handback.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from e4_gamma_queue import read, dump


def alive(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError: return False


def main(root):
    cfg = read(root/'launch_protocol.json')
    controller = read(root/'controller_launch.json')['pid']
    while not (root/'E5_CPU_resumed.json').exists():
        if alive(controller):
            time.sleep(15)
            continue
        childfile = root/'GPU_launch.json'
        if childfile.exists() and alive(read(childfile)['pid']):
            time.sleep(15)
            continue
        probe = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
            '-N1', '-n1', '-c1', 'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
        rows = [r for line in probe.stdout.splitlines() if len(r:=line.split(',')) == 2 and int(r[0]) == 0]
        if probe.returncode or len(rows) != 1 or int(rows[0][1]) > 3:
            time.sleep(15)
            continue
        pid = cfg['E5_CPU_pid']
        if not alive(pid):
            dump(root/'E5_resume_guard_error.json', dict(error='E5 CPU PID no longer exists; no signal sent'))
            return
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'e5_single_pipeline.py' in command and cfg['E5_root'] in command
        # The actual GPU is idle and the original E4 supervisor has exited.
        # Restore its reservation before allowing E5 to create any new child.
        Q = Path(cfg['project'])/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
        reservations = read(Q)
        if reservations.get('0', {}).get('status_file') == str(root/'status.json'):
            reservations['0'] = dict(owner='E5 recovered after E4 CPU supervisor exit',
                status_file=str(Path(cfg['E5_root'])/'status.json'))
            dump(Q, reservations)
        os.kill(pid, signal.SIGCONT)
        dump(root/'E5_CPU_resumed.json', dict(pid=pid, signal='SIGCONT', recovered_by_guard=True,
            no_GPU_signal=True, E4_controller_exited=True, E4_child_exited=True, GPU0_idle=True))
        return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    main(ap.parse_args().root)
