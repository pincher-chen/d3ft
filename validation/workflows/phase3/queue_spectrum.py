"""Run a saved-density spectrum on explicit CPU cores after a Slurm step exits.

The caller reserves these cores through this dependency; this is not a general
scheduler. Failed computations are preserved and never automatically restarted.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--jobid', required=True)
    ap.add_argument('--after-step', required=True)
    ap.add_argument('--cores', required=True)
    a = ap.parse_args()
    if not a.after_step.startswith(a.jobid + '.'):
        raise ValueError('Dependency belongs to another allocation')
    if not a.cores or any(c not in '0123456789,-' for c in a.cores):
        raise ValueError('Invalid CPU list')
    if a.out.exists() or not (a.source / 'summary.json').is_file():
        raise ValueError('Requires an existing complete source and a new output')
    a.record.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    command = ['srun', '--jobid=' + a.jobid, '--overlap', '-N1', '-n1',
               '--cpus-per-task=56', '--cpu-bind=none', 'taskset', '-c', a.cores,
               'env', 'CUDA_VISIBLE_DEVICES=', 'JAX_PLATFORMS=cpu', 'bash',
               str(root / 'validation/phase3_audit_20260908/run_env.sh'),
               sys.executable, '-u', str(Path(__file__).with_name('check_occupations.py')),
               str(a.source.resolve()), '--out', str(a.out.resolve())]
    (a.record / 'command.json').write_text(json.dumps(command, indent=2))
    def status(state, **fields):
        value = dict(state=state, time=datetime.datetime.now().isoformat(), **fields)
        tmp = a.record / 'status.tmp'
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(a.record / 'status.json')
        print(json.dumps(value), flush=True)
    status('waiting_for_step', step=a.after_step)
    while True:
        try:
            job = subprocess.check_output(['squeue', '-h', '-j', a.jobid, '-o', '%T'],
                                          text=True, stderr=subprocess.STDOUT, timeout=30).strip()
            steps = subprocess.check_output(['squeue', '-h', '--steps', '-j', a.jobid, '-o', '%i'],
                                            text=True, stderr=subprocess.STDOUT, timeout=30).split()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            status('scheduler_query_retry', error=str(error))
            time.sleep(15)
            continue
        if job != 'RUNNING':
            status('allocation_unavailable', job_state=job)
            return 1
        if a.after_step not in steps:
            break
        time.sleep(30)
    status('running')
    with (a.record / 'worker.log').open('x') as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    status('worker_exited', returncode=result.returncode)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
