"""E10-only recovery: one-point smoke before detached eight-GPU path queue."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import dump, now, read, sha, unique


def main(root):
    root = root.resolve()
    protocol = read(root / 'protocol.json')
    project = Path(protocol['project'])
    workflow = project / 'validation/workflows/response_design/e9_e10'
    recovery = Path(read(root / 'path_recovery_current.json')['root'])
    # Do not allow two recovery controllers to consume the same smoke/queue.
    lock = os.open(recovery / 'resume_started.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock, str(os.getpid()).encode())
    os.close(lock)

    def note(text):
        for path in [project / 'response_design_to_do_list.md', project / 'paper/response_design_EXPERIMENTS_20261014.md']:
            with path.open('a') as stream:
                stream.write('\n- ' + now() + ' ' + text + '\n')

    run = unique(recovery, 'single_k0_smoke')
    output = run / 'output'
    runtime = Path(protocol['E5_protocol']['runtime'])
    tmp = '/dev/shm/' + run.name
    env = {**os.environ, **protocol['E5_protocol']['environment']}
    env.update(
        PYTHONPATH=':'.join(map(str, [workflow, runtime,
            runtime / 'validation/workflows/response_design/reverse_response',
            runtime / 'validation/workflows/response_design', runtime / 'validation/workflows/phase3',
            Path(protocol['jrystal'])])),
        TMPDIR=tmp, PYTHONPYCACHEPREFIX=tmp + '/pycache',
        JAX_COMPILATION_CACHE_DIR=tmp + '/jax', PYTHONUNBUFFERED='1')
    source = read(root / 'SCF_states.json')['E4_initial']
    command = [
        'srun', '--jobid=250285', '--overlap', '--nodelist=hn35',
        '--cpu-bind=none', '--gpu-bind=none', '-N1', '-n1', '-c4',
        'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"', 'E10-smoke',
        'env', 'CUDA_VISIBLE_DEVICES=0', 'timeout', '--signal=TERM', '--kill-after=30s',
        '1800', protocol['python'], '-u', str(workflow / 'e10_path.py'),
        '--source', source, '--protocol', str(root / 'protocol.json'),
        '--indices', '[0]', '--output', str(output)]
    started = time.monotonic()
    with (run / 'stdout.log').open('xb') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(time=now(), pid=process.pid, node='hn35', jobid=250285, gpus=[0],
                      command=command, output=str(output), source=source, timeout_seconds=1800,
                      script_sha256=sha(workflow / 'e10_path.py'), classification='single-point smoke')
        dump(run / 'launch.json', record)
        note(f'E10只读种子修复后单k点冒烟已提交hn35/250285 GPU0，srunPID {process.pid}；两态SCF复用，E9暂停。通过原谱门后自动接8卡两态路径队列，未过则停止。')
        while process.poll() is None:
            dump(root / 'path_recovery_status.json', dict(
                time=now(), stage='single_k0_smoke_running', controller_pid=os.getpid(),
                srun_pid=process.pid, run=str(run), active_GPU_hours=(time.monotonic()-started)/3600))
            time.sleep(15)

    record.update(end=now(), returncode=process.returncode,
                  GPU_hours=(time.monotonic()-started)/3600)
    dump(run / 'launch.json', record)
    accepted = False
    detail = {}
    if process.returncode == 0:
        summary = read(output / 'summary.json')
        point = read(output / 'k_0000.json')
        seed = read(output / 'seed_preflight.json')
        detail = dict(maximum_ritz_Ha=max(point['residuals_Ha']),
                      orthogonality_max=point['orthogonality_max'], charge=seed['charge'],
                      source_state_sha256=point['source_state_sha256'])
        accepted = bool(summary['passed'] and point['passed'] and len(point['eigenvalues_Ha']) == 56
                        and max(point['residuals_Ha']) <= 1e-7
                        and point['orthogonality_max'] <= 1e-8
                        and abs(seed['charge'] - 96.) <= 1e-6)
    certificate = dict(time=now(), passed=accepted, state='E4_initial', index=0,
                       output=str(output), GPU_hours=record['GPU_hours'],
                       returncode=process.returncode, **detail)
    dump(recovery / 'smoke_acceptance.json', certificate)
    if not accepted:
        dump(root / 'path_recovery_status.json', dict(
            time=now(), stage='smoke_failure_retained_no_full_queue', smoke=certificate))
        note(f'E10单点冒烟未通过，returncode={process.returncode}；保留日志、停止扩展提交，不放宽数值标准。')
        raise RuntimeError('E10 smoke did not pass; full path queue not submitted')

    dump(root / 'path_smoke_acceptance.json', certificate)
    note(f'E10单k点冒烟通过原56带谱验收（Ritz≤1e-7Ha、正交≤1e-8、电荷96），立即自动启动8卡起点+终点路径；冒烟k0复用不重算。')
    dump(root / 'path_recovery_status.json', dict(
        time=now(), stage='smoke_passed_starting_full_path', controller_pid=os.getpid(),
        smoke=certificate))
    subprocess.run([sys.executable, str(workflow / 'followup.py'), '--root', str(root)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
