"""Complete the remaining diamond fine points with unchanged physical gates."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import e7_ct_pilot as pilot
from run_identity import reserve_name


def main(T):
    os.close(os.open(T/'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    cfg = pilot.load(T/'protocol.json')
    Y = pilot.E
    Q = Y/'gpu_reservations.json'
    gpus = cfg.get('gpus', [2, 5])
    child = None
    record = None
    started = None

    def state(stage, **kw):
        pilot.dump(T/'status.json', dict(time=pilot.now(), stage=stage, jobid=250285,
            requested_gpus=gpus, controller_pid=os.getpid(), timeout_seconds=43200,
            remaining_points=cfg['missing_points'], **kw))

    def claim():
        r = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
            '-N1', '-n1', '-c1', 'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
        if r.returncode: raise RuntimeError(r.stderr)
        rows = {int(v[0]):int(v[1]) for l in r.stdout.splitlines() if len(v:=l.split(',')) == 2}
        pilot.dump(T/'GPU_probe.json', dict(time=pilot.now(), memory_MiB=rows))
        if any(rows.get(g, 10**9) > 3 for g in gpus): return False
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = pilot.load(Q)
            if any(owners.get(str(g), {}).get('status_file') not in [None, str(T/'status.json')] for g in gpus):
                return False
            for g in gpus:
                owners[str(g)] = dict(owner='E7 diamond remaining fine POINTs, 12h',
                    status_file=str(T/'status.json'), controller_pid=os.getpid())
            pilot.dump(Q, owners)
        return True

    try:
        state('waiting_requested_cards_natural_release')
        while not (T/'resource_handover_ready.json').exists(): time.sleep(2)
        while not claim():
            state('waiting_requested_cards_natural_release')
            time.sleep(20)
        if pilot.spent()+len(gpus)*12+1/60 > pilot.PROTO['pilot_GPU_hours_cap']:
            raise RuntimeError('E7 pilot 120 GPUh cap cannot reserve 12h x2; no child launched')
        tag = reserve_name(Y, 'diamond_fine_remaining12h', output_parent=Y/'runs', log_parent=Y)
        tmp = '/dev/shm/'+tag
        env = {**os.environ, **cfg['environment']}
        env.update(PYTHONPATH=cfg['runtime']+':'+cfg['jrystal'], TMPDIR=tmp,
            PYTHONPYCACHEPREFIX=tmp+'/pycache', JAX_COMPILATION_CACHE_DIR=tmp+'/jax',
            D3FT_PHASE3_RESULTS_ROOT=str(Y/'runs'), D3FT_TRACE_STAGES='1')
        # Keep the original k_pool=True and all other numerical arguments;
        # the worker enforces strict metadata and per-POINT geometry matches.
        command = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1', '-c8',
            'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"', 'E7-diamond-remainder', 'env',
            'CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)), 'timeout', '43200', pilot.PY, '-u',
            str(Path(cfg['runtime'])/'validation/workflows/phase3/worker.py'),
            '--material', 'diamond', '--vector', '[0,0,0,0,0,0,0]', '--tag', tag,
            '--mode', 'evaluate', '--bulk-only', '--grid', '80', '--cutoff', '200',
            '--k', '12', '--epochs', '3000', '--empty', '0', '--strain-step', '0.0025',
            '--xc-grid-factor', '1', '--a0', str(cfg['original_a0_A']), '--fused-report',
            '--kpool', '--reuse-from', cfg['reuse_from']]
        log = (Y/(tag+'.log')).open('xb')
        started = time.monotonic()
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(time=pilot.now(), tag=tag, material='diamond', kind='fine',
            jobid=250285, gpus=gpus, pid=child.pid, command=command, environment=env,
            output=str(Y/'runs'/tag), status='running', timeout_seconds=43200,
            independent_new_process=True, inherited_PID=None, reused_completed_points=cfg['reused_points'],
            reuse_from=cfg['reuse_from'], failed_predecessor_retained=True)
        pilot.dump(Y/(tag+'_launch.json'), record)
        pilot.dump(T/'launch_index.json', dict(record=str(Y/(tag+'_launch.json')), tag=tag))
        pilot.note('E7 diamond剩余两POINT新独立进程启动 '+tag+'，250285/GPU'+','.join(map(str,gpus))+'、timeout12h；复用3个合格POINT，原124失败保留、设置和门不变。')
        while child.poll() is None:
            state('diamond_remaining_POINTs_running', pid=child.pid, tag=tag,
                  elapsed_seconds=time.monotonic()-started)
            time.sleep(15)
        log.close()
        record.update(end=pilot.now(), returncode=child.returncode,
            GPU_hours=len(gpus)*(time.monotonic()-started)/3600,
            status='completed' if child.returncode == 0 else 'failed')
        pilot.dump(Y/(tag+'_launch.json'), record)
        if child.returncode: raise RuntimeError('Diamond remainder exited '+str(child.returncode)+'; failed record retained')
        work = Y/'runs'/tag
        if ('DONE '+tag) not in (Y/(tag+'.log')).read_text(): raise ValueError('Missing worker DONE')
        pilot.dump(work/'equilibrium_jacobian.json', pilot.derive(work))
        pilot.dump(work/'native_acceptance.json', dict(passed=True,
            scope='Original native equilibrium/iso gates, reused POINTs with exact config/PP/code/geometry',
            independent_fine_certificate=False))
        choices = pilot.load(Y/'selected_runs.json')
        choices.setdefault('diamond', {})['fine'] = tag
        pilot.dump(Y/'selected_runs.json', choices)
        result = pilot.compare('diamond')
        if result is None: raise RuntimeError('Diamond independent comparison incomplete')
        pilot.dump(T/'acceptance.json', result)
        pilot.report()
        state('complete', passed=result['passed'], result=result)
        pilot.note('E7 diamond重提独立加密 '+('PASS' if result['passed'] else 'FAIL')+'；三原POINT复用、两缺点新算，124与旧结果保留，不放宽a0/B门。')
        print('E7 diamond独立加密 '+('PASS' if result['passed'] else 'FAIL')+'。\n已复用3POINT并补齐剩余2POINT。\n原失败和数值门保留，未冒称训练完成。', flush=True)
    except Exception as exc:
        state('failed', passed=False, error=repr(exc))
        pilot.note('E7 diamond剩余点队列停止：'+repr(exc)+'；现场保留，不放宽门。')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.wait()  # original bounded child; never send GPU signals
            if record is not None:
                record.update(end=pilot.now(), returncode=child.returncode,
                    GPU_hours=len(gpus)*(time.monotonic()-started)/3600, status='failed')
                pilot.dump(Y/(record['tag']+'_launch.json'), record)
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = pilot.load(Q)
            for g in gpus:
                if owners.get(str(g), {}).get('status_file') == str(T/'status.json'):
                    owners.pop(str(g))
            pilot.dump(Q, owners)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    main(ap.parse_args().root)
