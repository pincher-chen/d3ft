"""One-card Gamma joint design; CPU-only handover from the waiting E5 queue."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(p): return json.loads(Path(p).read_text())


def dump(p, v):
    p = Path(p)
    tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def main(root):
    os.close(os.open(root/'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    c = read(root/'launch_protocol.json')
    P = Path(c['project'])
    Q = P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    old = Path(c['E5_root'])
    oldpid = c['E5_CPU_pid']
    paused = False
    proc = None
    start = None
    max_memory = 0

    def now(): return dt.datetime.now().astimezone().isoformat()

    def status(stage, **kw):
        dump(root/'status.json', dict(time=now(), stage=stage, controller_pid=os.getpid(),
            jobid=250285, gpus=[0], max_updates=5, GPU_hours_cap=100, **kw))

    def note(s):
        for f in [P/'paper/response_design_EXPERIMENTS_20261014.md', P/'response_design_to_do_list.md']:
            with f.open('a') as out: out.write('\n- '+now()+' '+s+'\n')

    def probe():
        r = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
            '-N1', '-n1', '-c1', 'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
        if r.returncode: raise RuntimeError(r.stderr)
        rows = [v for l in r.stdout.splitlines() if len(v:=l.split(',')) == 4 and int(v[0]) == 0]
        if len(rows) != 1: raise RuntimeError('GPU0 probe missing/ambiguous')
        v = rows[0]
        return dict(time=now(), gpu=0, UUID=v[1].strip(), memory_MiB=int(v[2]), utilization_percent=int(v[3]))

    status('checking_Gamma_gates_and_GPU0')
    try:
        for f in read(root/'protocol.json')['required_acceptance_files']:
            assert read(f)['passed'], 'E3 derivative gate missing'
        assert read(old/'status.json')['stage'] == 'waiting_E6_independent_hydro_quartet'
        oldcmd = Path(f'/proc/{oldpid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'e5_single_pipeline.py' in oldcmd and str(old) in oldcmd
        assert not Path(f'/proc/{oldpid}/task/{oldpid}/children').read_text().strip(), 'E5 has an active child'
        assert probe()['memory_MiB'] <= 3, 'GPU0 is occupied'
        # Freeze only the waiting CPU scheduler. No srun or GPU process is signalled.
        os.kill(oldpid, signal.SIGSTOP)
        paused = True
        assert not Path(f'/proc/{oldpid}/task/{oldpid}/children').read_text().strip(), 'E5 child appeared; release CPU handover'
        gpu = probe()
        assert gpu['memory_MiB'] <= 3
        dump(root/'GPU0_before.json', gpu)
        dump(root/'E5_CPU_handover.json', dict(time=now(), pid=oldpid, scope='waiting CPU scheduler only',
            children=[], no_GPU_signal=True, original_status=read(old/'status.json'), auto_resume_on_exit=True))
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reservations = read(Q)
            assert reservations.get('0', {}).get('status_file') in [None, str(old/'status.json')]
            reservations['0'] = dict(owner='E4 Gamma MgSiN2 54-variable bounded joint design',
                status_file=str(root/'status.json'), controller_pid=os.getpid(), E5_resume_after=str(root/'status.json'))
            dump(Q, reservations)
        env = {**os.environ, **c['environment']}
        tmp = '/dev/shm/'+root.name
        env.update(PYTHONPATH=c['runtime']+':'+c['jrystal'], CUDA_VISIBLE_DEVICES='0',
            TMPDIR=tmp, PYTHONPYCACHEPREFIX=tmp+'/pycache', JAX_COMPILATION_CACHE_DIR=tmp+'/jax',
            D3FT_TRACE_STAGES='1', D3FT_RESPONSE_SINGLE_DEVICE='1')
        command = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1', '-c8',
            'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"', 'E4-Gamma', 'env', 'CUDA_VISIBLE_DEVICES=0',
            'timeout', str(c['timeout_seconds']), c['python'], '-u',
            str(Path(c['runtime'])/'validation/workflows/response_design/joint_design_dft.py'),
            '--protocol', str(root/'protocol.json'), '--output', str(root/'design')]
        log = (root/'design.log').open('xb')
        start = time.monotonic()
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rec = dict(time=now(), pid=proc.pid, jobid=250285, gpus=[0], command=command,
            environment=env, timeout_seconds=c['timeout_seconds'], status='running')
        dump(root/'GPU_launch.json', rec)
        note('E4 Gamma MgSiN2 54变量正式启动：250285 GPU0、新唯一目录'+str(root)+
             '，最多5更新/目标早停、原信赖域/力/能量/距离/谱门；≤100GPUh。只暂挂无子进程E5等待CPU，结束/失败自动恢复，E6继续。')
        while proc.poll() is None:
            sample = None
            try:
                sample = probe()
                max_memory = max(max_memory, sample['memory_MiB'])
                with (root/'GPU_samples.jsonl').open('a') as out: out.write(json.dumps(sample)+'\n')
            except Exception as exc:
                dump(root/'probe_retry.json', dict(time=now(), error=str(exc)))
            backend = root/'design/backend_status.json'
            status('Gamma_joint_design_running', pid=proc.pid, GPU_hours=(time.monotonic()-start)/3600,
                peak_sampled_MiB=max_memory, GPU=sample, backend=read(backend) if backend.exists() else None)
            time.sleep(15)
        log.close()
        hours = (time.monotonic()-start)/3600
        rec.update(end=now(), returncode=proc.returncode, GPU_hours=hours, peak_sampled_MiB=max_memory,
                   status='completed' if proc.returncode == 0 else 'failed')
        dump(root/'GPU_launch.json', rec)
        if proc.returncode: raise RuntimeError('Gamma design exited '+str(proc.returncode)+'; failure retained')
        result = read(root/'design/result.json')
        status('complete', result=result, GPU_hours=hours, peak_sampled_MiB=max_memory)
        note('E4 Gamma正式轨迹结束：接受更新'+str(result['completed_updates'])+'、目标达成='+str(result['target_reached'])+
             '；端点仍须独立事后审计，不将非零力结构称平衡；恢复E5等待/FD后续。')
    except Exception as exc:
        status('failed', error=repr(exc), GPU_hours=(time.monotonic()-start)/3600 if start else 0)
        note('E4 Gamma联合设计停止：'+repr(exc)+'；失败现场保留，原门不放宽。')
        raise
    finally:
        # Normal exceptions after launch cannot release GPU0 while its child runs.
        if proc is not None and proc.poll() is None:
            status('waiting_owned_child_natural_exit_before_E5_resume', pid=proc.pid)
            proc.wait()
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reservations = read(Q)
            if reservations.get('0', {}).get('status_file') == str(root/'status.json'):
                reservations['0'] = dict(owner='E5 resumed after E4 GPU0 handover', status_file=str(old/'status.json'))
                dump(Q, reservations)
        if paused:
            os.kill(oldpid, signal.SIGCONT)
            dump(root/'E5_CPU_resumed.json', dict(time=now(), pid=oldpid, signal='SIGCONT', no_GPU_signal=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    main(ap.parse_args().root)
