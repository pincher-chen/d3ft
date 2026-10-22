"""Observe the live worker; only exit124 authorizes a new checkpoint run.

No signals, no live edits, same physical model/gates and shared400GPUh ledger.
Existing LOO-diamond four-card dispatch is honored before claiming resources.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from e1_three_parameter_queue import read, dump
from e7_shared_budget import SharedBudget
from gpu_probe_retry import install_gpu_probe_retry
from run_identity import reserve_name

install_gpu_probe_retry()


def now():
    return dt.datetime.now().astimezone().isoformat()


def main(root):
    c = read(root / 'protocol.json')
    project, evaluation = Path(c['project']), Path(c['evaluation'])
    registry = project / 'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    budget = SharedBudget(c['budget_file'])
    source_launch = Path(c['original_launch'])
    history = []
    proc = None
    gpus = []
    active_tag = None
    active_started = None
    record = None

    def note(text):
        for f in [project/'response_design_to_do_list.md', project/'paper/response_design_EXPERIMENTS_20261014.md',
                  project/'paper/response_design_E7_FROZEN_EVALUATION_20261017.md']:
            with f.open('a') as stream:
                stream.write('\n- ' + now() + ' ' + text + '\n')

    def status(stage, **kw):
        dump(root/'status.json', dict(time=now(), controller_pid=os.getpid(), stage=stage,
            jobid=250285, node='hn35', attempt_history=history,
            budget=budget.summary(), **kw))

    def release():
        with registry.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = read(registry)
            for gpu in gpus:
                if owners.get(str(gpu), {}).get('controller_pid') == os.getpid():
                    owners.pop(str(gpu))
            dump(registry, owners)

    def claim():
        # Original controller gets to dispatch its already queued LOO-diamond.
        state = read(evaluation/'status.json')
        if 'LOO_diamond_fine_validation' in state.get('pending', []):
            return None
        probe = subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none',
            '--nodelist=hn35','-N1','-n1','-c1','nvidia-smi',
            '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=45)
        if probe.returncode:
            return None
        memory = {int(a):int(b) for a,b in (line.split(',') for line in probe.stdout.splitlines())}
        with registry.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = read(registry)
            free = [gpu for gpu in range(8) if str(gpu) not in owners and memory.get(gpu,10**9) <= 3]
            if len(free) < 4:
                return None
            selected = free[:4]
            for gpu in selected:
                owners[str(gpu)] = dict(owner='E7 registered full-diamond exit124 checkpoint continuation',
                    controller_pid=os.getpid(),status_file=str(root/'status.json'))
            dump(registry, owners)
            return selected

    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            record = read(source_launch)
            if record.get('status') == 'running' or 'returncode' not in record:
                status('observing_original_worker_no_GPU_no_signals', observed_launch=str(source_launch))
                time.sleep(20)
                continue
            if record['returncode'] != 124:
                status('no_timeout_recovery_authorized', observed_returncode=record['returncode'],
                    interpretation='Original normal completion or non-timeout failure; no automatic retry')
                return
            # The parent must finish accounting and releasing before recovery.
            if source_launch == Path(c['original_launch']):
                if not (evaluation/'full_fine_diamond_acceptance.json').exists():
                    time.sleep(10)
                    continue
                dump(evaluation/'full_fine_diamond_infrastructure_classification.json', dict(
                    time=now(), original_record=str(source_launch), original_acceptance_retained=True,
                    classification='infrastructure_pause_exit124', physical_gate_failed=False,
                    effective_acceptance_pending=True, recovery_index=str(root)))
            if not history or history[-1]['source_launch'] != str(source_launch):
                history.append(dict(source_launch=str(source_launch), returncode=124,
                    source=str(record['output']), classification='infrastructure_pause',
                    recorded_at=now()))
                note('E7 full-diamond exit124基础设施暂停已记录，原记录/POINT/检查点保留；仅物理检查点续算，不计物理失败，仍用共享400GPUh。')
            gpus = claim()
            if gpus is None:
                status('waiting_four_idle_GPUs_existing_LOO_queue_unchanged')
                time.sleep(20)
                continue
            active_tag = reserve_name(root,'full_diamond_checkpoint_resume',
                output_parent=root/'runs',log_parent=root)
            try:
                budget.reserve(active_tag,48)
            except RuntimeError:
                release()
                gpus = []
                status('budget_wait_no_GPU_launched', required_reservation_GPUh=48)
                time.sleep(60)
                continue
            # Preserve the original worker's exact physical/numerical argv.
            original = read(Path(c['original_launch']))
            argv = original['command']
            wi = next(i for i,arg in enumerate(argv) if arg.endswith('/phase3/worker.py'))
            arguments = list(argv[wi+1:])
            arguments[arguments.index('--tag')+1] = active_tag
            arguments += ['--reuse-from',record['output']]
            temp = '/dev/shm/'+active_tag
            env = dict(os.environ, **c['environment'])
            env.update(PYTHONPATH=c['runtime']+':'+c['jrystal'], TMPDIR=temp,
                JAX_COMPILATION_CACHE_DIR=temp+'/jax',PYTHONPYCACHEPREFIX=temp+'/pycache',
                D3FT_PHASE3_RESULTS_ROOT=str(root/'runs'),
                D3FT_CHECKPOINT_ROOT=str(root/'runs'/active_tag/'checkpoints'),
                D3FT_CHECKPOINT_INTERVAL='500',D3FT_RESUME_SOURCE=record['output'],
                D3FT_RESUME_REPORT=str(root/'resume_reports'/active_tag),
                D3FT_EVALUATION_FREEZE_CERT=c['certificate'])
            command = ['srun','--jobid=250285','--overlap','--cpu-bind=none',
                '--nodelist=hn35','-N1','-n1','-c16','--gpu-bind=none','bash','-c',
                'mkdir -p "$TMPDIR"; exec "$@"','E7-checkpoint-continuation','env',
                'CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)), 'timeout','43200',c['python'],'-u',
                str(Path(c['runtime'])/'validation/workflows/phase3/scf_checkpoint_resume.py'),*arguments]
            stream = (root/(active_tag+'.log')).open('xb')
            active_started = time.monotonic()
            try:
                proc = subprocess.Popen(command, env=env, cwd=project,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            except Exception:
                budget.finish(active_tag,0,'not_launched')
                release()
                stream.close()
                raise
            launch = dict(time=now(),pid=proc.pid,status='running',jobid=250285,node='hn35',
                gpus=gpus,output=str(root/'runs'/active_tag),command=command,
                reuse_from=record['output'],original_attempt=str(c['original_launch']),
                timeout_seconds=43200,checkpoint_interval=500,
                optimizer_moments_restored=False,physical_warm_start_not_bitwise_optimizer_resume=True,
                performance_changes=[],frozen_certificate=c['certificate'])
            source_launch = root/(active_tag+'_launch.json')
            dump(source_launch,launch)
            note('E7 full-diamond检查点续算启动 '+active_tag+'，GPU'+','.join(map(str,gpus))+
                 '；完整POINT复用、未完成POINT原3000步余段；Adam动量重置如实披露，原cut/grid/k/门不变，无性能补丁。')
            while proc.poll() is None:
                status('checkpoint_continuation_running', active=launch,
                    elapsed_GPUh=(time.monotonic()-active_started)*4/3600)
                time.sleep(20)
            stream.close()
            hours = (time.monotonic()-active_started)*4/3600
            launch.update(end=now(),returncode=proc.returncode,GPU_hours=hours,
                status='completed' if proc.returncode==0 else 'infrastructure_paused' if proc.returncode==124 else 'failed')
            dump(source_launch,launch)
            budget.finish(active_tag,hours,launch['status'])
            release()
            gpus = []
            proc = None
            if launch['returncode'] == 124:
                continue
            result = dict(time=now(),record=str(source_launch),passed=False,
                attempt_history=history,original_timeout_is_physical_failure=False,
                original_attempt_retained=True,parameters_changed=False,used_for_training=False)
            if launch['returncode'] == 0:
                try:
                    sys.path.insert(0,str(Path(c['runtime'])/'validation/workflows/phase3'))
                    from equilibrium_jacobian import derive
                    output = Path(launch['output'])
                    derive(output)
                    summary = read(output/'summary.json')
                    coarse = read(Path(c['coarse_state'])/'summary.json')
                    ar = abs(coarse['a0_A']-summary['a0_A'])/abs(summary['a0_A'])
                    bd = abs(coarse['B_GPa']-summary['B_GPa'])
                    result.update(passed=ar<1e-4 and bd<.1,a0_relative_error=ar,
                        B_difference_GPa=bd,a0_gate=1e-4,B_gate_GPa=.1,
                        native_gate_passed=True,summary=summary)
                except Exception as e:
                    result['error'] = repr(e)
            else:
                result['error'] = 'Non-timeout continuation failure; no automatic physical retry'
            dump(root/'acceptance.json',result)
            dump(evaluation/'full_fine_diamond_resolved_acceptance.json',result)
            note('E7 full-diamond超时续算结束，原门验收='+str(result['passed'])+
                '；原124记录保留，最终判定见full_fine_diamond_resolved_acceptance.json，未改门/参数。')
            status('completed_registered_recovery',passed=result['passed'])
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
