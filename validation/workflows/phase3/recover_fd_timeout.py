"""Recover one recorded FD timeout without interrupting a live calculation.

Only an authoritative terminal exit 124 triggers a new tag and longer timeout.
Original raw outputs and failures remain immutable. Scientific failures stop.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timeout_only(state):
    rows = state.get('results', [])
    failures = [x['returncode'] for x in rows if x['returncode']]
    return bool(state.get('state') == 'sequence_exited' and failures and
        all(x == 124 for x in failures))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('launch', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--old-trial', type=Path, required=True)
    args = ap.parse_args()
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    def status(state, **kw):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(), **kw)
        tmp = args.record/'status.tmp'; tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(args.record/'status.json'); print(json.dumps(obj), flush=True)
    launch = read(args.launch); plan = read(launch['plan'])
    if sha(launch['plan']) != launch['plan_sha256']:
        raise ValueError('Original FD plan changed')
    status('waiting_for_original_terminal_result')
    while True:
        state_path = Path(launch['record'])/'status.json'
        state = read(state_path) if state_path.exists() else {}
        if state.get('state') == 'sequence_exited':
            break
        try:
            os.kill(launch['controller_pid'], 0)
        except ProcessLookupError:
            status('original_controller_missing_no_automatic_restart'); return
        time.sleep(15)
    if not timeout_only(state):
        success = len(state.get('results', [])) == len(plan['experiments']) and not any(
            x['returncode'] for x in state.get('results', []))
        status('original_succeeded_no_retry' if success
            else 'non_timeout_failure_requires_diagnosis', original=state); return
    retry = copy.deepcopy(plan)
    retry['protocol']['frozen_at'] = datetime.datetime.now().isoformat()
    retry['protocol']['original_plan_sha256'] = launch['plan_sha256']
    suffix = '_timeoutretry'
    for experiment in retry['experiments']:
        command = experiment['command']; old_tag = command[command.index('--tag')+1]
        experiment['name'] += suffix; command[command.index('--tag')+1] = old_tag+suffix
        durations = [i for i,x in enumerate(command) if x == '7200s']
        if len(durations) != 1:
            raise ValueError('Expected original 7200 second worker timeout')
        command[durations[0]] = '86400s'
        results_root = next(x.split('=',1)[1] for x in command if x.startswith('D3FT_PHASE3_RESULTS_ROOT='))
        previous = Path(results_root)/old_tag
        if (previous/'metadata.json').exists():
            command.extend(['--reuse-from', str(previous)])
        if (Path(results_root)/(old_tag+suffix)).exists():
            raise ValueError('Retry tag already exists')
    for pair in retry['protocol']['pairs']:
        for sign in ['minus', 'plus']:
            pair[sign] += suffix
    def put(path, value):
        with path.open('x') as f:
            json.dump(value, f, indent=2)
    retry_plan = args.record/'retry_plan.json'; put(retry_plan, retry)
    driver = Path(__file__).with_name('run_gpu_sequence.py')
    retry_record = args.record/'sequence'
    with (args.record/'sequence_controller.log').open('x') as log:
        worker = subprocess.Popen([sys.executable,'-u',str(driver),str(retry_plan),
            '--record',str(retry_record)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    retry_launch = dict(controller_pid=worker.pid, job=launch['job'], gpus=launch['gpus'],
        plan=str(retry_plan.resolve()), plan_sha256=sha(retry_plan), record=str(retry_record.resolve()),
        supersedes_record=launch['record'], time=datetime.datetime.now().isoformat(),
        recovery_scope='New tags only after confirmed timeout. Matching completed points reused by archived worker validation; no numerical settings or thresholds changed.')
    launch_path = args.record/'retry_launch.json'; put(launch_path, retry_launch)
    auditor = Path(__file__).with_name('audit_fd_sequence.py'); audit_record = args.record/'audit'
    with (args.record/'audit_controller.log').open('x') as log:
        audit = subprocess.Popen([sys.executable,'-u',str(auditor),str(launch_path),'--record',str(audit_record)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    retry_launch.update(audit_controller_pid=audit.pid, audit_record=str(audit_record.resolve()))
    launches_path = args.campaign/'additional_fd_launches.json'
    launches = read(launches_path); launches.append(retry_launch)
    tmp = launches_path.with_suffix('.tmp'); tmp.write_text(json.dumps(launches,indent=2)); tmp.replace(launches_path)
    status('running_timeout_retry', worker_pid=worker.pid, auditor_pid=audit.pid)
    while audit.poll() is None:
        time.sleep(15)
    certificate = audit_record/'acceptance.json'
    if not certificate.exists() or not read(certificate).get('passed'):
        status('retry_failed_or_scientific_gate_failed', audit_status=read(audit_record/'status.json')); return
    # The old conditional trial must have stopped on its failed FD dependency.
    old_status = args.old_trial/'execution/status.json'
    for _ in range(20):
        old_state = read(old_status)
        if old_state.get('state') == 'fd_dependency_failed':
            break
        if old_state.get('state') != 'waiting_for_registered_fd_controls':
            status('unexpected_old_trial_state_no_rewire', details=old_state); return
        time.sleep(15)
    else:
        status('old_trial_did_not_stop_no_rewire'); return
    registry_path = args.campaign/'fd_gate_registry.json'; registry = read(registry_path)
    original_certificate = str(Path(launch['record'].replace('_sequence','_audit'))/'acceptance.json')
    if registry['acceptance_files'].count(original_certificate) != 1:
        raise ValueError('Cannot identify exactly one original FD dependency')
    put(args.record/'original_fd_gate_registry.json', registry)
    registry['acceptance_files'] = [str(certificate.resolve()) if p==original_certificate else p
        for p in registry['acceptance_files']]
    tmp=registry_path.with_suffix('.tmp'); tmp.write_text(json.dumps(registry,indent=2)); tmp.replace(registry_path)
    trial_dir = args.campaign/'actual_trials/all_t00_Si_timeoutretry'
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_v3_trial.py')),
        str(args.campaign),'--fold','all','--output',str(trial_dir)],check=True)
    runner = Path(__file__).with_name('run_v3_trial.py'); trial = trial_dir/'trial.json'
    with (trial_dir/'controller.log').open('x') as log:
        process = subprocess.Popen([sys.executable,'-u',str(runner),str(trial),'--record',str(trial_dir/'execution')],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    put(trial_dir/'launch.json', dict(controller_pid=process.pid, trial=str(trial.resolve()),
        record=str((trial_dir/'execution').resolve()), time=datetime.datetime.now().isoformat(),
        script_sha256=sha(runner), trial_sha256=sha(trial),
        scope='Conditional nonlinear first trial after registered timeout recovery; still no accepted-fit claim.'))
    status('fd_recovery_passed_and_conditional_trial_rewired', new_trial=str(trial_dir), controller_pid=process.pid)


if __name__ == '__main__':
    main()
