"""Bounded AlP FD queue for the three E7 parameters; no GPU cancellation.

The old theta_s child is observed to natural completion, never reused as a
different parameter. New children are independent processes with unique names.
Only idle, explicitly released pairs can be acquired. Deadline and concurrent
timeout reservations enforce the preregistered wall-clock and GPU-hour caps.
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

AXES = (5, 6, 1)  # kappa, mu, theta_p in the frozen seven-column order.
OMITTED_AXES = (0, 2, 3, 4)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def allowed_seconds(deadline, spent, active_reservations, cap, now=None, maximum_seconds=43200):
    now = now or dt.datetime.now().astimezone()
    remaining = min((deadline-now).total_seconds()-60,
                    (cap-spent-active_reservations)*3600/2-60)
    return max(0, min(maximum_seconds, int(remaining)))


def matching_vector(metadata, axis, sign, steps):
    target = [0.] * 7
    target[axis] = sign * steps[axis]
    return metadata.get('parameter_vector') == target


def observed_completion_code(tag, summary_exists, logtext):
    # A detached observer has no waitpid status. Classify scientific completion
    # from files; classify timeout only from the explicit srun error, not time.
    if summary_exists and ('DONE '+tag) in logtext:
        return 0
    return 124 if 'exit code 124' in logtext.lower() else 1


def main(root):
    import numpy as np
    import e1_extend_kmesh as e1
    from fd_acceptance import component
    from run_identity import reserve_name

    os.close(os.open(root/'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    cfg = read(root/'protocol.json')
    P, E = e1.P, e1.E
    X = E/'extension_20261015'
    Y = P/'validation/results/response_design_20261015_e7_ct_pilot'
    reservations = Y/'gpu_reservations.json'
    deadline = dt.datetime.fromisoformat(cfg['deadline'])
    fd = read(X/'formal_validation/FD_protocol.json')
    row = read(X/'response_row_k12.json')
    base = read(Path(row['source'])/'summary.json')
    assert read(X/'formal_validation/QE_acceptance.json')['passed']
    assert [fd['parameter_order'][a] for a in AXES] == ['x:kappa', 'x:mu', 'theta_p']
    pending = [tuple(x) for x in cfg.get('pending_tasks', [(a, s) for a in AXES for s in (-1, 1)])]
    active = {}
    completed = cfg.get('completed_points', {})
    certificates = cfg.get('pair_certificates', {})
    retries = cfg.get('same_parameter_retry_sources', {})
    spent = cfg['prior_GPU_hours_upper_bound']
    terminal = None
    legacy_recorded = False
    (root/'runs').mkdir(exist_ok=True)
    legacy = read(cfg['legacy_launch'])
    diamond_pid = cfg['diamond_controller_pid']
    groups = tuple(tuple(g) for g in cfg.get('GPU_groups', ((2, 3), (1, 4), (5, 6))))
    for rec in cfg.get('observed_active_GPU_children', []):
        # Observe existing independent children through a CPU-only handover;
        # their original timeout cannot be changed without restarting them.
        gpus = tuple(rec['gpus'])
        active[gpus] = dict(process=None, record=rec, log=None, observed=True,
            reservation=len(gpus)*rec['timeout_seconds']/3600)

    def now():
        return dt.datetime.now().astimezone().isoformat()

    def report(stage, **kwargs):
        dump(root/'status.json', dict(time=now(), stage=stage, jobid=250285,
             controller_pid=os.getpid(), deadline=cfg['deadline'],
             spent_GPU_hours_upper_bound=spent, active=[v['record'] for v in active.values()],
             pending=[list(x) for x in pending], completed_points=list(completed),
             passed_pairs=list(certificates), required_pairs=3, **kwargs))

    def policy_fallback(reason):
        # Never inspect held-out result files or silently call training complete.
        with (Y/'material_policy.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            policy = read(Y/'material_policy.json')
            policy.setdefault('excluded', {})['AlP'] = dict(reason=reason, time=now(),
                no_retry=True, retained_observables=['a0_A at K12'], old_failure_retained=True)
            policy['conditional_train'] = [m for m in policy.get('conditional_train', []) if m != 'AlP']
            policy.update(selected_training_materials=['Si', 'BN', 'diamond'],
                selected_LOO_folds=3, AlP_scope_certificate=str(root/'acceptance.json'),
                all_training_numerical_gates_still_required=True)
            dump(Y/'material_policy.json', policy)
        dump(root/'E7_fallback.json', dict(time=now(), reason=reason, materials=['Si', 'BN', 'diamond'],
             folds=3, heldouts_sealed=['SiC', 'GaP'], training_started=False))

    def release(gpus):
        with (Y/'gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            r = read(reservations)
            for g in gpus:
                if r.get(str(g), {}).get('status_file') == str(root/'status.json'):
                    r.pop(str(g))
            dump(reservations, r)

    def claim(gpus, memory):
        hold = cfg.get('E7_card_handover')
        if hold and set(gpus) & set(hold['gpus']):
            target = Path(hold['status_file'])
            if not target.exists() or read(target).get('stage') not in ('complete', 'failed'):
                return False
        if any(memory.get(g, 10**9) > 3 for g in gpus):
            return False
        if gpus == (2, 3) and alive(legacy['pid']):
            return False
        # The explicit E7_card_handover above withholds GPU2/5 only.
        # A live waiting diamond CPU must not block unrelated GPU1/4 or3/6.
        with (Y/'gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            r = read(reservations)
            allowed = {str(root/'status.json'), cfg['legacy_status_file'], cfg.get('previous_controller_status_file')}
            if gpus != (2, 3):
                allowed.add(cfg['diamond_status_file'])
            if any(r.get(str(g), {}).get('status_file') not in allowed | {None} for g in gpus):
                return False
            for g in gpus:
                r[str(g)] = dict(owner='E1 three-parameter FD only', status_file=str(root/'status.json'),
                    controller_pid=os.getpid())
            dump(reservations, r)
        return True

    def start(task, gpus, seconds):
        axis, sign = task
        h = fd['steps'][axis]
        vector = [0.] * 7
        vector[axis] = sign*h
        key = f'{axis},{sign}'
        reuse = retries.get(key)
        if reuse and not matching_vector(read(Path(reuse)/'metadata.json'), axis, sign, fd['steps']):
            raise ValueError('Refusing cross-parameter cached subpoints')
        tag = reserve_name(root, f'k12_axis{axis}_'+('minus' if sign < 0 else 'plus'),
                           output_parent=root/'runs', log_parent=root)
        out = root/'runs'/tag
        tmp = '/dev/shm/'+tag
        env = dict(cfg['environment'], D3FT_PHASE3_RESULTS_ROOT=str(root/'runs'),
            PYTHONPATH=cfg['runtime']+':'+cfg['environment']['PYTHONPATH'].split(':')[-1],
            CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), TMPDIR=tmp,
            PYTHONPYCACHEPREFIX=tmp+'/pycache', JAX_COMPILATION_CACHE_DIR=tmp+'/jax')
        command = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1', '-c4',
            'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"', 'E1-three-FD', 'env',
            'CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES'], 'timeout', str(seconds), e1.PY,
            '-u', str(Path(cfg['runtime'])/'validation/workflows/phase3/worker.py'),
            '--material', 'AlP', '--parameters', 'base', '--tag', tag, '--mode', 'evaluate',
            '--bulk-only', '--grid', '64', '--cutoff', '50', '--k', '12', '--epochs', '3000',
            '--empty', '0', '--strain-step', '0.005', '--a0',
            str(base['a0_A']+row['da0_dp'][axis]*sign*h), '--kpool', '--fused-report',
            '--fixed-mask', str(Path(row['source'])/'mask.npy'), '--vector', json.dumps(vector),
            '--pressure-tol', '1e-8']
        if reuse:
            command += ['--reuse-from', reuse]
        log = (root/(tag+'.log')).open('xb')
        proc = subprocess.Popen(command, env={**os.environ, **env}, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        rec = dict(start=now(), axis=axis, sign=sign, parameter=fd['parameter_order'][axis],
            vector=vector, tag=tag, jobid=250285, gpus=list(gpus), pid=proc.pid, output=str(out),
            command=command, environment=env, timeout_seconds=seconds, epochs=3000,
            independent_new_process=True, inherited_PID=None, reused_subpoints_from=reuse,
            status='running')
        dump(root/(tag+'_launch.json'), rec)
        active[gpus] = dict(process=proc, log=log, record=rec, monotonic=time.monotonic(),
                            reservation=2*seconds/3600)
        print('E1_THREE_START', tag, 'GPUs', list(gpus), flush=True)

    def finish(gpus, entry):
        nonlocal spent, terminal
        rec, proc = entry['record'], entry['process']
        logpath = Path(rec.get('original_log', root/(rec['tag']+'.log')))
        logtext = logpath.read_text()
        if entry.get('observed'):
            hours = len(gpus)*max(0., (dt.datetime.now().astimezone()-dt.datetime.fromisoformat(rec['start']).astimezone()).total_seconds())/3600
            done = (Path(rec['output'])/'summary.json').exists() and ('DONE '+rec['tag']) in logtext
            code = observed_completion_code(rec['tag'], (Path(rec['output'])/'summary.json').exists(), logtext)
            rec.update(returncode=None, inherited_waitpid_exit_code_unavailable=True,
                completion_evidence='DONE plus summary' if done else 'child ended without DONE/summary',
                external_timeout_reported_in_srun_log='exit code 124' in logtext.lower())
        else:
            entry['log'].close()
            hours = len(gpus)*(time.monotonic()-entry['monotonic'])/3600
            code = proc.returncode
            rec['returncode'] = code
        spent += hours
        rec.update(end=now(), GPU_hours=hours, status='completed' if code == 0 else 'failed')
        dump(root/(rec['tag']+'_launch.json'), rec)
        release(gpus)
        out = Path(rec['output'])
        key = f"{rec['axis']},{rec['sign']}"
        if code == 124:
            count = len(read(out/'points.json')) if (out/'points.json').exists() else 0
            old = Path(rec['reused_subpoints_from']) if rec['reused_subpoints_from'] else None
            previous = len(read(old/'points.json')) if old and (old/'points.json').exists() else 0
            if count > previous and not terminal:
                retries[key] = str(out)
                pending.append((rec['axis'], rec['sign']))
                e1.note('E1三参数 '+rec['tag']+' timeout124，保留失败；仅同参数合格子点可在截止/预算内新名续算。')
                return
        if code != 0 or not (out/'summary.json').exists() or ('DONE '+rec['tag']) not in logtext:
            terminal = terminal or 'FD child failed or incomplete: '+rec['tag']
            return
        try:
            derived = e1.derive(out)
            dump(out/'equilibrium_jacobian.json', derived)
            summary = read(out/'summary.json')
            completed[key] = dict(folder=str(out), summary=summary)
            dump(root/'completed_points.json', completed)
            axis = rec['axis']
            if all(f'{axis},{s}' in completed for s in (-1, 1)):
                pair = [completed[f'{axis},{s}']['summary'] for s in (-1, 1)]
                h = fd['steps'][axis]
                checks = [component(k, float(gradient*2*h), pair[1][k]-pair[0][k], base['a0_A'], .02)
                    for k, gradient in [('a0_A', row['da0_dp'][axis]), ('B_GPa', row['dB_equilibrium_dp'][axis])]]
                roots = all(abs(s['equilibrium_pressure_GPa'])/(3*s['B_GPa']) < 1e-7 for s in pair)
                certificate = dict(axis=axis, parameter=fd['parameter_order'][axis], step=h,
                    rows=checks, root_passed=roots, passed=roots and all(c['joint_gate_passed'] for c in checks))
                dump(root/f'FD_axis{axis}.json', certificate)
                if not certificate['passed']:
                    raise ValueError('Unchanged FD gate failed '+str(axis))
                certificates[str(axis)] = certificate
                e1.note('E1三参数子步骤 '+fd['parameter_order'][axis]+' a0/B原FD门PASS；其余四列未验证、非本稿所用。')
                print('E1_PAIR_PASS '+fd['parameter_order'][axis]+'\na0/B既定门未改变。\n原始子点和失败均保留。', flush=True)
        except Exception as exc:
            terminal = terminal or str(exc)

    report('handover_waiting')
    while not (root/'handover_ready.json').exists():
        time.sleep(1)
    try:
        while True:
            if not legacy_recorded and not alive(legacy['pid']):
                # The retired parent's waitpid exit code is unavailable; certify
                # only physical files, never claim the legacy pair was validated.
                out = Path(legacy['output'])
                dump(root/'legacy_natural_end.json', dict(time=now(), pid=legacy['pid'],
                    returncode=None, observed_only=True, scope='theta_s minus: non-manuscript diagnostic',
                    summary_exists=(out/'summary.json').exists(),
                    DONE_present=('DONE '+legacy['tag']) in Path(cfg['legacy_log']).read_text(),
                    reserved_GPU_hours_upper_bound=16, no_GPU_signal_sent=True))
                legacy_recorded = True
            for gpus, entry in list(active.items()):
                ended = not alive(entry['record']['pid']) if entry.get('observed') else entry['process'].poll() is not None
                if ended:
                    del active[gpus]
                    finish(gpus, entry)
            if len(certificates) == 3 and not terminal:
                break
            if dt.datetime.now().astimezone() >= deadline:
                terminal = terminal or '24h deadline: three required FD pairs not all accepted'
            if terminal:
                if not (root/'E7_fallback.json').exists():
                    policy_fallback(terminal)
                    pending.clear()
                    e1.note('E1三参数未完成/失败：'+terminal+'；E7按预注册改Si/BN/diamond三折，AlP仅保留K12晶格常数；旧结果和门不变。')
                if not active:
                    break
                report('fallback_no_new_dispatch_waiting_active_children', reason=terminal)
                time.sleep(15)
                continue
            if pending:
                try:
                    probe = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
                        '-N1', '-n1', '-c1', 'nvidia-smi', '--query-gpu=index,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
                    if probe.returncode:
                        raise RuntimeError(probe.stderr)
                    memory = {int(v[0]):int(v[1]) for line in probe.stdout.splitlines() if len(v:=line.split(',')) == 2}
                    dump(root/'GPU_probe.json', dict(time=now(), memory_MiB=memory))
                    for gpus in groups:
                        occupied = {g for gs in active for g in gs}
                        if not pending or set(gpus) & occupied:
                            continue
                        seconds = allowed_seconds(deadline, spent, sum(e['reservation'] for e in active.values()), 250,
                                                  maximum_seconds=cfg.get('timeout_seconds', 43200))
                        if seconds < 900:
                            if not active:
                                terminal = 'Insufficient E1 budget or time to reserve another point without changing gates'
                            continue
                        if claim(gpus, memory):
                            try:
                                start(pending.pop(0), gpus, seconds)
                            except Exception:
                                release(gpus)
                                raise
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    dump(root/'probe_retry.json', dict(time=now(), error=str(exc)))
            report('three_parameter_FD_running_or_waiting_released_cards')
            time.sleep(15)
        result = dict(time=now(), passed=terminal is None and len(certificates) == 3,
            adopted_k=12, certified_observables=['a0_A', 'B_GPa'],
            certified_parameter_order=[fd['parameter_order'][a] for a in AXES],
            three_parameter_FD=list(certificates.values()),
            unvalidated_non_manuscript_parameters=[fd['parameter_order'][a] for a in OMITTED_AXES],
            seven_independent_reoptimized_parameter_FD=False,
            total_GPU_hours_upper_bound=spent, deadline=cfg['deadline'], reason=terminal,
            old_SG15_rows_replaced=False, scope='three E7 parameters only')
        if result['passed']:
            # Store only the three accepted columns; full raw seven-column row
            # remains in its original, explicitly uncertified location.
            accepted = {k:row[k] for k in ('k', 'a0_A', 'B_GPa', 'source', 'source_hashes')}
            accepted.update(parameter_order=result['certified_parameter_order'],
                da0_dp=[row['da0_dp'][a] for a in AXES],
                dB_equilibrium_dp=[row['dB_equilibrium_dp'][a] for a in AXES],
                accepted_for_calibration=True, acceptance_scope='three parameters; a0/B only',
                seven_independent_reoptimized_parameter_FD=False,
                parameter_FD_certificate=str(root/'acceptance.json'),
                QE_certificate=str(X/'formal_validation/QE_acceptance.json'))
            dump(root/'accepted_response_row_k12_three_parameters.json', accepted)
            result['response_row'] = str(root/'accepted_response_row_k12_three_parameters.json')
            policy = read(Y/'material_policy.json')
            policy['AlP_three_parameter_FD_certificate'] = str(root/'acceptance.json')
            policy['AlP_response_row'] = result['response_row']
            policy['AlP_reference_and_other_training_gates_still_required'] = True
            dump(Y/'material_policy.json', policy)
        dump(root/'acceptance.json', result)
        dump(E/'latest_acceptance.json', result)
        report('complete_three_parameter_scope' if result['passed'] else 'AlP_excluded_E7_threefold', passed=result['passed'], reason=terminal)
        e1.note('E1本稿三参数范围 '+('完成' if result['passed'] else '未完成、AlP退出E7')+'；证书'+str(root/'acceptance.json')+'；未声称全七参数通过。')
        print('E1三参数范围 '+('PASS' if result['passed'] else 'FAIL/截止退出')+'\n证书和E7分支已落盘。\n四个非本稿参数未验证；旧数据保留。', flush=True)
    except Exception as exc:
        # An orchestration exception must not leave the deadline policy hanging.
        # Existing timeout-bounded children are not signalled or deleted.
        policy_fallback('Controller error: '+repr(exc))
        report('controller_error_no_new_dispatch', error=repr(exc))
        raise
    finally:
        busy = {g for gs in active for g in gs}
        for gpus in set(groups) | {tuple(r['gpus']) for r in cfg.get('observed_active_GPU_children', [])}:
            release([g for g in gpus if g not in busy])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
