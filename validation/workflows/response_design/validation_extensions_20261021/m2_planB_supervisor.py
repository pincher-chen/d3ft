"""Complete an already-running immutable M2 worker as authorized reverse3/forward1.

Never inject into, edit, or signal the worker before its first COMPLETE forward
record has passed the original numerical checks. Only then terminate unwanted
future repetitions. Preserve native artifacts and disclose the controlled exit.
This CPU-only supervisor itself exits normally and releases the existing M5 gate.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def read(p):
    return json.loads(Path(p).read_text())


def now():
    return dt.datetime.now().astimezone().isoformat()


def dump(p, value):
    p = Path(p)
    tmp = p.with_name(p.name + '.planB.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(reverse, forward, ref):
    """Reapply the original full54 gradient gates, not a reduced-data shortcut."""
    assert len(ref) == 54 and all(math.isfinite(x) for x in ref)
    assert len(reverse) == 3
    for i, row in enumerate(reverse):
        assert row['repeat'] == i and row['N'] == 54
        assert row['mode'] == 'reverse' and row['response_solves'] == 1
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
        assert len(row['gradient_eV']) == 54
        assert all(math.isfinite(x) for x in row['gradient_eV'])
        assert max(abs(x-y) for x,y in zip(row['gradient_eV'], ref)) <= 5e-6
    assert len(forward) >= 1
    row = forward[0]
    assert row['repeat'] == 0 and row['N'] == 54
    assert row['mode'] == 'forward' and row['response_solves'] == 54
    assert [x['axis'] for x in row['axes']] == list(range(54))
    assert len(row['gradient_eV']) == 54
    assert all(math.isfinite(x['seconds']) and x['seconds'] > 0 for x in row['axes'])
    for x,y in zip(row['gradient_eV'], ref):
        assert math.isfinite(x) and abs(x-y) <= 5e-6 + 1e-6*abs(y)
    expected = row['rhs_seconds'] + sum(x['seconds'] for x in row['axes'])
    assert abs(expected-row['seconds']) <= 1e-7*max(1, expected)
    return {'passed': True, 'reverse_repeats': 3, 'forward_repeats': 1,
            'forward_axes': 54, 'original_numerical_thresholds_unchanged': True}


def worker_candidates(script, output):
    found = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = (d/'cmdline').read_bytes().split(b'\0')
            args = [x.decode() for x in args if x]
            # Match Python argv exactly; never timeout, srun, bash or another task.
            if len(args) < 3 or 'python' not in Path(args[0]).name or args[1] != str(script):
                continue
            if args[args.index('--output')+1] != str(output):
                continue
            env = dict(x.split(b'=',1) for x in (d/'environ').read_bytes().split(b'\0') if b'=' in x)
            assert env[b'SLURM_JOB_ID'] == b'250285'
            assert env[b'SLURM_STEP_ID'] == b'15190'
            assert env[b'CUDA_VISIBLE_DEVICES'] == b'0'
            found.append({'pid': int(d.name), 'stat': (d/'stat').read_text(), 'argv': args})
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return found


def snapshot():
    for attempt in range(3):
        try:
            p = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, timeout=45, check=True)
            rows = [[int(float(x.strip())) for x in l.split(',')] for l in p.stdout.strip().splitlines()]
            assert len(rows) == 8
            return {'time': now(), 'node': 'hn35', 'jobid': 250285, 'raw': p.stdout, 'rows': rows}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def main(root):
    R = root; M = R/'M2-A'; D = M/'planB_20261021T2130'
    launch = read(M/'logs/exclusive_N54_r3_launch.json')
    O = Path(launch['output']); script = Path(launch['command'][launch['command'].index('--root')-1])
    frozen_sha = sha(script)
    assert os.environ['SLURMD_NODENAME'] == 'hn35'
    assert os.environ['SLURM_JOB_ID'] == '250285'
    assert read(D/'CPU_acceptance.json')['passed']
    assert len(worker_candidates(script, O)) == 1, 'Expected exactly the owned running M2 Python worker'
    with (D/'manager_started.json').open('x') as f:
        json.dump({'pid_on_hn35': os.getpid(), 'time': now()}, f)
    dump(D/'status.json', {'stage': 'waiting_CPU_handover', 'time': now()})
    while not (D/'handover_ready.json').exists():
        time.sleep(1)
    base = read(D/'ledger_before_handover.json')
    ref = read(Path(read(M/'protocol.json')['E3_root'])/'mg_mode_acceptance.json')['reverse_gradient_eV']
    active = next(j for j in base['jobs'] if j['experiment']=='M2-A' and j['pid']==launch['pid'])
    last_probe = 0
    outcome = None

    def ledger(end=None):
        elapsed = ((end or time.time())-launch['start_epoch'])/3600
        active['GPU_hours'] = elapsed
        if end is not None:
            active.update(end_epoch=end, effective_returncode=0 if outcome and outcome.get('passed') else 1,
                          completion_source=str(D/'summary_of_controlled_exit.json'))
        base['GPU_hours']['M2-A'] = elapsed; base['time'] = now()
        dump(R/'GPU_ledger.json', base)
        effective = json.loads(json.dumps(base))
        c = Path(read(R/'M3-A/current_continuation.json')['root'])/'GPU_ledger.json'
        effective['GPU_hours']['M3-A'] = read(c)['total_M3_GPU_hours']
        q = Path(read(R/'M5-A/current_sharded.json')['root'])/'GPU_ledger.json'
        effective['GPU_hours']['M5-A'] = read(q)['total_M5_GPU_hours']
        effective['M5_sharded_ledger'] = str(q)
        dump(R/'GPU_ledger_effective.json', effective)
        return elapsed

    try:
        while True:
            if time.monotonic()-last_probe >= 30:
                snap = snapshot(); last_probe = time.monotonic()
                with (M/'occupancy.jsonl').open('a') as f:
                    f.write(json.dumps(snap)+'\n')
                if any(row[1]>0 or row[2]>0 for row in snap['rows'] if row[0]!=0):
                    dump(M/'exclusive_violation.json', snap)
                ledger()
                status = read(O/'status.json')
                dump(D/'status.json', {'stage': 'waiting_complete_forward_repeat_0', 'time': now(),
                                       'native_worker_status': status, 'worker_source_unchanged': sha(script)==frozen_sha})
            # This file is written only AFTER all54 axes, synchronization and native checks.
            if (O/'forward_raw.json').exists():
                acceptance = validate(read(O/'reverse_raw.json'), read(O/'forward_raw.json'), ref)
                assert not (M/'exclusive_violation.json').exists(), 'Exclusivity violation'
                assert sha(script) == frozen_sha, 'Timing source changed during execution'
                assert ledger() <= 32, 'Original32GPUhcap exceeded'
                dump(D/'forward0_verified_before_stop.json', {**acceptance, 'time': now(),
                     'raw_SHA256': {n:sha(O/n) for n in ['reverse_raw.json','forward_raw.json']}})
                break
            status = read(O/'status.json')
            if status['stage'] == 'failed':
                raise RuntimeError('Native worker failed before a complete accepted forward: '+str(status))
            if not worker_candidates(script, O):
                raise RuntimeError('Native worker absent before complete forward record')
            time.sleep(1)

        # No process signal can occur before validate(full54) succeeded above.
        candidates = worker_candidates(script, O)
        assert len(candidates) <= 1
        stop = {'time': now(), 'reason': 'User planB removes forward repeats1/2; repeat0 already complete and checked',
                'not_a_physical_failure': True, 'no_scancel': True,
                'native_worker': candidates, 'signal': 'SIGTERM only to exact owned Python worker after forward0'}
        dump(D/'summary_of_controlled_exit.json', stop)
        if candidates:
            os.kill(candidates[0]['pid'], signal.SIGTERM)
        deadline = time.monotonic()+180
        while worker_candidates(script, O):
            if time.monotonic()>deadline:
                raise RuntimeError('Owned worker did not release after controlled termination; no escalation')
            time.sleep(1)
        end = time.time()
        reverse = read(O/'reverse_raw.json'); forward = read(O/'forward_raw.json')
        acceptance = validate(reverse, forward, ref)
        result = {'passed': True, 'time': now(), 'N': 54, 'reverse_repeats': 3, 'forward_repeats': 1,
                  'reverse': reverse, 'forward': forward[:1], 'node': 'hn35', 'jobid': 250285,
                  'native_output': str(O), 'native_artifacts_preserved': True,
                  'timing_worker_source_SHA256': frozen_sha,
                  'software_sha256': {n:sha(script.with_name(n)) for n in
                      ['gamma_response.py','geometry_layout.py','projected_solver.py',script.name]},
                  'timing_intervals_unchanged': True, 'warmup_memory': read(O/'memory_peak.json'),
                  'controlled_exit': str(D/'summary_of_controlled_exit.json'),
                  'scope': 'Three reverse timings and ONE forward timing; no forward repeat variability estimate',
                  'total_allocated_GPU_step_seconds': end-launch['start_epoch'],
                  'active_GPU_hours': (end-launch['start_epoch'])/3600,
                  'whole_eight_GPU_node_reservation_hours': 8*(end-launch['start_epoch'])/3600}
        # No invented primal residual or reconstructed warmup value: preserve native evidence.
        loglines = Path(launch['log']).read_text().splitlines()
        result['native_stage_records'] = [json.loads(l.split('E3_TIMING ',1)[1]) for l in loglines if l.startswith('E3_TIMING ')]
        result['warmup_accounting'] = 'Native warmup/compilation was outside measured intervals; native stage timestamps retained; launch overhead is included only in allocated walltime.'
        dump(D/'result.json', result)
        dump(D/'acceptance.json', {**acceptance, 'exclusive_violation': False, 'result': str(D/'result.json'),
                                  'primal_check_evidence': 'Unmodified native worker checks primal<=1e-7Ha before warmup; native log and frozen source retained',
                                  'no_native_exit0_claim': True})
        outcome = {'passed': True, 'outcome': 'exclusive3reverse1forward_completed',
                   'acceptance': str(D/'acceptance.json'), 'result': str(D/'result.json')}
    except Exception as ex:
        traceback.print_exc()
        outcome = {'passed': False, 'outcome': 'planB_stopped_incomplete_or_invalid', 'error': repr(ex),
                   'traceback': traceback.format_exc(), 'no_physical_failure_claim': True}
        end = time.time()
    used = ledger(end)
    summary = {**outcome, 'time': now(), 'GPU_hours': used, 'cap_GPU_hours': 32,
               'reverse_repeats_required': 3, 'forward_repeats_required': 1}
    dump(M/'summary.json', summary)
    dump(M/'controller_status.json', {'stage': 'terminal', **summary})
    dump(D/'status.json', {'stage': 'terminal', **summary})
    P = R.parents[3]
    line = f"\n- {now()} M2方案B终态：{summary['outcome']}；3反向+1正向，{used:.4f}/32 GPUh。原始数据全部保留，当前repeat0无中断；结果/验收`{D}/result.json`、`acceptance.json`。外围控制器受控结束未授权后续重复，不将worker退出信号误记为物理失败。M5按原八卡空闲条件接续。\n"
    for name in ['response_design_to_do_list.md','paper/response_design_EXPERIMENTS_20261014.md']:
        with (P/name).open('a') as f:
            f.write(line)
    print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    main(ap.parse_args().root)
