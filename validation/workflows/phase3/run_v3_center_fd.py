"""Execute registered training-only FD controls at a genuinely accepted center.

Each existing allocation runs its controls sequentially; independent allocations
can progress together. Original worker commands, idle-device checks and scoped
two-step auditors are retained. This is not a nonlinear-fit stopping decision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading

from prepare_v3_center_fd import accepted_center, LANES
from audit_v3_lm_step import derivative_evidence, read, sha, verify_hashes
from fit_step import dataset


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def validate(preparation, acceptance):
    preparation, acceptance = Path(preparation).resolve(), Path(acceptance).resolve()
    index = read(preparation/'fd_evidence_index.json')
    verify_hashes(index['registered_source_sha256'])
    if index['registered_source_sha256'].get(str(acceptance)) != sha(acceptance):
        raise ValueError('Execution acceptance is not the registered center source')
    accepted, trial, _ = accepted_center(acceptance)
    work = read(preparation/'center_manifest.json')
    if work != read(trial['candidate']):
        raise ValueError('Execution manifest is not the accepted training center')
    held = read(trial['proposal'])['holdouts']
    if set(work).intersection(held) or index['holdouts'] != held or index['fold'] != trial['fold']:
        raise ValueError('Execution includes held materials or changes the fold')
    vector, rows, _, _ = dataset(preparation/'center_manifest.json', held)
    if vector.tolist() != accepted['candidate_vector'] or index['starting_vector'] != vector.tolist():
        raise ValueError('Execution FD center differs from accepted vector')
    registry = read(preparation/'fd_gate_registry.json')
    if registry['starting_vector'] != vector.tolist():
        raise ValueError('FD gate registry refers to another center')
    descriptors = read(preparation/'preparation.json')['experiments']
    if not descriptors or len({d['plan'] for d in descriptors}) != len(descriptors):
        raise ValueError('Empty or duplicate execution plans')
    if len({d['record'] for d in descriptors}) != len(descriptors):
        raise ValueError('Duplicate sequence records')
    if len({d['audit_record'] for d in descriptors}) != len(descriptors):
        raise ValueError('Duplicate audit records')
    if set(registry['acceptance_files']) != {str(Path(d['audit_record'])/'acceptance.json') for d in descriptors}:
        raise ValueError('Execution and registered acceptance files differ')
    lanes = {}
    sources = dict(index['registered_source_sha256'])
    for name in ['center_manifest.json', 'fd_evidence_index.json', 'fd_gate_registry.json', 'preparation.json']:
        sources[str(preparation/name)] = sha(preparation/name)
    for case in index['cases']:
        verify_hashes(case['base_input_sha256'])
        sources.update(case['base_input_sha256'])
        if case['base'] not in work.values():
            raise ValueError('FD case uses a material outside the accepted training set')
        if sources.get(case['source_protocol']) != case['source_protocol_sha256']:
            raise ValueError('FD case does not use its registered plan')
    for d in descriptors:
        if d['material'] not in work or (d['job'], d['gpus']) != LANES[d['material']]:
            raise ValueError('FD execution resource or training material changed')
        plan_path = Path(d['plan'])
        if sources.get(str(plan_path)) != sha(plan_path):
            raise ValueError('FD plan is not registered')
        plan = read(plan_path)
        if plan['jobid'] != str(d['job']) or plan['gpus'] != d['gpus'] or not plan['stop_on_failure']:
            raise ValueError('FD sequence resource or failure handling changed')
        if plan['protocol']['base'] != work[d['material']] or len(plan['experiments']) != 4:
            raise ValueError('FD plan does not contain four endpoints at this center')
        meta = read(Path(work[d['material']])/'metadata.json')
        worker = Path(next(x for x in plan['experiments'][0]['command'] if x.endswith('/worker.py')))
        runtime = worker.parents[3]
        sources.update({str(runtime/name):digest for name,digest in meta['code_sha256'].items()})
        sources[str(worker)] = meta['workflow_sha256']['worker.py']
        sources.update(meta['pseudopotentials'])
        for path in [d['record'], d['audit_record']]:
            if Path(path).exists():
                raise ValueError('FD record already exists; inspect it rather than restarting')
        key = (d['job'], tuple(d['gpus']))
        lanes.setdefault(key, []).append(d)
    keys = list(lanes)
    for i, (job, gpus) in enumerate(keys):
        if any(job == other and set(gpus).intersection(cards) for other, cards in keys[i+1:]):
            raise ValueError('Independent FD lanes overlap devices')
    workflow = Path(__file__).parent
    for name in ['run_v3_center_fd.py', 'prepare_v3_center_fd.py', 'run_gpu_sequence.py',
                 'audit_frozen_targets_fd_sequence.py', 'check_parameter_fd.py',
                 'fd_acceptance.py', 'equilibrium_jacobian.py', 'force_acceptance.py',
                 'audit_v3_lm_step.py', 'fit_step.py']:
        sources[str(workflow/name)] = sha(workflow/name)
    verify_hashes(sources)
    return lanes, sources, vector.tolist(), rows


def execute_lane(descriptors, sources, stop, record):
    results = []
    workflow = Path(__file__).parent
    for d in descriptors:
        if stop.is_set():
            break
        verify_hashes(sources)
        name = Path(d['plan']).stem
        with (record/(name+'_sequence.log')).open('x') as log:
            process = subprocess.Popen([sys.executable, str(workflow/'run_gpu_sequence.py'),
                d['plan'], '--record', d['record']], stdout=log, stderr=subprocess.STDOUT)
            launch = dict(plan=d['plan'], plan_sha256=sha(d['plan']), record=d['record'],
                controller_pid=process.pid, job=d['job'], gpus=d['gpus'], audit_record=d['audit_record'])
            launch_path = record/(name+'_launch.json')
            put(launch_path, launch)
            returncode = process.wait()
        # The original runner can exit zero after a failed worker. Its auditor
        # verifies all four worker exit codes and the actual endpoint evidence.
        verify_hashes(sources)
        with (record/(name+'_audit.log')).open('x') as log:
            audit_process = subprocess.run([sys.executable, str(workflow/'audit_frozen_targets_fd_sequence.py'),
                str(launch_path), '--record', d['audit_record']], stdout=log, stderr=subprocess.STDOUT)
        acceptance_path = Path(d['audit_record'])/'acceptance.json'
        passed = bool(returncode == 0 and audit_process.returncode == 0 and acceptance_path.exists()
            and read(acceptance_path).get('frozen_target_scope_passed') is True)
        result = dict(plan=d['plan'], sequence_returncode=returncode,
            audit_returncode=audit_process.returncode, passed=passed)
        put(record/(name+'_result.json'), result)
        results.append(result)
        if not passed:
            stop.set()
            break
    return results


def guarded_lane(descriptors, sources, stop, record):
    try:
        return execute_lane(descriptors, sources, stop, record)
    except Exception:
        stop.set()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('preparation', type=Path)
    ap.add_argument('acceptance', type=Path)
    ap.add_argument('--record', type=Path)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    lanes, sources, vector, rows = validate(args.preparation, args.acceptance)
    if args.check_only:
        print(json.dumps(dict(validated=True, registered_lanes=len(lanes),
            registered_plans=sum(map(len, lanes.values())), computations_launched=0)))
        return
    if args.record is None:
        ap.error('--record is required for execution')
    args.record.mkdir(parents=True, exist_ok=False)
    record = args.record.resolve()
    put(record/'execution_inputs.json', dict(source_sha256=sources,
        starting_vector=vector, acceptance=str(args.acceptance.resolve())))
    snapshot = record/'source_snapshot'; snapshot.mkdir()
    for path in sources:
        if Path(path).parent == Path(__file__).parent:
            (snapshot/Path(path).name).write_bytes(Path(path).read_bytes())
    def status(state, **fields):
        value = dict(state=state, time=datetime.datetime.now().isoformat(), refit_complete=False, **fields)
        temporary = record/'status.tmp'; temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(record/'status.json')
        print(json.dumps(value), flush=True)
    stop = threading.Event()
    status('running_registered_center_fd', registered_lanes=len(lanes))
    try:
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures = [pool.submit(guarded_lane, ds, sources, stop, record) for ds in lanes.values()]
            results = []
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception:
                    stop.set()
                    raise
        verify_hashes(sources)
        if len(results) != sum(map(len, lanes.values())) or not all(x['passed'] for x in results):
            status('center_fd_failed', results=results)
            return
        controls, raw_sources = derivative_evidence(args.preparation/'fd_evidence_index.json',
            args.preparation/'center_manifest.json', rows, vector)
        put(record/'coverage.json', dict(passed=True, **controls, source_sha256=raw_sources,
            execution_input_sha256=sources, held_materials_evaluated=False, refit_complete=False))
        status('completed_center_fd', passed=True, results=results)
    except Exception as error:
        stop.set()
        status('center_fd_error', error=str(error))
        raise


if __name__ == '__main__':
    main()
