"""Rebuild operational v3 progress from recorded plans and raw endpoint files.

Controller liveness is local only and is never treated as GPU-process proof.
This reporter neither launches calculations nor promotes a trial to acceptance.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def live(pid):
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def rebuild(campaign):
    launches = [read(campaign/'Si_theta_p_v3_fd_launch.json', {})]
    launches += read(campaign/'additional_fd_launches.json', [])
    superseded_records = {x['supersedes_record'] for x in launches if x.get('supersedes_record')}
    sequences = []
    for launch in launches:
        if not launch or launch.get('superseded_by') or launch.get('record') in superseded_records:
            continue
        plan = read(launch['plan'], {})
        protocol = plan.get('protocol', {})
        record = Path(launch['record'])
        state = read(record/'status.json', {})
        audit = Path(launch.get('audit_record',
            str(record).replace('_sequence', '_audit')))
        acceptance = read(audit/'acceptance.json', {})
        endpoints = []
        for pair in protocol.get('pairs', []):
            for sign in ['plus', 'minus']:
                root = Path(pair[sign])
                points = read(root/'points.json', [])
                summary = read(root/'summary.json')
                endpoints.append(dict(path=str(root), sign=sign, alpha=pair['alpha'],
                    summary_present=summary is not None, completed_points=len(points),
                    last_point=points[-1]['label'] if points else None,
                    last_electronic_residual=points[-1].get('residual') if points else None))
        sequences.append(dict(material=read(Path(protocol['base'])/'summary.json', {}).get('material'),
            parameter=protocol.get('parameter_name'), job=launch['job'], gpus=launch['gpus'],
            record=str(record), controller_pid=launch['controller_pid'],
            controller_live_local=live(launch['controller_pid']), state=state.get('state'),
            experiment=state.get('experiment'), registered_endpoints=len(endpoints),
            completed_endpoint_summaries=sum(x['summary_present'] for x in endpoints),
            audit_state=read(audit/'status.json', {}).get('state'),
            scientific_gate_passed=bool(acceptance.get('passed',
                acceptance.get('frozen_target_scope_passed', False))), endpoints=endpoints))
    trials = []
    for launch_path in sorted((campaign/'actual_trials').glob('*/launch.json')):
        launch = read(launch_path, {})
        record = Path(launch['record'])
        result = read(record/'actual_trial.json', {})
        trials.append(dict(trial=launch['trial'], controller_pid=launch['controller_pid'],
            controller_live_local=live(launch['controller_pid']),
            state=read(record/'status.json', {}).get('state'),
            actual_evaluation_complete=result.get('actual_evaluation_complete', False),
            accepted=result.get('accepted', False), rho=result.get('rho')))
    numerical = None
    numerical_launch = read(campaign/'candidate_numerical_execution_launch.json', {})
    if numerical_launch:
        record = Path(numerical_launch['record'])
        state = read(record/'status.json', {})
        protocol = read(record/'registered_protocol/protocol.json', {})
        results = [read(p, {}) for p in sorted(record.glob('*_result.json'))]
        completed = read(record/'audit.json', {})
        numerical = dict(controller_pid=numerical_launch['controller_pid'],
            controller_live_local=live(numerical_launch['controller_pid']),
            state=state.get('state'), material=state.get('material'), axis=state.get('axis'),
            registered_pairs=len(protocol.get('comparisons', [])),
            recorded_pair_results=len(results), passed_pair_results=sum(bool(x.get('passed')) for x in results),
            all_registered_pairs_passed=completed.get('all_registered_pairs_passed',False),
            accepted=False, scope='Conditional numerical execution; registration is not calculation or acceptance.')
    reference_spectra = []
    reference_directory = campaign/'reference_spectra_20261001'
    for launch in read(reference_directory/'launches.json', []):
        record = Path(launch['record']); plan = read(launch['plan'], {})
        acceptance = read(record/'acceptance.json', {})
        reference_spectra.append(dict(job=launch['job'],gpus=launch['gpus'],
            materials=list(plan.get('spectra',{})),controller_pid=launch['controller_pid'],
            controller_live_local=live(launch['controller_pid']),
            state=read(record/'status.json',{}).get('state'),
            completed_summaries=sum((Path(p)/'summary.json').exists() for p in plan.get('spectra',{}).values()),
            scientifically_passed=bool(acceptance.get('passed')),accepted=False))
    reference_resolution = None
    reference_directory = campaign/'reference_numerical_20261001'
    reference_launches = read(reference_directory/'launches.json', [])
    if reference_launches:
        evidence=reference_directory/'evidence'
        results=[read(p,{}) for p in evidence.glob('*_result.json')]
        reference_resolution=dict(registered_pairs=35,recorded_pair_results=len(results),
            passed_pair_results=sum(bool(row.get('passed')) for row in results),
            all_registered_pairs_passed=read(evidence/'audit.json',{}).get('all_registered_pairs_passed',False),
            lanes=[dict(job=launch['job'],gpus=launch['gpus'],materials=launch['materials'],
                controller_pid=launch['controller_pid'],controller_live_local=live(launch['controller_pid']),
                state=read(Path(launch['record'])/'status.json',{}).get('state')) for launch in reference_launches],
            accepted=False,scope='Zero-parameter reference resolution only; registration is not a scientific PASS.')
    obj = dict(updated=datetime.datetime.now().isoformat(), new_fd_sequences=sequences,
        registered_new_fd_endpoints=sum(x['registered_endpoints'] for x in sequences),
        completed_endpoint_summaries=sum(x['completed_endpoint_summaries'] for x in sequences),
        new_direction_gates_passed=sum(x['scientific_gate_passed'] for x in sequences),
        nonlinear_trials=trials,
        candidate_numerical_validation=numerical,
        reference_spectra=reference_spectra,
        reference_numerical_validation=reference_resolution,
        nonlinear_evaluations_completed=sum(x['actual_evaluation_complete'] for x in trials),
        whole_material_holdout_results_completed=0, phase3_complete=False,
        remaining=['current-reference full-mesh spectra and numerical budgets', 'parameter derivative and candidate numerical budgets',
            'accepted shared nonlinear calibration', 'independent whole-material refits and final evaluation'],
        scope='Derived progress only. A summary file is not a scientific PASS. Local controller liveness is not node/GPU worker proof.',
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    tmp = campaign/'EXECUTION_SNAPSHOT.tmp'
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(campaign/'EXECUTION_SNAPSHOT.json')
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign', type=Path)
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    while True:
        result = rebuild(args.campaign.resolve())
        print(json.dumps({k: result[k] for k in ['updated', 'registered_new_fd_endpoints',
            'completed_endpoint_summaries', 'new_direction_gates_passed',
            'nonlinear_evaluations_completed']}), flush=True)
        if not args.watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
