"""Queue independent zero-initialized holdout trials after reference validation.

Only first proposals are executed and audited here. This neither decides that a
fit has converged nor evaluates a held material, and never inherits all-fit
parameters or uses held outcomes for scheduling or acceptance.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_v3_lm_step import derivative_evidence, numerical_envelope, read, sha, verify_hashes
from fit_step import dataset


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def reference_ready(plan):
    record = Path(plan['reference_record'])
    if (record/'audit.json').exists():
        try:
            report = read(record/'audit.json')
        except json.JSONDecodeError:
            # The original assembler publishes this file by writing it directly.
            # A transient partial read is not an authoritative terminal failure.
            report = None
        if report is not None:
            if not report.get('all_registered_pairs_passed'):
                raise ValueError('Reference numerical campaign has unresolved failures')
            return True
    state = read(record/'status.json') if (record/'status.json').exists() else {}
    if state.get('state') in ['reference_assembly_failed', 'reference_lanes_incomplete']:
        raise ValueError('Reference assembly failed; holdout trials cannot start')
    try:
        os.kill(plan['reference_controller_pid'], 0)
    except ProcessLookupError:
        raise ValueError('Reference controller missing without a completed audit')
    return False


def run_fold(campaign, descriptor, before, pins, record):
    verify_hashes(pins)
    protocol = read(campaign/'protocol.json')
    fold = descriptor['fold']
    proposal = read(campaign/(fold+'_proposal.json'))
    held = protocol['folds'][fold]
    vector, rows, _, _ = dataset(protocol['manifest'], held)
    if any(vector) or proposal['starting_vector'] != vector.tolist() or proposal['holdouts'] != held:
        raise ValueError('Holdout trial is not independently initialized at its registered untrained center')
    # Reconstruct the reference and FD evidence for this exact training split
    # before spending GPUs on its first candidate.
    numerical_envelope(before, protocol['manifest'], rows)
    derivative_evidence(descriptor['fd_index'], protocol['manifest'], rows, vector.tolist())
    workflow = Path(__file__).parent
    trial_dir = Path(descriptor['trial_directory'])
    if trial_dir.exists():
        raise ValueError('Holdout trial already exists; inspect rather than restart')
    with (record/(fold+'_prepare.log')).open('x') as log:
        subprocess.run([sys.executable, str(workflow/'prepare_v3_trial.py'), str(campaign),
            '--fold', fold, '--output', str(trial_dir), '--spectrum-executor', 'guarded'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    trial = read(trial_dir/'trial.json')
    verify_hashes(trial['preparation_source_hashes'])
    if trial['holdouts'] != held or set(trial['training_materials']).intersection(held):
        raise ValueError('Prepared trial includes a held material')
    if read(trial['proposal']) != proposal:
        raise ValueError('Prepared trial changed its independent proposal')
    with (trial_dir/'controller.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(workflow/'run_v3_trial.py'),
            str(trial_dir/'trial.json'), '--record', str(trial_dir/'execution')],
            stdout=log, stderr=subprocess.STDOUT)
        put(trial_dir/'launch.json', dict(controller_pid=process.pid, trial=str(trial_dir/'trial.json'),
            record=str(trial_dir/'execution'), trial_sha256=sha(trial_dir/'trial.json'),
            script_sha256=sha(workflow/'run_v3_trial.py'), time=datetime.datetime.now().isoformat(),
            scope='Conditional independent first proposal only; no completed refit or held evaluation.'))
        code = process.wait()
    verify_hashes(pins)
    actual_path = trial_dir/'execution/actual_trial.json'
    if code or not actual_path.exists() or not read(actual_path).get('actual_evaluation_complete'):
        raise ValueError('Independent holdout candidate did not complete its actual evaluation')
    actual = read(actual_path)
    numeric = trial_dir/'candidate_numerical'
    if actual['nominal_actual_loss_gate']:
        with (trial_dir/'candidate_numerical.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(workflow/'run_v3_numerical_validation_parallel.py'),
                str(trial_dir), '--campaign', str(campaign), '--record', str(numeric)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (numeric/'audit.json').exists():
            raise ValueError('Independent candidate numerical campaign has no complete audit')
        verify_hashes(pins)
        source = Path(descriptor['jacobian_protocol'])
        # The prospective numerical response policy is unchanged; only the
        # explicit original fold FD source is substituted before its LM audit.
        jacobian = read(source)
        original = str(campaign/'material_fd_coverage_20261001/all_fd_evidence_index.json')
        jacobian['input_sha256'].pop(original)
        jacobian['input_sha256'][descriptor['fd_index']] = sha(descriptor['fd_index'])
        put(numeric/'jacobian_resolution_protocol.json', jacobian)
    # A genuine nominal-loss rejection needs no invented numerical audit.
    # The original auditor returns its real rejection before numerical gating.
    verify_hashes(pins)
    decision_dir = trial_dir/'first_lm_step'
    with (trial_dir/'first_lm_step.log').open('x') as log:
        subprocess.run([sys.executable, str(workflow/'audit_v3_lm_step.py'), str(trial_dir/'trial.json'),
            '--before-numerical', str(before), '--after-numerical', str(numeric),
            '--fd-index', descriptor['fd_index'], '--out', str(decision_dir)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    decision = read(decision_dir/'acceptance.json')
    if decision['decision'] == 'missing_or_invalid_acceptance_evidence':
        raise ValueError('Independent first-step acceptance evidence is invalid: '+decision.get('error',''))
    return dict(fold=fold, trial=str(trial_dir), decision=decision['decision'],
        accepted=decision['accepted'], refit_complete=False, held_material_evaluated=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args()
    plan = read(args.plan)
    pins = plan['source_sha256'] | {str(args.plan.resolve()): sha(args.plan)}
    verify_hashes(pins)
    if [d['fold'] for d in plan['folds']] != ['leave_diamond_out', 'leave_BN_out']:
        raise ValueError('Independent holdout fold order changed')
    campaign = Path(plan['campaign'])
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    put(args.record/'plan.json', plan)
    def status(state, **fields):
        value = dict(state=state, time=datetime.datetime.now().isoformat(),
            refit_complete=False, held_materials_evaluated=False, **fields)
        temporary = args.record/'status.tmp'; temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(args.record/'status.json')
        print(json.dumps(value), flush=True)
    try:
        while True:
            verify_hashes(pins)
            if reference_ready(plan):
                break
            status('waiting_for_completed_reference_validation')
            time.sleep(30)
        results = []
        for descriptor in plan['folds']:
            status('executing_independent_first_trial', fold=descriptor['fold'], completed=results)
            results.append(run_fold(campaign, descriptor, Path(plan['reference_record']), pins, args.record))
            put(args.record/(descriptor['fold']+'_result.json'), results[-1])
        status('completed_independent_first_trials', results=results)
    except Exception as error:
        status('holdout_first_trial_queue_failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
