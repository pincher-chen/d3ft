"""Audit an actual first trial, including an ineligible nominal-loss result.

Numerical execution may terminate without audit.json after a real loss
rejection. That must still receive a replayable LM decision. Only a proved
loss/rho rejection prepares a retry; this module never launches GPU work.
"""
import json
from pathlib import Path
import subprocess
import sys

from run_v3_numerical_validation import sha, verify_hashes


def read(path):
    return json.loads(Path(path).read_text())


def actual_trial(campaign, candidate_record):
    completed = candidate_record / 'audit.json'
    if completed.exists():
        return Path(read(completed)['trial']) / 'trial.json'
    state = read(candidate_record / 'status.json')
    if state.get('state') != 'actual_trial_not_eligible':
        return None
    protocol = read(candidate_record / 'registered_protocol/protocol.json')
    trials = [Path(p) for p in protocol['input_sha256'] if Path(p).name == 'trial.json']
    if len(trials) != 1:
        raise ValueError('No unique registered initial trial')
    recovery = campaign / 'Si_theta_p_timeout_recovery/status.json'
    if recovery.exists():
        recovered = read(recovery)
        if recovered.get('state') == 'fd_recovery_passed_and_conditional_trial_rewired':
            trials = [Path(recovered['new_trial']) / 'trial.json']
    trial_path = trials[0]
    trial = read(trial_path); proposal = read(trial['proposal'])
    if trial['fold'] != 'all' or proposal['candidate_vector'] != protocol['candidate_vector']:
        raise ValueError('Effective first trial differs from original registration')
    verify_hashes(trial['preparation_source_hashes'])
    actual = read(trial_path.parent / 'execution/actual_trial.json')
    if actual != state['actual'] or not actual.get('actual_evaluation_complete'):
        raise ValueError('Ineligible nominal result is not the complete recorded actual trial')
    verify_hashes(actual['source_hashes'])
    return trial_path


def finalize(campaign, before, after, out):
    campaign, before, after, out = map(Path, [campaign, before, after, out])
    trial = actual_trial(campaign, after)
    if trial is None:
        return dict(state='no_complete_actual_trial_for_lm_audit', accepted=False, refit_complete=False)
    command = [sys.executable, '-u', str(Path(__file__).with_name('audit_v3_lm_step.py')), str(trial),
               '--before-numerical', str(before), '--after-numerical', str(after),
               '--fd-index', str(campaign / 'material_fd_coverage_20261001/all_fd_evidence_index.json'),
               '--out', str(out / 'first_lm_step')]
    with (out / 'first_lm_step.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    acceptance_path = out / 'first_lm_step/acceptance.json'
    if result.returncode or not acceptance_path.exists():
        return dict(state='first_lm_audit_failed_without_decision', returncode=result.returncode,
                    accepted=False, refit_complete=False)
    decision = read(acceptance_path)
    report = dict(state='completed_first_lm_audit', decision=decision['decision'],
                  lm_audit_returncode=result.returncode, accepted=False, refit_complete=False)
    if decision['decision'] == 'rejected_actual_loss' and decision.get('accepted') is False:
        helper = Path(__file__).with_name('prepare_v3_rejected_trial.py')
        policy = read(campaign / 'rejected_trial_preflight_20261001/prospective_policy.json')
        if sha(helper) != policy['script_sha256']:
            raise ValueError('Prospectively registered rejection helper changed')
        registered = [Path(p) for p in read(trial)['preparation_source_hashes']
                      if Path(p).name == 'fd_gate_registry.json']
        if len(registered) != 1:
            raise ValueError('No unique center registry')
        index = read(registered[0].parent / 'protocol.json').get('iteration_index', 0) + 1
        retry = out / 'rejected_retry'
        with (out / 'rejected_retry.log').open('x') as log:
            prepared = subprocess.run([sys.executable, '-u', str(helper), str(acceptance_path),
                '--campaign', str(campaign), '--trial-index', str(index), '--out', str(retry)],
                stdout=log, stderr=subprocess.STDOUT)
        report.update(retry_preparation_returncode=prepared.returncode, retry_record=str(retry),
                      actual_retry_calculations_launched=0)
    return report
