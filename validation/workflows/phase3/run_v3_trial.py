"""Execute a complete nonlinear candidate after registered FD controls pass.

Records actual loss and fresh full-mesh spectra; never promotes missing numerical
certificates into acceptance. This is one trial, not a completed nonlinear refit.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_trial import audit_spectrum
from fit_step import dataset, validate_trial


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trial', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args(); trial = read(args.trial)
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record/'trial.json').write_text(json.dumps(trial, indent=2))
    def status(state, **kwargs):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(), **kwargs)
        tmp = args.record/'status.tmp'; tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(args.record/'status.json'); print(json.dumps(obj), flush=True)
    status('waiting_for_registered_fd_controls')
    while True:
        pending = []
        for item in trial['fd_acceptance_files']:
            path = Path(item)
            if not path.exists():
                sibling = path.parent/'status.json'
                if sibling.exists():
                    st = read(sibling)
                    if st.get('state') in ['worker_sequence_failed', 'pair_check_failed_without_certificate',
                        'controller_missing_without_successful_terminal_record']:
                        status('fd_dependency_failed', dependency=item, details=st); return
                pending.append(item); continue
            evidence = read(path)
            passed = evidence.get('joint_gate_passed',
                evidence.get('passed', evidence.get('frozen_target_scope_passed', False)))
            if not passed:
                status('fd_scientific_gate_failed', dependency=item); return
        if not pending: break
        time.sleep(30)
    for file, sha in trial['preparation_source_hashes'].items():
        if digest(file) != sha:
            status('frozen_source_changed', file=file); return
    proposal = read(trial['proposal'])
    baseline = read(trial['baseline_manifest']); candidate = read(trial['candidate'])
    spectra = read(trial['spectra'])
    if set(candidate) != set(proposal['input_fingerprints']) or set(spectra) != set(candidate):
        raise ValueError('Incomplete training set or changed holdout split')
    for material, fingerprints in proposal['input_fingerprints'].items():
        for file, sha in fingerprints.items():
            if digest(Path(baseline[material])/file) != sha:
                raise ValueError('Baseline changed')
    sequence = args.record/'sequence'
    status('executing_full_candidate_and_fresh_spectra')
    driver = Path(__file__).with_name('run_gpu_sequence.py')
    with (args.record/'sequence_controller.log').open('x') as log:
        result = subprocess.run([sys.executable, '-u', str(driver), trial['sequence_plan'],
            '--record', str(sequence)], stdout=log, stderr=subprocess.STDOUT)
    terminal = read(sequence/'status.json')
    expected_count = len(read(trial['sequence_plan'])['experiments'])
    if (result.returncode or terminal.get('state') != 'sequence_exited' or
        len(terminal.get('results', [])) != expected_count or
        any(x['returncode'] for x in terminal.get('results', []))):
        status('candidate_worker_failed', details=terminal); return
    validate_trial(trial['baseline_manifest'], trial['candidate'], proposal['holdouts'])
    for material, path in candidate.items():
        if read(Path(path)/'metadata.json')['code_sha256'] != read(Path(baseline[material])/'metadata.json')['code_sha256']:
            raise ValueError('Candidate core changed')
    spectra_evidence = {m: audit_spectrum(Path(candidate[m]), Path(spectra[m])) for m in candidate}
    v, before_rows, before, jacobian = dataset(trial['baseline_manifest'], proposal['holdouts'])
    cv, after_rows, after, _ = dataset(trial['candidate'], proposal['holdouts'])
    identity = lambda rows: [(r['material'], r['observable'], r['reference'], r['sigma']) for r in rows]
    if identity(before_rows) != identity(after_rows) or identity(before_rows) != identity(proposal['training_rows']):
        raise ValueError('Training observations changed')
    if v.tolist() != proposal['starting_vector'] or cv.tolist() != proposal['candidate_vector']:
        raise ValueError('Nonlinear evaluation uses the wrong vector')
    predicted = float(before@before-(before+jacobian@(cv-v))@(before+jacobian@(cv-v)))
    actual = float(before@before-after@after)
    rho = actual/predicted if predicted > 0 else None
    report = dict(accepted=False, actual_evaluation_complete=True,
        loss_before=float(before@before), loss_after=float(after@after),
        actual_reduction=actual, predicted_reduction=predicted, rho=rho,
        nominal_actual_loss_gate=bool(actual > 0 and rho is not None and rho > .1),
        full_spectra_passed=True, spectra=spectra_evidence,
        holdouts=proposal['holdouts'], training_rows=after_rows,
        completed_nonlinear_refit=False, whole_material_holdout_results_completed=False,
        missing_acceptance=['candidate numerical budgets', 'applicable full derivative acceptance'],
        scope='Complete actual first-trial evidence only. Missing budgets still forbid acceptance; continue real LM iterations after independent certification, then evaluate holdouts.',
        source_hashes={str(Path(path)/file): digest(Path(path)/file)
            for path in candidate.values() for file in ['metadata.json', 'summary.json', 'points.json', 'mask.npy']},
        script_sha256=digest(Path(__file__)))
    (args.record/'actual_trial.json').write_text(json.dumps(report, indent=2))
    status('completed_actual_trial_pending_numerical_acceptance', rho=rho, accepted=False)


if __name__ == '__main__':
    main()
