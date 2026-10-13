"""Reconstruct full spectra at immutable v3 zero-parameter reference densities.

This independent evidence does not certify parameter derivatives or accept an LM
step. Explicit predecessor records serialize use of an existing allocation.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_trial import audit_spectrum


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(plan):
    for path, expected in plan['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Registered reference source changed: '+path)


def dependency_state(dependency):
    """Only an authoritative terminal record releases this resource lane.

    Failure of an unrelated FD experiment is allowed to release the lane: the
    zero-parameter reference spectrum remains an independent calculation.
    A missing controller is never treated as successful termination.
    """
    path = Path(dependency['record'])/'status.json'
    if path.exists():
        record = read(path)
        if record.get('state') in dependency['terminal_states']:
            return 'terminal'
    try:
        os.kill(dependency['controller_pid'], 0)
    except ProcessLookupError:
        return 'missing_controller'
    return 'waiting'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args(); plan = read(args.plan)
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    script_hash = digest(__file__)

    def status(state, **fields):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(),
                   accepted=False, controller_sha256=script_hash, **fields)
        temp = args.record/'status.tmp'; temp.write_text(json.dumps(obj, indent=2)+'\n')
        temp.replace(args.record/'status.json'); print(json.dumps(obj), flush=True)

    try:
        verify(plan)
        while True:
            states = [{**dep, 'state': dependency_state(dep)} for dep in plan['dependencies']]
            if any(row['state'] == 'missing_controller' for row in states):
                status('dependency_controller_missing', dependencies=states); return
            if all(row['state'] == 'terminal' for row in states):
                break
            status('waiting_for_predecessors', dependencies=states); time.sleep(30)
        verify(plan)
        # The existing driver checks device memory via srun immediately before
        # each command; it never submits or cancels an allocation.
        status('executing_reference_spectra')
        with (args.record/'sequence_controller.log').open('x') as log:
            result = subprocess.run([sys.executable, '-u', plan['sequence_driver'],
                plan['sequence_plan'], '--record', str(args.record/'sequence')],
                stdout=log, stderr=subprocess.STDOUT)
        verify(plan)
        terminal = read(args.record/'sequence/status.json')
        expected = [row['name'] for row in read(plan['sequence_plan'])['experiments']]
        rows = terminal.get('results', [])
        if (result.returncode or terminal.get('state') != 'sequence_exited' or
            [row['name'] for row in rows] != expected or any(row['returncode'] for row in rows)):
            status('reference_worker_failed', details=terminal); return
        evidence = {}
        for material, spectrum in plan['spectra'].items():
            evidence[material] = audit_spectrum(Path(plan['baseline'][material]), Path(spectrum))
        report = dict(passed=True, accepted=False, baseline_manifest=plan['baseline_manifest'],
            spectra=plan['spectra'], evidence=evidence, source_hashes=plan['source_hashes'],
            runtime_provenance=plan['runtime_provenance'], controller_sha256=script_hash,
            scope='Full SCF-mesh fixed-occupation evidence at the exact frozen reference density only; not an experimental gap, numerical-axis certificate, parameter derivative, or accepted nonlinear refit.')
        (args.record/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
        status('completed_reference_spectra', materials=list(evidence), passed=True)
    except Exception as exc:
        status('reference_spectrum_invalid', error=str(exc))
        raise


if __name__ == '__main__':
    main()
