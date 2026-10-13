"""Wait for a registered FD sequence and audit both steps against its v3 base.

No worker restart, linear-result acceptance or relaxed resolution budgets.
The sequence must finish successfully before raw endpoints are compared.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('launch', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args()
    launch = json.loads(args.launch.read_text())
    plan_path = Path(launch['plan']); plan = json.loads(plan_path.read_text())
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != launch['plan_sha256']:
        raise ValueError('Registered sequence plan was modified')
    protocol = plan['protocol']; base = Path(protocol['base'])
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record/'protocol.json').write_text(json.dumps(protocol, indent=2))
    def status(state, **kwargs):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(), **kwargs)
        tmp = args.record/'status.tmp'; tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(args.record/'status.json'); print(json.dumps(obj), flush=True)
    status('waiting_for_fd_sequence')
    terminal = Path(launch['record'])/'status.json'
    while True:
        if terminal.exists():
            current = json.loads(terminal.read_text())
            if current.get('state') == 'sequence_exited':
                results = current.get('results', [])
                if (len(results) != len(plan['experiments']) or
                    any(x['returncode'] for x in results) or current.get('stopped_after_failure')):
                    status('worker_sequence_failed', sequence=current); return
                break
        try:
            os.kill(launch['controller_pid'], 0)
        except ProcessLookupError:
            status('controller_missing_without_successful_terminal_record'); return
        time.sleep(15)
    base_meta = json.loads((base/'metadata.json').read_text())
    if hashlib.sha256((base/'metadata.json').read_bytes()).hexdigest() != protocol['base_metadata_sha256']:
        raise ValueError('Frozen base metadata changed')
    checker = Path(__file__).with_name('check_parameter_fd.py')
    certificates = []
    status('checking_reoptimized_pairs')
    for pair in protocol['pairs']:
        for sign in ('minus', 'plus'):
            path = Path(pair[sign]); meta = json.loads((path/'metadata.json').read_text())
            if meta['code_sha256'] != base_meta['code_sha256']:
                raise ValueError('Endpoint core differs from frozen v3 base')
        path = args.record/('a'+str(pair['alpha'])+'_certificate.json')
        with (args.record/('a'+str(pair['alpha'])+'_check.log')).open('x') as log:
            proc = subprocess.run([sys.executable, str(checker), str(base), pair['minus'], pair['plus'],
                '--rtol', str(protocol['relative_agreement_tolerance']), '--out', str(path)],
                stdout=log, stderr=subprocess.STDOUT)
        if not path.exists():
            status('pair_check_failed_without_certificate', alpha=pair['alpha'], returncode=proc.returncode); return
        certificate = json.loads(path.read_text())
        certificates.append(certificate)
    index = protocol['parameter_index']; rtol = protocol['relative_agreement_tolerance']
    a, b = certificates; rows = []
    for row_a, row_b in zip(a['rows'], b['rows']):
        if row_a['observable'] != row_b['observable']:
            raise ValueError('Different observable sets in the two FD steps')
        dx_a, dx_b = a['parameter_difference'][index], b['parameter_difference'][index]
        ga, gb = row_a['actual_difference']/dx_a, row_b['actual_difference']/dx_b
        floor = 1e-7 if row_a['observable'] == 'a0_A' else 1e-3
        # Convert the original observable agreement floor into derivative units.
        tolerance = rtol*max(abs(ga), abs(gb)) + floor/min(abs(dx_a), abs(dx_b))
        rows.append(dict(observable=row_a['observable'], small_step_derivative=ga,
            large_step_derivative=gb, absolute_step_difference=abs(ga-gb),
            tolerance=tolerance, passed=abs(ga-gb)<=tolerance))
    passed = all(c['joint_gate_passed'] for c in certificates) and all(r['passed'] for r in rows)
    report = dict(passed=passed, both_joint_gates=[c['joint_gate_passed'] for c in certificates],
        two_step_consistency=rows, protocol=protocol,
        protocol_note='Two-step derivative tolerance converts the unchanged 2% agreement and 1e-7 Angstrom/1e-3 GPa absolute floors; signal budgets remain 1e-4*a0 and 0.1 GPa.',
        scope=protocol['scope'], plan_sha256=launch['plan_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest())
    (args.record/'acceptance.json').write_text(json.dumps(report, indent=2))
    status('completed_two_step_fd', passed=passed)


if __name__ == '__main__':
    main()
