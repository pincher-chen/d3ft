"""Audit all frozen training rows, preserving diagnostics and four-row failures.

Unlike the older sparse auditor, diagnostic B rows may coexist with AlP's sole
training target a0. Neither diagnostics nor absent targets become certified.
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
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('launch',type=Path);ap.add_argument('--record',type=Path,required=True)
    args=ap.parse_args();launch=read(args.launch);plan=read(launch['plan']);protocol=plan['protocol']
    if sha(launch['plan']) != launch['plan_sha256']:
        raise ValueError('Registered FD plan changed')
    base=Path(protocol['base']);bm=read(base/'metadata.json');bs=read(base/'summary.json')
    target=Path(protocol['required_observables_source'])
    if sha(target) != protocol['required_observables_source_sha256'] or sha(base/'metadata.json') != protocol['base_metadata_sha256']:
        raise ValueError('Frozen FD input changed')
    required={r['observable'] for r in read(target)['rows'] if r['material']==bs['material']}
    if not required or required != set(protocol['required_observables']):
        raise ValueError('Scoped rows differ from all frozen training targets')
    args.record.mkdir(parents=True,exist_ok=False)
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.record/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def status(state,**kw):
        value=dict(state=state,time=datetime.datetime.now().isoformat(),**kw)
        tmp=args.record/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(args.record/'status.json')
        print(json.dumps(value),flush=True)
    status('waiting_for_fd_sequence')
    while True:
        terminal=read(Path(launch['record'])/'status.json') if (Path(launch['record'])/'status.json').exists() else {}
        if terminal.get('state')=='sequence_exited':
            results=terminal.get('results',[])
            if len(results)!=len(plan['experiments']) or any(r['returncode'] for r in results) or terminal.get('stopped_after_failure'):
                status('worker_sequence_failed',details=terminal);return
            break
        try:os.kill(launch['controller_pid'],0)
        except ProcessLookupError:
            status('controller_missing_without_successful_terminal_record');return
        time.sleep(15)
    certificates=[];index=protocol['parameter_index'];scale=protocol['parameter_scale']
    checker=Path(__file__).with_name('check_parameter_fd.py')
    for pair in protocol['pairs']:
        for name,sign in [('minus',-1),('plus',1)]:
            path=Path(pair[name]);meta=read(path/'metadata.json');summary=read(path/'summary.json')
            expected=np.array(bs['parameter_vector']);expected[index]+=sign*pair['alpha']*scale
            if meta['code_sha256']!=bm['code_sha256'] or not np.allclose(summary['parameter_vector'],expected,rtol=0,atol=1e-14):
                status('pair_check_failed_without_certificate',error='Frozen core or declared coordinate changed');return
        cert=args.record/('a'+str(pair['alpha'])+'_certificate.json')
        with (args.record/('a'+str(pair['alpha'])+'_check.log')).open('x') as log:
            subprocess.run([sys.executable,str(checker),str(base),pair['minus'],pair['plus'],
                '--rtol',str(protocol['relative_agreement_tolerance']),'--out',str(cert)],stdout=log,stderr=subprocess.STDOUT)
        if not cert.exists():
            status('pair_check_failed_without_certificate',alpha=pair['alpha']);return
        certificates.append(read(cert))
    a,b=certificates;ar={r['observable']:r for r in a['rows']};br={r['observable']:r for r in b['rows']}
    if not required.issubset(ar) or not required.issubset(br):
        status('pair_check_failed_without_certificate',error='Missing frozen training row');return
    rows=[]
    for key in sorted(required):
        dx=[c['parameter_difference'][index] for c in certificates]
        ga,gb=ar[key]['actual_difference']/dx[0],br[key]['actual_difference']/dx[1]
        floor=1e-7 if key=='a0_A' else 1e-3
        tolerance=protocol['relative_agreement_tolerance']*max(abs(ga),abs(gb))+floor/min(abs(x) for x in dx)
        rows.append(dict(observable=key,small_step_derivative=ga,large_step_derivative=gb,
            absolute_step_difference=abs(ga-gb),tolerance=tolerance,passed=abs(ga-gb)<=tolerance))
    gates=[all(c[key]['joint_gate_passed'] for key in required) for c in [ar,br]]
    passed=all(gates) and all(r['passed'] for r in rows)
    result=dict(frozen_target_scope_passed=passed,required_observables=sorted(required),
        both_scoped_pair_gates=gates,original_four_observable_joint_gates=[c['joint_gate_passed'] for c in certificates],
        diagnostic_observables=sorted(set(ar)-required),diagnostics_certified=False,
        two_step_consistency=rows,protocol=protocol,plan_sha256=launch['plan_sha256'],
        checker_sha256=sha(checker),script_sha256=sha(__file__),scope=protocol['scope'])
    (args.record/'acceptance.json').write_text(json.dumps(result,indent=2))
    status('completed_two_step_scoped_fd',frozen_target_scope_passed=passed)


if __name__=='__main__':
    main()
