"""Gate, execute and audit two-step reoptimized electronic geometry FD pilots.

Probe one N force component, longitudinal stress and engineering shear stress.
This is a directional check, not certification of all components or a relaxation.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--record', required=True, type=Path)
    args=ap.parse_args()
    plan=json.loads(args.plan.read_text())
    args.record.mkdir(parents=True,exist_ok=False)
    (args.record/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def status(state, **kw):
        obj=dict(state=state,time=datetime.datetime.now().isoformat(),**kw)
        tmp=args.record/'status.tmp'; tmp.write_text(json.dumps(obj,indent=2)+'\n')
        tmp.replace(args.record/'status.json'); print(json.dumps(obj),flush=True)
    parent=Path(plan['parent']); protocol=plan['protocol']
    dep=Path(plan['after_record']) if plan.get('after_record') else None
    status('waiting_for_parent' if dep else 'checking_completed_parent')
    while dep is not None:
        if (dep/'status.json').exists():
            state=json.loads((dep/'status.json').read_text())
            if state.get('state')=='sequence_exited':
                rows=state.get('results',[])
                if not rows or any(x['returncode'] for x in rows):
                    status('dependency_failed',dependency=state); return
                break
        try:
            import os
            os.kill(plan['after_pid'],0)
        except ProcessLookupError:
            status('dependency_controller_missing'); return
        time.sleep(15)
    p=json.loads((parent/'point.json').read_text())
    if p['residual']>protocol['electronic_residual_max'] or p['acoustic_sum_rule_Ha_Bohr']>protocol['asr_absolute_max_Ha_Bohr']:
        status('scientific_gate_failed',residual=p['residual'],asr=p['acoustic_sum_rule_Ha_Bohr'])
        return
    # The protocol was recorded before these FD states are computed.
    (args.record/'parent_evidence.json').write_text(json.dumps({
        str(parent/f):sha(parent/f) for f in ['point.json','metadata.json','mask.npy','state.pkl']
    },indent=2)+'\n')
    status('running_reoptimized_fd')
    workers=[]
    for child in plan['children']:
        cp=Path(child['plan']); cp.write_text(json.dumps(child['content'],indent=2)+'\n')
        cmd=[sys.executable,'-u',plan['sequence_driver'],str(cp),'--record',child['record']]
        log=(args.record/(child['name']+'.log')).open('x')
        result=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        workers.append((child,result,log))
    while any(proc.poll() is None for _,proc,_ in workers):
        time.sleep(10)
    for child, result, log in workers:
        log.close()
        # run_gpu_sequence records worker failures but itself exits zero.
        st=json.loads((Path(child['record'])/'status.json').read_text())
        if result.returncode or st.get('state')!='sequence_exited' or any(x['returncode'] for x in st.get('results',[])):
            status('worker_failed',child=child['name'],details=st); return
    records=[]
    for case in plan['cases']:
        plus,minus=Path(case['plus']),Path(case['minus'])
        a,b=[json.loads((x/'point.json').read_text()) for x in [plus,minus]]
        for path,data in [(plus,a),(minus,b)]:
            if data['residual']>protocol['electronic_residual_max']:
                raise ValueError('Unconverged FD point')
            if not np.array_equal(np.load(parent/'mask.npy'),np.load(path/'mask.npy')):
                raise ValueError('Miller mask changed in geometry FD')
        if case['quantity']=='force':
            delta=np.asarray(a['positions_bohr'])-np.asarray(b['positions_bohr'])
            expected=np.zeros_like(delta)
            expected[case['atom'],case['axis']]=2*case['h']
            if not np.allclose(delta,expected,rtol=0,atol=1e-10):
                raise ValueError('Actual displacement differs from prescribed Bohr step')
        else:
            for data, sign in [(a,1),(b,-1)]:
                eps=np.zeros((3,3))
                if case['row']==case['col']:
                    eps[case['row'],case['col']]=sign*case['h']
                else:
                    eps[case['row'],case['col']]=sign*case['h']/2
                    eps[case['col'],case['row']]=sign*case['h']/2
                expected=np.asarray(p['cell_bohr'])@(np.eye(3)+eps).T
                if not np.allclose(data['cell_bohr'],expected,rtol=0,atol=1e-10):
                    raise ValueError('Actual affine cell differs from prescribed strain')
        derivative=(a['energy_Ha']-b['energy_Ha'])/(2*case['h'])
        if case['quantity']=='force':
            derivative=-derivative
            reference=p['forces_Ha_Bohr'][case['atom']][case['axis']]
            tolerance=protocol['force_fd_absolute_max_Ha_Bohr']
        else:
            derivative/=p['volume_bohr3']
            reference=p['stress_Ha_Bohr3'][case['row']][case['col']]
            tolerance=protocol['stress_fd_absolute_max_GPa']/29421.015697
        records.append(dict(quantity=case['name'],h=case['h'],AD=reference,FD=derivative,
            absolute_error=abs(derivative-reference),tolerance=tolerance,
            energy_signal_Ha=abs(a['energy_Ha']-b['energy_Ha']),
            passes=bool(abs(derivative-reference)<=tolerance and
                        abs(a['energy_Ha']-b['energy_Ha'])>=protocol['energy_signal_min_Ha']),
            plus=str(plus),minus=str(minus),plus_sha256=sha(plus/'point.json'),minus_sha256=sha(minus/'point.json')))
    for row in records:
        counterparts=[x for x in records if x['quantity']==row['quantity'] and x['h']!=row['h']]
        row['step_difference']=abs(row['FD']-counterparts[0]['FD'])
        row['passes_two_steps']=bool(row['passes'] and row['step_difference']<=row['tolerance'])
    with (args.record/'force_stress_fd.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    passed=all(x['passes_two_steps'] for x in records)
    (args.record/'acceptance.json').write_text(json.dumps(dict(passed=passed,records=records,
        protocol=protocol,plan_sha256=sha(args.plan),script_sha256=sha(Path(__file__)),
        scope=protocol['scope']),indent=2)+'\n')
    status('completed_directional_fd',passed=passed)


if __name__=='__main__':
    main()
