"""Check consistency of two resolved, collinear central-FD step sizes.

This checks stability across the two steps, not an asymptotic convergence order.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from fd_acceptance import OBSERVABLES


def compare_steps(small, large, rtol=.02):
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError('Invalid step comparison tolerance')
    if Path(small['base']).resolve() != Path(large['base']).resolve():
        raise ValueError('FD bases differ')
    for certificate in (small, large):
        if not (certificate.get('joint_gate_passed') is True
                and certificate.get('acceptance_mode') == 'four_observable_joint'):
            raise ValueError('Requires resolved four-observable certificates')
    a, b = small['parameter_difference'], large['parameter_difference']
    if len(a) != 7 or len(b) != 7 or not all(math.isfinite(x) for x in a+b):
        raise ValueError('Invalid parameter difference')
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    if not 0 < na < nb:
        raise ValueError('Expected nonzero small then large steps')
    if any(abs(x/na-y/nb)>1e-10 for x,y in zip(a,b)):
        raise ValueError('FD directions differ')
    def index(certificate):
        rows = certificate['rows']
        if len(rows)!=4 or {r['observable'] for r in rows} != OBSERVABLES:
            raise ValueError('Incomplete or duplicate observable rows')
        return {r['observable']:r for r in rows}
    left, right = index(small), index(large)
    rows=[]
    for name in sorted(OBSERVABLES):
        x, y = left[name]['actual_difference']/na, right[name]['actual_difference']/nb
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite FD derivative')
        error=abs(x-y)
        floor=(1e-7 if name=='a0_A' else 1e-3)/na
        rows.append(dict(observable=name, small_step_derivative=x, large_step_derivative=y,
                         absolute_difference=error,
                         passed=error <= rtol*max(abs(x),abs(y))+floor))
    return dict(step_ratio=nb/na, rtol=rtol, rows=rows,
                passed=all(r['passed'] for r in rows),
                scope='Two-step central-FD stability, not proof of asymptotic order or occupation validity')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('small',type=Path);ap.add_argument('large',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--rtol',type=float,default=.02)
    args=ap.parse_args()
    result=compare_steps(json.loads(args.small.read_text()),json.loads(args.large.read_text()),args.rtol)
    result['input_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.small,args.large,Path(__file__))}
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
