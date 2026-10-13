"""Reconstruct B' step-size diagnostics from completed fixed-basis volume points."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def second_derivatives(pressure, h):
    second = lambda step: (pressure[step]+pressure[-step]-2*pressure[0])/step**2
    d1,d2=second(h),second(2*h)
    return d1,d2,(4*d1-d2)/3


def audit(work):
    result=json.loads((work/'summary.json').read_text())
    points=json.loads((work/'points.json').read_text())
    args=json.loads((work/'metadata.json').read_text())['arguments']
    center=[p for p in points if p['label'].startswith('eq_')][-1]
    bulk=result['B_GPa'];h=args.get('strain_step',.01)
    iso={round(p['strain'][0][0],6):p for p in points if p['label'].startswith('iso_')}
    used=[center]+[iso[round(step,6)] for step in [-2*h,-h,h,2*h]]
    if not np.isfinite(bulk) or bulk<=0 or abs(center['pressure_GPa'])/(3*bulk)>=1e-5:
        raise ValueError('Need a stable equilibrium')
    if any(not np.isfinite(p['residual']) or p['residual']>1e-6 for p in used):
        raise ValueError('Need stationary volume points')
    pressure={0:center['pressure_GPa']}
    for step in [-2*h,-h,h,2*h]:pressure[step]=iso[round(step,6)]['pressure_GPa']
    derivatives=second_derivatives(pressure,h)
    values=[(d/(3*bulk)-1)/3 for d in derivatives]
    return dict(source=str(work),material=result['material'],h=h,Bprime_h=values[0],
                Bprime_2h=values[1],Bprime_Richardson=values[2],step_correction=values[2]-values[0],
                fingerprints={name:hashlib.sha256((work/name).read_bytes()).hexdigest()
                              for name in ['metadata.json','summary.json','points.json','mask.npy']})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',type=Path);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    out=dict(scope='Numerical Bprime diagnostic; no automatic temperature conversion or fitted target.',
             rows=[audit(p) for p in args.runs])
    with args.out.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
