"""Compare full QE elastic references along one cutoff or k-mesh axis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(coarse,fine,axis):
    paths=[Path(coarse),Path(fine)]
    metas=[json.loads((p/'metadata.json').read_text()) for p in paths]
    data=[json.loads((p/'summary.json').read_text()) for p in paths]
    args=[m['arguments'] for m in metas]
    if axis not in ['cutoff','k'] or args[1][axis]<=args[0][axis]:
        raise ValueError('Expected a refined cutoff or k mesh')
    for key in ['executable_sha256','pseudopotentials']:
        if metas[0][key]!=metas[1][key]:raise ValueError('Changed '+key)
    if any(a.get('mode')!='elastic' for a in args):raise ValueError('Requires elastic references')
    # bulk_strain_step was added later and is unused in elastic mode. Older
    # elastic metadata omit it; that schema change is not a physical change.
    settings=[{k:v for k,v in a.items() if k not in ['tag','reuse_from','a0','bulk_strain_step',axis]} for a in args]
    if settings[0]!=settings[1]:raise ValueError('Changed more than one numerical setting')
    material=json.loads(Path(__file__).with_name('materials.json').read_text())[args[0]['material']]
    strains=json.loads(args[0]['strain_steps'])
    branches=['D']+(['C44'] if material['C44'] is not None else [])
    keys=[f'{branch}_eps{h}' for h in strains for branch in branches]
    for d in data:
        if d['material']!=args[0]['material'] or any(k not in d for k in keys):
            raise ValueError('Incomplete or mismatched material results')
        if not np.isfinite([d['a0_A'],d['pressure_GPa'],*[d[k] for k in keys]]).all():
            raise ValueError('Nonfinite observables')
        if d['a0_A']<=0 or abs(d['pressure_GPa'])/(3*material['Bguess'])>=1e-5:
            raise ValueError('Need a converged equilibrium')
        for p in d['points']:
            if p.get('precision_source')!='QE XML in explicit Hartree atomic units':
                raise ValueError('Requires full precision XML')
            if not np.isfinite([p['energy_Ha'],p['pressure_GPa']]).all():
                raise ValueError('Nonfinite point')
            if p['label'].startswith('shear_'):
                f=np.asarray(p['forces'])
                if not np.isfinite(f).all() or np.abs(f).max()>=1e-6:
                    raise ValueError('Unrelaxed shear point')
    rows=[]
    for key in ['a0_A',*keys]:
        x,y=data[0][key],data[1][key]
        error=abs(y/x-1) if key=='a0_A' else abs(y-x)
        budget=1e-4 if key=='a0_A' else .1
        rows.append(dict(observable=key,coarse=x,fine=y,difference=y-x,error=error,
                         budget=budget,passed=error<budget))
    return dict(material=data[0]['material'],axis=axis,settings=[a[axis] for a in args],rows=rows,
                passed=all(r['passed'] for r in rows),
                sources={str(p):{n:hashlib.sha256((p/n).read_bytes()).hexdigest()
                         for n in ['metadata.json','summary.json']} for p in paths},
                scope='QE elastic comparison on this numerical axis and these strain amplitudes only; '
                      'no B, other-axis or D3FT certificate. Budgets are numerical, not statistical.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('coarse',type=Path);ap.add_argument('fine',type=Path)
    ap.add_argument('--axis',choices=['cutoff','k'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=compare(a.coarse,a.fine,a.axis)
    result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
