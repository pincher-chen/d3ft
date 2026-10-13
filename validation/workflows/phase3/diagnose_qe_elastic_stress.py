"""Compare energy curvature with stress work along the same QE strain path.

QE reports compression-positive Cauchy stress in Hartree atomic units.
dE/dd = -V tr(stress F' F^-1). For tetragonal F=diag(1+d,1+d,(1+d)^-2),
V=V0; for yz shear det(F)=1-d^2. The formulas below include these geometric
factors. The central stress derivative has its own O(h^2) truncation error.
This is a diagnostic, not a replacement for energy-curvature convergence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from qe_precision import GPA, read_state


def diagnose(source):
    summary=json.loads((source/'summary.json').read_text())
    metadata=json.loads((source/'metadata.json').read_text())
    amplitudes=json.loads(metadata['arguments']['strain_steps'])
    rows=[]
    hashes={}
    for h in amplitudes:
        for branch,observable,denominator in [('tet','D',6),('shear','C44',4)]:
            key=f'{observable}_eps{h}'
            if key not in summary:
                if branch=='tet':raise ValueError('Missing tetragonal amplitude')
                continue
            points={round(float(p['label'].split('_')[1]),6):p for p in summary['points']
                    if p['label'].startswith(branch+'_')}
            work_derivatives=[]
            for strain in [-h,h]:
                record=points[round(strain,6)]
                values,xml=read_state(source/record['label'])
                hashes[record['label']]=values['xml_sha256']
                stress=np.asarray(values['stress_QE_Ry_bohr3'])/2
                if branch=='tet':
                    derivative=-(stress[0,0]+stress[1,1]-2*stress[2,2])/(1+strain)
                else:
                    derivative=-(stress[1,2]+stress[2,1]-strain*(stress[1,1]+stress[2,2]))
                work_derivatives.append(derivative)
            curvature=(work_derivatives[1]-work_derivatives[0])*GPA/(2*h*denominator)
            rows.append(dict(observable=observable,strain=h,energy_GPa=summary[key],
                             stress_GPa=float(curvature),energy_minus_stress_GPa=summary[key]-float(curvature)))
    return dict(material=summary['material'],source=str(source),rows=rows,point_xml_sha256=hashes,
                summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
                scope='Diagnostic of energy versus stress curvature along identical strain paths; '
                      'each stencil has its own finite-strain error. No convergence certificate.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=diagnose(args.source)
    result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
