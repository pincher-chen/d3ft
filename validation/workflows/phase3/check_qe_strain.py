"""Certify two strain amplitudes using full-precision QE XML observables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('source',type=Path);ap.add_argument('--coarse',type=float,default=.005)
ap.add_argument('--fine',type=float,default=.0025);ap.add_argument('--budget-gpa',type=float,default=.1)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args()
if not 0<a.fine<a.coarse or a.budget_gpa<=0:raise ValueError('Invalid amplitudes or budget')
d=json.loads((a.source/'summary.json').read_text())
m=json.loads(Path(__file__).with_name('materials.json').read_text())[d['material']]
if any(p.get('precision_source')!='QE XML in explicit Hartree atomic units' for p in d['points']):
    raise ValueError('Rounded stdout energies cannot certify this small-strain comparison')
rows=[]
for branch in ['D']+(['C44'] if m['C44'] is not None else []):
    coarse=d[f'{branch}_eps{a.coarse:g}'];fine=d[f'{branch}_eps{a.fine:g}']
    if not np.isfinite([coarse,fine]).all():raise ValueError('Nonfinite elastic constants')
    rows.append(dict(observable=branch,coarse_GPa=coarse,fine_GPa=fine,
        difference_GPa=fine-coarse,passed=abs(fine-coarse)<a.budget_gpa))
result=dict(material=d['material'],source=str(a.source),coarse_strain=a.coarse,fine_strain=a.fine,
    budget_GPa=a.budget_gpa,rows=rows,passed=all(r['passed'] for r in rows),
    source_summary_sha256=hashlib.sha256((a.source/'summary.json').read_bytes()).hexdigest(),
    scope='QE strain-amplitude comparison at these settings; no cutoff/k or D3FT certificate')
with a.out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(1)
