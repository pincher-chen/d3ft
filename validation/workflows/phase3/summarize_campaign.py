"""Report actual baseline/final errors separately for training and holdouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('campaign',type=Path);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
materials=json.loads((HERE/'materials.json').read_text())
summary=json.loads((a.campaign/'summary.json').read_text())
before=json.loads((a.campaign/'initial_manifest.json').read_text())
after=json.loads((a.campaign/'final_manifest.json').read_text())
held=set(summary['holdout_materials'])
trained={r['material'] for r in summary['training_rows']}
rows=[]
for name in sorted(trained|held):
    m=materials[name]
    start=json.loads((Path(before[name])/'summary.json').read_text())
    end=json.loads((Path(after[name])/'summary.json').read_text())
    if not np.allclose(end['parameter_vector'],summary['final_parameter_vector'],rtol=0,atol=1e-14):
        raise ValueError('Final manifest contains an unevaluated parameter vector')
    targets=[('a0_A',m['a_ref'])]
    if m['C11'] is not None:
        targets += [('B_GPa',(m['C11']+2*m['C12'])/3),('C11_minus_C12_GPa',m['C11']-m['C12'])]
    if m['C44'] is not None:targets.append(('C44_GPa',m['C44']))
    for key,reference in targets:
        values=np.array([start[key],end[key]],float)
        if not np.isfinite(values).all():raise ValueError('Nonfinite result')
        errors=(values/reference-1)*100
        rows.append(dict(material=name,role='holdout' if name in held else 'train',
            observable=key,reference=reference,baseline=values[0],final=values[1],
            baseline_error_percent=errors[0],final_error_percent=errors[1],
            absolute_error_improved=bool(abs(errors[1])<abs(errors[0])),
            baseline_source=before[name],final_source=after[name]))
with (a.out/'observations.csv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups={}
for label,selected in [('train',[r for r in rows if r['role']=='train']),
                       ('holdout',[r for r in rows if r['role']=='holdout'])]+[
                       ('material_'+m,[r for r in rows if r['material']==m]) for m in sorted(trained|held)]:
    if not selected:continue
    initial=np.array([r['baseline_error_percent'] for r in selected])
    final=np.array([r['final_error_percent'] for r in selected])
    groups[label]=dict(rows=len(selected),baseline_RMS_percent=float(np.sqrt(np.mean(initial**2))),
        final_RMS_percent=float(np.sqrt(np.mean(final**2))),baseline_MAE_percent=float(np.mean(abs(initial))),
        final_MAE_percent=float(np.mean(abs(final))),improved_rows=sum(r['absolute_error_improved'] for r in selected))
report=dict(groups=groups,stop_reason=summary['stop_reason'],accepted_steps=summary['accepted_steps'],
    spectra_checked=summary['spectra_checked'],holdout_spectra=summary['holdout_spectra'],
    final_parameter_vector=summary['final_parameter_vector'],
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [a.campaign/'summary.json',a.campaign/'initial_manifest.json',a.campaign/'final_manifest.json']},
    scope='Actual recomputed observables; unweighted relative-error metrics, not confidence intervals',
    limitations=['B and D are correlated derived quantities',
        'Room-temperature reference data have no applied zero-point/thermal correction',
        'Numerical convergence certificates are separate from this fit report'])
(a.out/'metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(groups,indent=2))
