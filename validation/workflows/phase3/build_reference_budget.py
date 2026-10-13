"""Separate archived fitting weights from reported experimental uncertainties."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
materials=json.loads((HERE/'materials.json').read_text())
record=json.loads((HERE/'reference_uncertainty.json').read_text())
for source in [HERE/'reference_uncertainty.json',HERE/'materials.json',ROOT/'reference_exp.md',Path(__file__)]:
    (a.out/source.name).write_bytes(source.read_bytes())
rows=[]
for name,m in materials.items():
    if name not in record['materials']:continue
    r=record['materials'][name]
    targets=[('a0_A',m['a_ref'],r['a_pm_A'],'reported_pm')]
    if m['C11'] is not None:
        u,v=r['C11_pm_GPa'],r['C12_pm_GPa']
        bounds=[None,None] if u is None or v is None else [(u+2*v)/3,u+v]
        targets += [('B_GPa',(m['C11']+2*m['C12'])/3,bounds[0],'propagated_interval_bound'),
                    ('C11_minus_C12_GPa',m['C11']-m['C12'],bounds[1],'propagated_interval_bound')]
    if m['C44'] is not None:targets.append(('C44_GPa',m['C44'],r['C44_pm_GPa'],'reported_pm'))
    for key,value,uncertainty,kind in targets:
        rows.append(dict(material=name,observable=key,reference=value,role=m['role'],
            archived_relative_fit_weight=m['sigma_rel'],reported_pm_or_interval_bound=uncertainty,
            uncertainty_kind=kind if uncertainty is not None else 'not_reported_in_local_record',
            one_sigma_established=False,C11_C12_covariance_known=False,
            static_lattice_to_experiment_correction_applied=False,
            temperature_K=r.get('a_temperature_K' if key=='a0_A' else 'elastic_temperature_K'),
            temperature_note=r.get('temperature_note',''),source='reference_exp.md',source_line=r['source_line']))
with (a.out/'observations.csv').open('x') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
meta=dict(source_sha256=hashlib.sha256((ROOT/'reference_exp.md').read_bytes()).hexdigest(),
    transcription_sha256=hashlib.sha256((HERE/'reference_uncertainty.json').read_bytes()).hexdigest(),
    rows=rows,scope=record['verification_scope'],
    propagation='B bound=(u11+2*u12)/3; D bound=u11+u12, conditional on both reported input intervals holding; not standard deviations',
    limitations=['B and D share C11/C12 and are correlated; do not count them as independent statistical observations',
        'Finite-temperature and zero-point effects are uncorrected, unquantified systematic terms',
        'Archived material weights are algorithmic weights, not measured confidence intervals',
        'The tunable model changes both pseudopotential and XC terms'])
(a.out/'budget.json').write_text(json.dumps(meta,indent=2))
print(f'{len(rows)} reference observations; missing uncertainties remain null')
