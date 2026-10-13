"""Check C44 quadrature using a baseline's exact equilibrium/relaxed geometries."""
import argparse
import json
from pathlib import Path
import numpy as np
from equilibrium_jacobian import derive
from force_acceptance import check_force_sum

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('baseline',type=Path)
ap.add_argument('--center',required=True,type=Path)
ap.add_argument('--minus',required=True,type=Path)
ap.add_argument('--plus',required=True,type=Path)
ap.add_argument('--elastic-budget-gpa',type=float,default=.1)
ap.add_argument('--force-budget',type=float,default=1e-7)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args()
if a.elastic_budget_gpa<=0 or a.force_budget<=0:raise ValueError('Positive error budgets required')
derive(a.baseline)
b=json.loads((a.baseline/'summary.json').read_text())
bm=json.loads((a.baseline/'metadata.json').read_text())
points=json.loads((a.baseline/'points.json').read_text())
center=[p for p in points if p['label'].startswith('eq_')][-1]
h=bm['arguments'].get('strain_step',.01)
old=[center]+[[p for p in points if p['label'].startswith('shear_') and abs(p['strain'][1][2]-sign*h)<1e-12][-1] for sign in [-1,1]]
mask=np.load(a.baseline/'mask.npy')
rows=[];fine=[];factors=[]
for source,reference in zip([a.center,a.minus,a.plus],old):
    d=json.loads((source/'summary.json').read_text());m=json.loads((source/'metadata.json').read_text())
    if m['parameter_vector']!=bm['parameter_vector'] or m['pseudopotentials']!=bm['pseudopotentials']:
        raise ValueError('Changed parameters or pseudopotentials')
    for key in ['grid_sizes','cutoff_energy','k_grid_sizes','empty_bands','xc','spin']:
        if m['config'][key]!=bm['config'][key]:raise ValueError(f'Uncontrolled change in {key}')
    if not np.array_equal(np.load(source/'mask.npy'),mask):raise ValueError('Changed Miller basis')
    for key in ['a0_A','strain','displacement_Bohr']:
        np.testing.assert_allclose(d[key],reference[key],rtol=0,atol=1e-12)
    if bool(d.get('balanced_displacement',False))!=bool(reference.get('balanced_displacement',False)):
        raise ValueError('Changed displacement convention')
    if not np.isfinite(d['residual']) or d['residual']>1e-6:raise ValueError('Nonstationary fine state')
    check=check_force_sum(d['forces'],floor=bm['arguments'].get('asr_force_floor',0.))
    if not check['passes_ionic_force']:raise ValueError('Fine-grid geometry needs further relaxation')
    factor=m['config']['d3ft'].get('xc_grid_factor',1);factors.append(factor)
    fine.append(d)
    rows.append(dict(source=str(source),label=reference['label'],fine_force_check=check,
        energy_difference_Ha=d['energy_Ha']-reference['energy_Ha'],
        pressure_difference_GPa=d['pressure_GPa']-reference['pressure_GPa'],
        max_force_difference=float(np.max(np.abs(np.array(d['forces'])-reference['forces'])))))
base_factor=bm['config']['d3ft'].get('xc_grid_factor',1)
if len(set(factors))!=1 or factors[0]<=base_factor:raise ValueError('Need one consistently finer XC grid')
if abs(fine[0]['pressure_GPa'])/(3*b['B_GPa'])>=1e-5:raise ValueError('Fine grid needs re-equilibration')
value=(fine[1]['energy_Ha']+fine[2]['energy_Ha']-2*fine[0]['energy_Ha'])*29421.02648438959/(4*center['volume_bohr3']*h*h)
difference=value-b['C44_GPa']
passed=abs(difference)<a.elastic_budget_gpa and max(p['max_force_difference'] for p in rows)<a.force_budget
result=dict(baseline=str(a.baseline),base_xc_factor=base_factor,fine_xc_factor=factors[0],
            C44_base_GPa=b['C44_GPa'],C44_fine_GPa=value,difference_GPa=difference,
            elastic_budget_GPa=a.elastic_budget_gpa,force_budget_Ha_Bohr=a.force_budget,passed=passed,rows=rows,
            scope='C44 at these force-stationary geometries and near-zero pressure only; no B/D/cutoff/k certificate')
with a.out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not passed:raise SystemExit(1)
