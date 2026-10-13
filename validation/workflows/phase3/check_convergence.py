"""Compare stationary, common-geometry samples; this is not an elastic certificate."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(paths, axis, bulk):
    runs=[]
    for path in paths:
        path=Path(path)
        folder=path.parent if path.is_file() else path
        meta=json.loads((folder/'metadata.json').read_text())
        point=json.loads((path if path.is_file() else path/'summary.json').read_text())
        if (not path.is_file() and meta['arguments']['mode']!='sample') or point['residual']>1e-6:
            raise ValueError(f'{path}: requires a stationary sample')
        runs.append((path,meta,point))
    if len(runs)<2: raise ValueError('At least two settings are required')
    first=runs[0]
    for path,meta,point in runs[1:]:
        for key in ['parameter_vector','pseudopotentials']:
            if meta[key]!=first[1][key]: raise ValueError(f'{path}: different {key}')
        for key in ['a0_A','strain','displacement_Bohr']:
            if not np.allclose(point[key],first[2][key],rtol=0,atol=1e-12):
                raise ValueError(f'{path}: different geometry {key}')
        for key in ['cutoff_energy','grid_sizes','k_grid_sizes','empty_bands','xc']:
            if key!=axis and meta['config'][key]!=first[1]['config'][key]:
                raise ValueError(f'{path}: uncontrolled {key}')
        if meta['config']['d3ft'].get('xc_grid_factor',1)!=first[1]['config']['d3ft'].get('xc_grid_factor',1):
            raise ValueError(f'{path}: uncontrolled XC quadrature')
    runs.sort(key=lambda r:r[1]['config'][axis])
    settings=[r[1]['config'][axis] for r in runs]
    if len(set(settings))!=len(settings): raise ValueError('Duplicate settings')
    reference=runs[-1]
    materials=json.loads((Path(__file__).with_name('materials.json')).read_text())
    natoms=len(materials[first[1]['arguments']['material']]['symbols'])
    rows=[]
    for path,meta,p in runs[:-1]:
        de=abs(p['energy_Ha']-reference[2]['energy_Ha'])*1000/natoms
        da=abs(p['pressure_GPa']-reference[2]['pressure_GPa'])/(3*bulk)
        rows.append(dict(source=str(path),setting=meta['config'][axis],energy_error_mHa_atom=de,
                         lattice_error_from_pressure=da,passes_energy_pressure=bool(de<1 and da<1e-4)))
    energies=[r[2]['energy_Ha'] for r in runs]
    return dict(axis=axis,reference=str(reference[0]),bulk_estimate_GPa=bulk,
                scope='Common-geometry energy/pressure only; no elastic or parameter-transfer certificate',
                energies_monotone_decreasing=bool(np.all(np.diff(energies)<=1e-10)),candidates=rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',type=Path)
    ap.add_argument('--axis',required=True,choices=['cutoff_energy','grid_sizes','k_grid_sizes'])
    ap.add_argument('--bulk',required=True,type=float)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    if args.bulk<=0: raise ValueError('bulk must be positive')
    result=compare(args.runs,args.axis,args.bulk)
    with args.out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
