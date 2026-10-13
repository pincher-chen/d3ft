"""Compare complete, independently equilibrated runs along one numerical axis.

Budgets are numerical resolution goals, not experimental error bars. A pass
certifies only the chosen pair and parameter vector. Raw runs are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from equilibrium_jacobian import derive


def miller_set(mask):
    if mask.dtype!=bool or mask.ndim!=3:
        raise ValueError('Expected a Boolean three-dimensional FFT mask')
    axes=[np.rint(np.fft.fftfreq(n)*n).astype(int) for n in mask.shape]
    indices=np.nonzero(mask)
    return set(zip(*(axis[index].tolist() for axis,index in zip(axes,indices))))


def compare(coarse, fine, axis, lattice_budget=1e-4, elastic_budget=.1,
            fixed_physical_xc_grid=False):
    if lattice_budget <= 0 or elastic_budget <= 0:
        raise ValueError('Positive budgets required')
    sources=[Path(coarse),Path(fine)]
    summaries=[json.loads((p/'summary.json').read_text()) for p in sources]
    metas=[json.loads((p/'metadata.json').read_text()) for p in sources]
    materials=json.loads(Path(__file__).with_name('materials.json').read_text())
    for summary,meta in zip(summaries,metas):
        material=materials[summary['material']]
        required=['a0_A','B_GPa']
        if material['C11'] is not None:required.append('C11_minus_C12_GPa')
        if material['C44'] is not None:required.append('C44_GPa')
        if meta['arguments'].get('mode')!='evaluate' or any(k not in summary for k in required):
            raise ValueError('Incomplete material evaluation')
    for p in sources:
        derive(p)  # pressure, electronic stationarity, relaxed raw-force checks
        points=json.loads((p/'points.json').read_text())
        if not all(np.isfinite(x['energy_Ha']) and np.isfinite(x['pressure_GPa'])
                   and np.isfinite(x['residual']) for x in points):
            raise ValueError('Nonfinite point data')
    for key in ['material','parameter_vector']:
        if summaries[0][key]!=summaries[1][key]:
            raise ValueError(f'Changed {key}')
    if metas[0]['pseudopotentials']!=metas[1]['pseudopotentials']:
        raise ValueError('Changed pseudopotentials')
    axes={'cutoff':'cutoff_energy','grid':'grid_sizes','k':'k_grid_sizes'}
    settings=[]
    for meta in metas:
        cfg=meta['config']; args=meta['arguments']
        settings.append({**{a:cfg[k] for a,k in axes.items()},
            'xc_grid':cfg.get('d3ft',{}).get('xc_grid_factor',1),
            'strain':args.get('strain_step',.01),
            'force_floor':args.get('asr_force_floor',0.),
            **{k:cfg[k] for k in ['xc','empty_bands','spin']}})
    if axis not in [*axes,'xc_grid','strain']:
        raise ValueError('Unknown numerical axis')
    if fixed_physical_xc_grid:
        if axis != 'grid':
            raise ValueError('A fixed physical XC grid bridge must compare coarse FFT grids')
        actual_xc_shapes = [tuple(n*int(s['xc_grid']) for n in np.load(p/'mask.npy').shape)
                             for p,s in zip(sources,settings)]
        if actual_xc_shapes[0] != actual_xc_shapes[1]:
            raise ValueError('Physical XC grid changed in a claimed fixed-XC coarse-grid bridge')
    for key in settings[0]:
        if fixed_physical_xc_grid and key == 'xc_grid':
            continue  # factor compensates coarse-grid size; physical shape checked above
        if key!=axis and settings[0][key]!=settings[1][key]:
            raise ValueError(f'Uncontrolled change in {key}')
    x,y=settings[0][axis],settings[1][axis]
    if not (y<x if axis=='strain' else y>x):
        raise ValueError('Second run must refine the selected axis')
    if axis in ['xc_grid','strain','k']:
        if not np.array_equal(*(np.load(p/'mask.npy') for p in sources)):
            raise ValueError('Changed Miller basis')
    if axis=='grid':
        if miller_set(np.load(sources[0]/'mask.npy'))!=miller_set(np.load(sources[1]/'mask.npy')):
            raise ValueError('Grid comparison changed physical Miller frequencies')
    rows=[]
    for key in ['a0_A','B_GPa','C11_minus_C12_GPa','C44_GPa']:
        if key not in summaries[0] and key not in summaries[1]:
            continue
        values=[s[key] for s in summaries]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite {key}')
        difference=values[1]-values[0]
        error=abs(difference/values[1]) if key=='a0_A' else abs(difference)
        budget=lattice_budget if key=='a0_A' else elastic_budget
        rows.append(dict(observable=key,coarse=values[0],fine=values[1],
                         difference=difference,error=error,budget=budget,passed=error<budget))
    return dict(material=summaries[0]['material'],axis=axis,settings=settings,
        fixed_physical_xc_grid=fixed_physical_xc_grid,
        physical_xc_grid_shapes=[tuple(n*int(s['xc_grid']) for n in np.load(p/'mask.npy').shape)
                                for p,s in zip(sources,settings)],
        parameter_vector=summaries[0]['parameter_vector'],rows=rows,
        passed=all(r['passed'] for r in rows),
        sources={str(p):{name:hashlib.sha256((p/name).read_bytes()).hexdigest()
            for name in ['metadata.json','summary.json','points.json','mask.npy']} for p in sources},
        scope='One numerical axis, these settings and parameters only; budgets are not statistical uncertainties')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('coarse',type=Path);ap.add_argument('fine',type=Path)
    ap.add_argument('--axis',choices=['cutoff','grid','k','xc_grid','strain'],required=True)
    ap.add_argument('--lattice-budget',type=float,default=1e-4)
    ap.add_argument('--elastic-budget-gpa',type=float,default=.1)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fixed-physical-xc-grid',action='store_true',
                    help='Coarse-grid bridge: XC factors may differ only when actual XC FFT shapes are identical')
    a=ap.parse_args()
    result=compare(a.coarse,a.fine,a.axis,a.lattice_budget,a.elastic_budget_gpa,
                   a.fixed_physical_xc_grid)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
