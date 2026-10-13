"""Check a real re-equilibrated parameter pair against the full Jacobian."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from equilibrium_jacobian import derive
from fd_acceptance import component, aggregate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('base',type=Path); ap.add_argument('minus',type=Path); ap.add_argument('plus',type=Path)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--rtol',type=float,default=.02)
    ap.add_argument('--agreement-only', action='store_true',
                    help='Diagnostic exit gate only; does not certify full production FD resolution')
    args=ap.parse_args()
    runs=[json.loads((p/'summary.json').read_text()) for p in [args.base,args.minus,args.plus]]
    metas=[json.loads((p/'metadata.json').read_text()) for p in [args.base,args.minus,args.plus]]
    base,minus,plus=runs
    vectors=np.array([r['parameter_vector'] for r in runs])
    np.testing.assert_allclose(vectors[0],(vectors[1]+vectors[2])/2,rtol=0,atol=1e-14)
    delta=vectors[2]-vectors[1]
    if not np.linalg.norm(delta)>0: raise ValueError('Zero perturbation')
    refmask=np.load(args.base/'mask.npy')
    for path,meta,result in zip([args.minus,args.plus],metas[1:],runs[1:]):
        if not np.array_equal(refmask,np.load(path/'mask.npy')): raise ValueError('Miller basis changed')
        for key in ['grid_sizes','cutoff_energy','k_grid_sizes','empty_bands']:
            if meta['config'][key]!=metas[0]['config'][key]: raise ValueError(f'{key} changed')
        if meta['arguments'].get('strain_step',.01)!=metas[0]['arguments'].get('strain_step',.01):
            raise ValueError('Strain estimator changed')
        if result['material']!=base['material']: raise ValueError('Material changed')
        if meta['arguments'].get('asr_force_floor',0.)!=metas[0]['arguments'].get('asr_force_floor',0.):
            raise ValueError('Force acceptance budget changed')
        if meta['pseudopotentials']!=metas[0]['pseudopotentials']: raise ValueError('Pseudopotentials changed')
        if meta['config']['d3ft'].get('xc_grid_factor',1)!=metas[0]['config']['d3ft'].get('xc_grid_factor',1):
            raise ValueError('XC quadrature changed')
        if abs(result['equilibrium_pressure_GPa'])/(3*result['B_GPa'])>1e-7:
            raise ValueError('Parameter FD requires a tighter pressure root')
        derive(path)  # validates all point residuals and internal forces
    jac=derive(args.base)
    gradients={'a0_A':np.array(jac['dlogV_equilibrium_dp'])*base['a0_A']/3,
               'B_GPa':jac['dB_equilibrium_dp'],
               'C11_minus_C12_GPa':jac.get('dD_equilibrium_dp'),
               'C44_GPa':jac.get('dC44_equilibrium_dp')}
    rows=[]
    for key,g in gradients.items():
        if g is None or key not in base: continue
        predicted=float(np.array(g)@delta); actual=plus[key]-minus[key]
        rows.append(component(key,predicted,actual,base['a0_A'],args.rtol))
    gates=aggregate(rows)
    result=dict(base=str(args.base),minus=str(args.minus),plus=str(args.plus),parameter_difference=delta.tolist(),
                rtol=args.rtol,rows=rows,**gates,
                acceptance_mode='agreement_only_diagnostic' if args.agreement_only else 'four_observable_joint',
                passed=gates['agreement_passed'] if args.agreement_only else gates['joint_gate_passed'])
    sources=[directory/name for directory in (args.base,args.minus,args.plus)
             for name in ('summary.json','metadata.json','points.json','mask.npy')]
    sources += [Path(__file__), Path(__file__).with_name('fd_acceptance.py'),
                Path(__file__).with_name('equilibrium_jacobian.py')]
    result['input_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}
    with args.out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
