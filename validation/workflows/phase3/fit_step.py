"""Scaled LM proposal and actual-loss acceptance, with whole-material holdouts.

Inputs are a JSON mapping material -> completed worker output directory.
Experimental sigmas currently reproduce the archived relative weighting; they
are NOT independently measured, per-observable statistical uncertainties.
No candidate is accepted using only its linear prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from equilibrium_jacobian import derive

HERE=Path(__file__).resolve().parent
MATERIALS=json.loads((HERE/'materials.json').read_text())
SCALE=np.array([.1,.05,.05,.01,.005,.1,.02])

def input_fingerprints(manifest,holdouts):
    paths=json.loads(Path(manifest).read_text())
    selected={m for m,c in MATERIALS.items() if c['role'].startswith('train')}-set(holdouts)
    return {m:{name:hashlib.sha256((Path(paths[m])/name).read_bytes()).hexdigest()
               for name in ['metadata.json','summary.json','mask.npy']} for m in sorted(selected)}

def validate_trial(baseline,candidate,holdouts):
    paths=[json.loads(Path(p).read_text()) for p in [baseline,candidate]]
    selected={m for m,c in MATERIALS.items() if c['role'].startswith('train')}-set(holdouts)
    for mat in sorted(selected):
        dirs=[Path(p[mat]) for p in paths]
        metas=[json.loads((p/'metadata.json').read_text()) for p in dirs]
        if metas[0]['pseudopotentials']!=metas[1]['pseudopotentials']:
            raise ValueError(f'{mat}: trial changed pseudopotentials')
        for key in ['grid_sizes','cutoff_energy','k_grid_sizes','empty_bands','xc','spin']:
            if metas[0]['config'][key]!=metas[1]['config'][key]:
                raise ValueError(f'{mat}: trial changed {key}')
        if metas[0]['arguments'].get('strain_step',.01)!=metas[1]['arguments'].get('strain_step',.01):
            raise ValueError(f'{mat}: trial changed strain estimator')
        if metas[0]['arguments'].get('asr_force_floor',0.)!=metas[1]['arguments'].get('asr_force_floor',0.):
            raise ValueError(f'{mat}: trial changed force acceptance budget')
        if metas[0]['config'].get('d3ft',{}).get('xc_grid_factor',1)!=metas[1]['config'].get('d3ft',{}).get('xc_grid_factor',1):
            raise ValueError(f'{mat}: trial changed XC quadrature')
        if not np.array_equal(np.load(dirs[0]/'mask.npy'),np.load(dirs[1]/'mask.npy')):
            raise ValueError(f'{mat}: trial changed Miller basis; use --fixed-mask BASE/mask.npy')

def dataset(manifest,holdouts):
    paths=json.loads(Path(manifest).read_text())
    expected={m for m,c in MATERIALS.items() if c['role'].startswith('train')}-set(holdouts)
    missing=expected-set(paths)
    if missing: raise ValueError(f'Missing training materials: {sorted(missing)}')
    rows=[]; vector=None
    for mat in sorted(expected):
        work=Path(paths[mat]); d=json.loads((work/'summary.json').read_text()); jac=derive(work)
        if d['material']!=mat: raise ValueError('Material/path mismatch')
        v=np.array(d['parameter_vector'])
        if vector is None: vector=v
        if not np.array_equal(vector,v): raise ValueError('Training points use different parameters')
        m=MATERIALS[mat]
        observables=[('a0_A',m['a_ref'],np.array(jac['dlogV_equilibrium_dp'])*d['a0_A']/3)]
        if m['C11'] is not None:
            observables += [('B_GPa',(m['C11']+2*m['C12'])/3,jac['dB_equilibrium_dp']),
                            ('C11_minus_C12_GPa',m['C11']-m['C12'],jac['dD_equilibrium_dp'])]
        if m['C44'] is not None:
            observables += [('C44_GPa',m['C44'],jac['dC44_equilibrium_dp'])]
        for key,ref,grad in observables:
            rows.append(dict(material=mat,observable=key,residual=(d[key]-ref)/ref,
                             gradient=(np.array(grad)/ref).tolist(),sigma=m['sigma_rel'],reference=ref))
    r=np.array([x['residual']/x['sigma'] for x in rows])
    j=np.array([np.array(x['gradient'])/x['sigma'] for x in rows])
    return vector,rows,r,j

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('manifest'); ap.add_argument('--holdout',action='append',default=[])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--damping',type=float,default=1.)
    ap.add_argument('--radius',type=float,default=1.)
    ap.add_argument('--candidate',help='Completed candidate manifest for actual-loss acceptance')
    ap.add_argument('--proposal',type=Path,help='Proposal JSON corresponding to candidate')
    args=ap.parse_args()
    if set(args.holdout)-set(MATERIALS): raise ValueError('Unknown holdout material')
    v,rows,r,j=dataset(args.manifest,args.holdout)
    if args.candidate:
        if args.proposal is None: raise ValueError('--proposal required for acceptance')
        prop=json.loads(args.proposal.read_text())
        if sorted(prop['holdouts'])!=sorted(args.holdout): raise ValueError('Proposal holdouts changed')
        if prop['input_fingerprints']!=input_fingerprints(args.manifest,args.holdout):
            raise ValueError('Proposal baseline outputs changed')
        validate_trial(args.manifest,args.candidate,args.holdout)
        cv,crows,cr,cj=dataset(args.candidate,args.holdout)
        if not np.array_equal(v,prop['starting_vector']) or not np.allclose(cv,prop['candidate_vector'],rtol=0,atol=1e-14):
            raise ValueError('Proposal and actual evaluation parameter mismatch')
        identities=lambda rr:[(x['material'],x['observable'],x['sigma'],x['reference']) for x in rr]
        if identities(rows)!=identities(crows): raise ValueError('Training rows changed during trial')
        predicted=np.dot(r,r)-np.dot(r+j@(cv-v),r+j@(cv-v))
        actual=float(np.dot(r,r)-np.dot(cr,cr))
        rho=actual/predicted if predicted>0 else None
        result=dict(accepted=rho is not None and rho>.1 and actual>0,
                    loss_before=float(r@r),loss_after=float(cr@cr),actual_reduction=actual,
                    predicted_reduction=float(predicted),rho=rho,holdouts=args.holdout)
        result['baseline_fingerprints']=prop['input_fingerprints']
        result['candidate_fingerprints']=input_fingerprints(args.candidate,args.holdout)
    else:
        if args.damping<=0 or args.radius<=0: raise ValueError('Positive damping and radius required')
        scaled=j*SCALE
        u,s,vt=np.linalg.svd(scaled,full_matrices=False)
        z=-vt.T@((s/(s*s+args.damping))*(u.T@r))
        z*=min(1.,args.radius/max(np.linalg.norm(z),1e-30))
        # Positive PBE constants, moderate KB strengths. Clip before predicting.
        constants=np.array([.06672455060314922,.031090690869654894,.804,.2195149727645171])
        bounds=np.r_[np.full(3,.5),.5*constants]
        candidate=np.clip(v+SCALE*z,-bounds,bounds)
        pred=r+j@(candidate-v)
        result=dict(starting_vector=v.tolist(),candidate_vector=candidate.tolist(),
                    holdouts=args.holdout,training_rows=rows,parameter_scales=SCALE.tolist(),
                    singular_values=s.tolist(),damping=args.damping,radius=args.radius,
                    loss_before=float(r@r),predicted_loss=float(pred@pred),accepted=False,
                    status='proposal_only_requires_full_re_equilibration',
                    input_fingerprints=input_fingerprints(args.manifest,args.holdout),
                    weighting='archived relative material weights; not statistical confidence intervals')
    with args.out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='training_rows'},indent=2))

if __name__=='__main__': main()
