"""Freeze v3 nonlinear calibration and grouped holdout proposals, not results.

Rebuild the dataset from raw worker output and require exact agreement with the
frozen v3 matrix. Holdout proposals use only the retained materials' rows.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from fit_step import dataset, input_fingerprints, SCALE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--jacobian',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); frozen=json.loads(a.jacobian.read_text())
    if frozen['version']!=3 or frozen['manifest_sha256']!=sha(a.manifest):
        raise ValueError('Wrong or modified frozen v3 manifest')
    v,rows,r,j=dataset(a.manifest,[])
    if not (np.array_equal(v,frozen['parameter_vector']) and
            np.array_equal(r,frozen['normalized_residual']) and
            np.array_equal(j,frozen['normalized_jacobian'])):
        raise ValueError('Current worker output differs from frozen v3 dataset')
    fingerprints=input_fingerprints(a.manifest,[])
    for material, files in fingerprints.items():
        if any(frozen['input_fingerprints'][material][k]!=h for k,h in files.items()):
            raise ValueError('Frozen material fingerprints changed')
    a.output.mkdir(parents=True,exist_ok=False)
    def put(name,data):
        with (a.output/name).open('x') as f:json.dump(data,f,indent=2)
    protocol=dict(frozen_at=datetime.datetime.now().isoformat(),
        manifest=str(a.manifest.resolve()),manifest_sha256=sha(a.manifest),
        jacobian=str(a.jacobian.resolve()),jacobian_sha256=sha(a.jacobian),
        parameter_names=frozen['parameter_names'],parameter_scales=SCALE.tolist(),
        folds={'all':[],'leave_diamond_out':['diamond'],'leave_BN_out':['BN']},
        initial_damping=1.,initial_radius=.04,
        initialization='same untrained baseline in each independent refit',
        weighting='archived relative material weights, not statistical variances',
        acceptance_required=['real full re-equilibration of every training material',
            'actual decrease and rho>0.1','candidate numerical budgets',
            'parameter derivative acceptance on the applicable frozen model',
            'fresh density-bound full-mesh occupied-state spectrum'],
        holdout_rule='Fit excludes all rows of the held material. Evaluate after stopping; never tune radius, bounds or stopping from its outcome.',
        physical_model='frozen v3 default local trapezoid; response Simpson pilots are separate and cannot be used to explain a fitting improvement',
        state='prepared_unexecuted_nonlinear_refits',
        script_sha256=sha(Path(__file__)))
    put('protocol.json',protocol)
    summaries=[]
    for fold,held in protocol['folds'].items():
        fv,fr,rr,jj=dataset(a.manifest,held)
        mask=np.array([row['material'] not in held for row in rows])
        if not (np.array_equal(jj,j[mask]) and np.array_equal(rr,r[mask])):
            raise ValueError('Grouped split does not match the retained frozen rows')
        scaled=jj*SCALE;u,s,vt=np.linalg.svd(scaled,full_matrices=False)
        z=-vt.T@((s/(s*s+1.))*(u.T@rr));z*=min(1.,.04/max(np.linalg.norm(z),1e-30))
        constants=np.array([.06672455060314922,.031090690869654894,.804,.2195149727645171])
        bounds=np.r_[np.full(3,.5),.5*constants]
        candidate=np.clip(fv+SCALE*z,-bounds,bounds)
        pred=rr+jj@(candidate-fv)
        prop=dict(starting_vector=fv.tolist(),candidate_vector=candidate.tolist(),
            holdouts=held,training_rows=fr,parameter_scales=SCALE.tolist(),
            singular_values=s.tolist(),damping=1.,radius=.04,
            loss_before=float(rr@rr),predicted_loss=float(pred@pred),accepted=False,
            status='proposal_only_requires_full_re_equilibration',
            input_fingerprints=input_fingerprints(a.manifest,held),
            weighting=protocol['weighting'],protocol_sha256=sha(a.output/'protocol.json'))
        put(fold+'_proposal.json',prop)
        summaries.append(dict(fold=fold,training_materials=len(prop['input_fingerprints']),
            rows=len(fr),parameters=len(v),loss_before=prop['loss_before'],
            predicted_loss=prop['predicted_loss'],scaled_condition=float(s[0]/s[-1]),
            actual_loss=None,accepted=False))
    with (a.output/'proposals.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    put('audit.json',dict(raw_matches_frozen_v3_exactly=True,
        full_matrix_shape=list(j.shape),grouped_split_verified=True,
        proposals=summaries,nonlinear_evaluations_completed=0,
        whole_material_holdout_results_completed=0))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
