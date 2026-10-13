"""Prepare the next frozen-v3 LM trial after independently replayed acceptance.

This never submits a job or evaluates a held material. The same fixed stopping
review and parameter rules apply to all independently initialized refit folds.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from audit_v3_lm_step import audit, verify_hashes
from fit_step import dataset, input_fingerprints, SCALE


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('acceptance',type=Path)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--fd-registry',type=Path,required=True,
        help='Explicitly registered controls at the new iteration starting vector')
    ap.add_argument('--trial-index',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();accepted=read(args.acceptance)
    if not accepted.get('accepted') or accepted.get('decision')!='accepted_real_lm_step':
        raise ValueError('A real fully audited LM step is required; nothing prepared')
    audit_source=Path(__file__).with_name('audit_v3_lm_step.py')
    if accepted['audit_script_sha256']!=sha(audit_source):
        raise ValueError('Acceptance auditor changed')
    verify_hashes(accepted['source_sha256'])
    inputs=accepted['audit_inputs']
    replay=audit(inputs['trial'],inputs['before_numerical'],inputs['after_numerical'],inputs['fd_index'])
    if not replay['accepted'] or any(replay[k]!=accepted[k] for k in replay):
        raise ValueError('Acceptance does not replay from its raw evidence')
    if args.trial_index<1:
        raise ValueError('The first trial is independently initialized by prepare_v3_trial')
    prior=read(inputs['trial']);proposal=read(prior['proposal']);campaign=args.campaign.resolve()
    held=proposal['holdouts'];fold=prior['fold']
    if read(campaign/'protocol.json')['folds'][fold]!=held:
        raise ValueError('Refit holdout definition changed')
    # The held baselines remain untouched; dataset excludes their entire rows.
    current=read(prior['baseline_manifest'])|read(prior['candidate'])
    args.out.mkdir(parents=True,exist_ok=False);out=args.out.resolve()
    put(out/'starting_manifest.json',current)
    vector,rows,r,j=dataset(out/'starting_manifest.json',held)
    if vector.tolist()!=accepted['candidate_vector']:
        raise ValueError('Accepted vector differs from new starting point')
    relative=accepted['actual_reduction']/accepted['loss_before']
    policy=dict(loss_relative_review_threshold=1e-3,initial_radius=.04,
        initial_damping=1.,accepted_radius_growth=1.5,maximum_radius=2.,
        accepted_damping_divisor=2.,minimum_damping=1e-6,maximum_trials=20,
        scope='Common fold-independent controls inherited from the actual LM workflow. Small loss reduction requests stopping review; it does not certify stationarity or automatically complete a refit.')
    put(out/'iteration_policy.json',policy)
    if relative<policy['loss_relative_review_threshold'] or args.trial_index>=policy['maximum_trials']:
        put(out/'status.json',dict(state='stopping_review_required' if args.trial_index<20 else 'trial_limit_requires_review',
            relative_actual_loss_reduction=relative,refit_complete=False,held_materials_evaluated=False))
        print('No new GPU trial: stopping review is required.');return
    registry=read(args.fd_registry)
    if registry.get('starting_vector')!=vector.tolist() or not registry.get('acceptance_files'):
        raise ValueError('New iteration FD registry is not explicitly tied to the new center')
    damping=max(proposal['damping']/2,1e-6);radius=min(proposal['radius']*1.5,2.)
    scaled=j*SCALE;u,s,vt=np.linalg.svd(scaled,full_matrices=False)
    z=-vt.T@((s/(s*s+damping))*(u.T@r));z*=min(1.,radius/max(np.linalg.norm(z),1e-30))
    constants=np.array([.06672455060314922,.031090690869654894,.804,.2195149727645171])
    bounds=np.r_[np.full(3,.5),.5*constants]
    candidate=np.clip(vector+SCALE*z,-bounds,bounds)
    predicted=r+j@(candidate-vector)
    context=out/'preparation_context';context.mkdir()
    manifest=context/'manifest.json';put(manifest,current)
    matrix=context/'iteration_jacobian.json'
    put(matrix,dict(version=3,stage='nonlinear_iteration_point_not_untrained_production',
        manifest_sha256=sha(manifest),parameter_vector=vector.tolist(),rows=rows,
        normalized_residual=r.tolist(),normalized_jacobian=j.tolist(),
        input_fingerprints=input_fingerprints(manifest,held),holdouts=held))
    protocol=read(campaign/'protocol.json')
    protocol.update(manifest=str(manifest),manifest_sha256=sha(manifest),jacobian=str(matrix),jacobian_sha256=sha(matrix),
        parent_untrained_protocol=str(campaign/'protocol.json'),parent_untrained_protocol_sha256=sha(campaign/'protocol.json'),
        iteration_index=args.trial_index,state='prepared_next_real_nonlinear_trial',initialization='accepted point in this independently initialized fold')
    put(context/'protocol.json',protocol)
    next_proposal=dict(starting_vector=vector.tolist(),candidate_vector=candidate.tolist(),holdouts=held,
        training_rows=rows,parameter_scales=SCALE.tolist(),singular_values=s.tolist(),
        damping=damping,radius=radius,loss_before=float(r@r),predicted_loss=float(predicted@predicted),
        accepted=False,status='proposal_only_requires_full_re_equilibration',
        input_fingerprints=input_fingerprints(manifest,held),weighting=protocol['weighting'],
        protocol_sha256=sha(context/'protocol.json'))
    put(context/(fold+'_proposal.json'),next_proposal)
    provenance={m:dict(files={name:sha(Path(path)/name) for name in ['metadata.json','summary.json','points.json','mask.npy']})
                for m,path in current.items()}
    put(context/'full_input_provenance.json',provenance)
    put(context/'fd_gate_registry.json',registry)
    put(context/'Si_theta_p_v3_fd_plan.json',read(campaign/'Si_theta_p_v3_fd_plan.json'))
    for name in ['frozen_runtime','frozen_runtime_GaP']:
        (context/name).symlink_to(campaign/name,target_is_directory=True)
    trial=out/f'{fold}_t{args.trial_index:02d}'
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_v3_trial.py')),str(context),
        '--fold',fold,'--output',str(trial)],check=True)
    put(out/'transition.json',dict(time=datetime.datetime.now().isoformat(),state='prepared_not_submitted',
        prior_acceptance=str(args.acceptance.resolve()),prior_acceptance_sha256=sha(args.acceptance),
        proposal=str(trial/'proposal.json'),trial=str(trial/'trial.json'),fold=fold,holdouts=held,
        actual_trials_launched=0,refit_complete=False,held_materials_evaluated=False,
        script_sha256=sha(__file__)))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(next_trial=str(trial),computations_launched=0,held=held)))


if __name__=='__main__':
    main()
