"""Prepare a smaller LM retry after replaying a real loss/rho rejection.

Missing evidence and unresolved numerical reduction are not loss rejections.
The center and whole-material split stay fixed; no GPU job is launched.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from audit_v3_lm_step import audit, derivative_evidence, read, sha, verify_hashes
from fit_step import dataset, input_fingerprints, SCALE


POLICY = dict(rejected_damping_multiplier=4., rejected_radius_divisor=2.,
              minimum_retry_radius=1e-6, maximum_trials=20,
              scope='Common prospective retry rule for every independent fold. '
                    'A radius floor or trial limit requests review, not fit completion.')


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)


def replay_rejection(path):
    recorded = read(path)
    if recorded.get('accepted') is not False or recorded.get('decision') != 'rejected_actual_loss':
        raise ValueError('Only a real loss/rho rejection can prepare this retry')
    if recorded.get('audit_script_sha256') != sha(Path(__file__).with_name('audit_v3_lm_step.py')):
        raise ValueError('Rejection auditor changed')
    verify_hashes(recorded['source_sha256'])
    inputs = recorded['audit_inputs']
    replay = audit(inputs['trial'], inputs['before_numerical'], inputs['after_numerical'], inputs['fd_index'])
    if (replay.get('accepted') is not False or replay.get('decision') != 'rejected_actual_loss' or
        any(replay[key] != recorded.get(key) for key in replay)):
        raise ValueError('Real rejection does not replay from raw evidence')
    return recorded, inputs


def retry_step(vector, residual, jacobian, proposal):
    damping = float(proposal['damping']) * POLICY['rejected_damping_multiplier']
    radius = float(proposal['radius']) / POLICY['rejected_radius_divisor']
    if not np.isfinite([damping, radius]).all() or min(damping, radius) <= 0:
        raise ValueError('Invalid prior LM controls')
    scaled = jacobian * SCALE
    u, singular_values, vt = np.linalg.svd(scaled, full_matrices=False)
    z = -vt.T @ ((singular_values / (singular_values**2 + damping)) * (u.T @ residual))
    z *= min(1., radius / max(np.linalg.norm(z), 1e-30))
    constants = np.array([.06672455060314922, .031090690869654894, .804, .2195149727645171])
    bounds = np.r_[np.full(3, .5), .5 * constants]
    candidate = np.clip(vector + SCALE * z, -bounds, bounds)
    predicted = residual + jacobian @ (candidate - vector)
    return candidate, damping, radius, singular_values, float(predicted @ predicted)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rejection', type=Path)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--trial-index', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rejected, inputs = replay_rejection(args.rejection)
    trial = read(inputs['trial']); proposal = read(trial['proposal'])
    campaign = args.campaign.resolve(); fold = trial['fold']; held = proposal['holdouts']
    protocol = read(campaign / 'protocol.json')
    # Initial proposals reside inside the trial. Later ones also have their
    # original preparation context explicitly pinned in the trial sources.
    protocol_sources = [Path(p) for p in trial['preparation_source_hashes']
                        if Path(p).name == 'fd_gate_registry.json']
    if len(protocol_sources) != 1:
        raise ValueError('No unique pinned FD registry for the rejected center')
    registry_path = protocol_sources[0]
    verify_hashes(trial['preparation_source_hashes'])
    registry = read(registry_path)
    if registry['acceptance_files'] != trial['fd_acceptance_files']:
        raise ValueError('Retry would change the center FD requirements')
    if protocol['folds'][fold] != held:
        raise ValueError('Retry would change the independent whole-material split')
    context_protocol = registry_path.parent / 'protocol.json'
    old_index = read(context_protocol).get('iteration_index', 0)
    if args.trial_index != old_index + 1:
        raise ValueError('Retry index must follow the actual prior trial index')
    vector, rows, residual, jacobian = dataset(trial['baseline_manifest'], held)
    if vector.tolist() != proposal['starting_vector']:
        raise ValueError('Rejected candidate cannot become the new center')
    if registry.get('starting_vector', vector.tolist()) != vector.tolist():
        raise ValueError('FD registry labels a different parameter center')
    controls, control_sources = derivative_evidence(inputs['fd_index'], trial['baseline_manifest'], rows, vector.tolist())
    candidate, damping, radius, singular_values, predicted_loss = retry_step(vector, residual, jacobian, proposal)
    args.out.mkdir(parents=True, exist_ok=False); out = args.out.resolve()
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    put(out / 'iteration_policy.json', POLICY)
    if args.trial_index >= POLICY['maximum_trials'] or radius < POLICY['minimum_retry_radius']:
        put(out / 'status.json', dict(state='retry_limit_requires_review', refit_complete=False,
            held_materials_evaluated=False, trial_index=args.trial_index, radius=radius))
        return
    if predicted_loss >= float(residual @ residual) or np.array_equal(candidate, vector):
        put(out / 'status.json', dict(state='no_descent_retry_requires_review', refit_complete=False,
            held_materials_evaluated=False))
        return
    context = out / 'preparation_context'; context.mkdir()
    manifest = context / 'manifest.json'; put(manifest, read(trial['baseline_manifest']))
    matrix = context / 'iteration_jacobian.json'
    put(matrix, dict(version=3, stage='unchanged_center_after_real_rejection', manifest_sha256=sha(manifest),
        parameter_vector=vector.tolist(), rows=rows, normalized_residual=residual.tolist(),
        normalized_jacobian=jacobian.tolist(), input_fingerprints=input_fingerprints(manifest, held), holdouts=held))
    protocol.update(manifest=str(manifest), manifest_sha256=sha(manifest), jacobian=str(matrix), jacobian_sha256=sha(matrix),
        iteration_index=args.trial_index, state='prepared_retry_after_replayed_real_rejection',
        initialization='unchanged starting center of this independently initialized fold')
    put(context / 'protocol.json', protocol)
    put(context / (fold + '_proposal.json'), dict(starting_vector=vector.tolist(), candidate_vector=candidate.tolist(),
        holdouts=held, training_rows=rows, parameter_scales=SCALE.tolist(), singular_values=singular_values.tolist(),
        damping=damping, radius=radius, loss_before=float(residual @ residual), predicted_loss=predicted_loss,
        accepted=False, status='proposal_only_requires_full_re_equilibration',
        input_fingerprints=input_fingerprints(manifest, held), weighting=protocol['weighting'],
        protocol_sha256=sha(context / 'protocol.json')))
    provenance = {m: dict(files={name: sha(Path(path) / name) for name in ['metadata.json', 'summary.json', 'points.json', 'mask.npy']})
                  for m, path in read(manifest).items()}
    put(context / 'full_input_provenance.json', provenance)
    put(context / 'fd_gate_registry.json', registry | dict(starting_vector=vector.tolist()))
    put(context / 'Si_theta_p_v3_fd_plan.json', read(campaign / 'Si_theta_p_v3_fd_plan.json'))
    for name in ['frozen_runtime', 'frozen_runtime_GaP']:
        (context / name).symlink_to(campaign / name, target_is_directory=True)
    next_trial = out / f'{fold}_t{args.trial_index:02d}'
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_v3_trial.py')), str(context),
                    '--fold', fold, '--output', str(next_trial)], check=True)
    put(out / 'transition.json', dict(time=datetime.datetime.now().isoformat(), state='prepared_not_submitted',
        rejection=str(args.rejection.resolve()), rejection_sha256=sha(args.rejection),
        trial=str(next_trial / 'trial.json'), fold=fold, holdouts=held, starting_vector=vector.tolist(),
        fd_controls=controls, fd_source_sha256=control_sources, actual_trials_launched=0,
        refit_complete=False, held_materials_evaluated=False, script_sha256=sha(__file__)))
    print(json.dumps(dict(next_trial=str(next_trial), computations_launched=0, held=held)))


if __name__ == '__main__':
    main()
