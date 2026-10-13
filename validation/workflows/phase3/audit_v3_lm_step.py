"""Audit a real LM transition using raw calculations and resolution evidence.

No candidate is accepted from stored PASS flags or a linear prediction. Loss
intervals below are empirical resolution envelopes, not statistical confidence
intervals or certified continuum bounds. Missing evidence leaves the step pending.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from audit_trial import audit_spectrum
from check_elastic_convergence import compare
from equilibrium_jacobian import derive
from fd_acceptance import component
from fit_step import dataset, validate_trial, SCALE
from check_jacobian_resolution import audit_pair as audit_jacobian_pair


AXES = {'grid', 'cutoff', 'xc_grid', 'strain', 'k'}
FILES = ['metadata.json', 'summary.json', 'points.json', 'mask.npy']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, 'Changed evidence: '+path)


def loss_interval(rows, envelope):
    lower = upper = 0.
    for row in rows:
        key = (row['material'], row['observable'])
        error = envelope[key]/(row['reference']*row['sigma'])
        residual = abs(row['residual']/row['sigma'])
        lower += max(0., residual-error)**2
        upper += (residual+error)**2
    return dict(lower=float(lower), upper=float(upper),
        scope='Empirical resolution envelope; not a confidence interval or continuum proof.')


def numerical_envelope(record, manifest, rows):
    """Recompute all five axes and spectra at the actual parameter vector."""
    record = Path(record)
    completed = read(record/'audit.json')
    reference_mode = completed.get('evidence_kind') == 'frozen_reference_resolution'
    if reference_mode:
        protocol_path = Path(completed['protocol'])
        verify_hashes(completed['source_hashes'])
        require(sha(manifest) == sha(completed['manifest']), 'Reference numerical manifest mismatch')
    else:
        trial = Path(completed['trial'])
        protocol_path = trial/'numerical_protocol_executable/protocol.json'
    require(sha(protocol_path) == completed['protocol_sha256'], 'Numerical protocol changed')
    protocol = read(protocol_path)
    verify_hashes(protocol['input_sha256'])
    jacobian_protocol_path=record/'jacobian_resolution_protocol.json'
    jacobian_protocol=read(jacobian_protocol_path)
    verify_hashes(jacobian_protocol['input_sha256'])
    work = read(manifest)
    selected = {r['material'] for r in rows}
    require(selected.issubset(work), 'Numerical manifest is missing training materials')
    if reference_mode:
        require(protocol.get('evidence_kind') == 'frozen_reference_resolution' and
                protocol['candidate_vector'] == [0.]*7 and not protocol['holdouts'],
                'Reference resolution evidence uses an invalid model or split')
        for material in selected:
            require(audit_spectrum(Path(work[material]),Path(completed['baseline_spectra'][material])) ==
                    completed['baseline_spectral_evidence'][material], 'Reference density-bound spectrum changed')
    require(sha(Path(__file__).with_name('check_elastic_convergence.py')) == protocol['comparison_script_sha256'],
            'Registered numerical comparison code changed')
    envelope = {(r['material'], r['observable']):0. for r in rows}
    registered = {(p['material'], p['axis']):p for p in protocol['comparisons']
                  if p['material'] in selected}
    require(len(registered) == sum(p['material'] in selected for p in protocol['comparisons']) == 5*len(selected) and
            set(registered) == {(m,a) for m in selected for a in AXES},
            'Incomplete or duplicated five-axis numerical registration')
    sources = {str(protocol_path):sha(protocol_path), str(record/'audit.json'):sha(record/'audit.json'),
               str(jacobian_protocol_path):sha(jacobian_protocol_path),
               **jacobian_protocol['input_sha256']}
    for (material, axis), pair in registered.items():
        key = material+'_'+axis
        require(set(pair['required_training_observables']) == {
            row['observable'] for row in rows if row['material']==material},
            'Numerical derivative observation scope differs from the authoritative training rows')
        result_path = record/(key+'_result.json')
        certificate_path = record/(key+'_certificate.json')
        result, certificate = read(result_path), read(certificate_path)
        require(result.get('state') == 'pair_passed' and result.get('passed') is True,
                'Numerical pair has not passed: '+key)
        verify_hashes({str(Path(p)/name):h for p, files in certificate['sources'].items()
                       for name,h in files.items()})
        coarse, fine = Path(pair['coarse_candidate']), Path(result['fine'])
        require(str(coarse) in certificate['sources'] and str(fine) in certificate['sources'],
                'Numerical pair identity changed')
        anchor = Path(work[material])
        require(read(anchor/'summary.json')['parameter_vector'] == protocol['candidate_vector'],
                'Numerical evidence is for a different parameter vector')
        if coarse != anchor:
            # Only the two registered bridge cases may replace the anchor.
            grid_pair = registered[(material,'grid')]
            require(grid_pair.get('grid_bridge_output') == str(coarse), 'Unregistered coarse bridge')
        recomputed = compare(coarse, fine, axis, pair['gate']['lattice_relative'],
            pair['gate']['elastic_absolute_GPa'],
            pair['axis_definition'] == 'coarse FFT grid with physical XC grid held constant')
        require(recomputed['passed'] and recomputed['rows'] == certificate['rows'],
                'Stored numerical result differs from its raw calculation')
        jacobian_result=audit_jacobian_pair(coarse,fine,pair,jacobian_protocol)
        require(jacobian_result['passed'],'Equilibrium Jacobian numerical resolution failed: '+key)
        sources.update(jacobian_result['source_sha256'])
        spectrum = protocol_path.parent/'spectra'/key
        require(audit_spectrum(fine,spectrum) == certificate['spectrum'], 'Numerical spectrum changed')
        base_core = read(anchor/'metadata.json')['code_sha256']
        fine_core = read(fine/'metadata.json')['code_sha256']
        changed = {p for p in set(base_core)|set(fine_core) if base_core.get(p) != fine_core.get(p)}
        if changed:
            require(changed == {'d3ft/optim/preconditioner.py'}, 'Unverified numerical core change')
            runtime_name = 'frozen_runtime_GaP_memory' if material in ['GaP','AlAs'] else 'frozen_runtime_memory'
            provenance_path = (Path(certificate['runtime_provenance']) if reference_mode else
                               record/runtime_name/'numerical_runtime_provenance.json')
            if reference_mode:
                require(sha(provenance_path) == certificate['runtime_provenance_sha256'],
                        'Reference numerical runtime provenance changed')
            provenance = read(provenance_path)
            evidence_path = Path(provenance['equivalence_report'])
            evidence = read(evidence_path)
            require(sha(evidence_path) == certificate['equivalence_report_sha256'] and evidence['passed'] and
                    all(c['exact_elementwise_equal'] for c in evidence['cases']) and
                    base_core['d3ft/optim/preconditioner.py'] == evidence['old_sha256'] and
                    fine_core == provenance['core_sha256'], 'Memory source equivalence mismatch')
            verify_hashes(evidence['input_files'])
            sources[str(evidence_path)] = sha(evidence_path)
        for r in recomputed['rows']:
            identity = (material,r['observable'])
            if identity in envelope:
                envelope[identity] += abs(r['difference'])
        sources[str(certificate_path)] = sha(certificate_path)
        sources[str(result_path)] = sha(result_path)
        sources[str(spectrum/'summary.json')] = sha(spectrum/'summary.json')
    # Twofold measured-axis sum, plus the previously declared absolute FD floors.
    # This is an operational resolution envelope, never a mathematical error bound.
    envelope = {key:2*value+(1e-7 if key[1] == 'a0_A' else 1e-3)
                for key,value in envelope.items()}
    return envelope, sources


def derivative_evidence(index, manifest, rows, starting_vector):
    """Recompute two-step FD controls plus material-wide directional checks.

    Seven independent tested parameter directions and one two-step check per
    training material are required. This is not an entrywise 24x7 FD claim.
    Held material values never contribute to the objective or stopping criteria.
    """
    index = Path(index); registered = read(index); all_work = read(manifest)
    verify_hashes(registered['registered_source_sha256'])
    selected = {r['material'] for r in rows}
    require(selected.issubset(all_work), 'FD manifest is missing training materials')
    work = {m:all_work[m] for m in selected}
    require(registered['starting_vector'] == starting_vector, 'FD center differs from actual LM baseline')
    sources = {str(index):sha(index)}
    tests = []; directions = []; covered = set()
    for case in registered['cases']:
        verify_hashes(case['base_input_sha256'])
        base = Path(case['base']); summary = read(base/'summary.json'); meta = read(base/'metadata.json')
        material = summary['material']
        require(summary['parameter_vector'] == starting_vector, 'FD uses another center')
        require(meta['code_sha256'] == read(Path(next(iter(work.values())))/'metadata.json')['code_sha256'],
                'Implementation control is not on the current frozen core')
        if case['role'] == 'training_material':
            require(material in work and base.resolve() == Path(work[material]).resolve(),
                    'Material directional FD is not bound to this training baseline')
            covered.add(material)
            required = {r['observable'] for r in rows if r['material'] == material}
        else:
            require(case['role'] == 'implementation_control', 'Unknown FD evidence role')
            required = set(case['observables'])
            available = {key for key in ['a0_A','B_GPa','C11_minus_C12_GPa','C44_GPa'] if key in summary}
            require(required == available and required, 'Implementation control has incomplete observable scope')
        require(len(case['pairs']) == 2 and {p['alpha'] for p in case['pairs']} ==
                {case['small_alpha'],case['large_alpha']} and 0 < case['small_alpha'] < case['large_alpha'],
                'Need two distinct positive FD steps')
        direction = np.asarray(case['physical_direction'], float)
        require(direction.shape == (7,) and np.isfinite(direction).all() and np.linalg.norm(direction)>0,
                'Invalid FD direction')
        jac = derive(base)
        gradients = dict(a0_A=np.asarray(jac['dlogV_equilibrium_dp'])*summary['a0_A']/3,
            B_GPa=jac['dB_equilibrium_dp'],C11_minus_C12_GPa=jac.get('dD_equilibrium_dp'),
            C44_GPa=jac.get('dC44_equilibrium_dp'))
        derivatives = []
        for pair in sorted(case['pairs'],key=lambda p:p['alpha']):
            endpoints = [Path(pair[sign]) for sign in ['minus','plus']]
            values = []
            for path, sign in zip(endpoints,[-1,1]):
                result, emeta = read(path/'summary.json'), read(path/'metadata.json')
                expected = np.asarray(starting_vector)+sign*pair['alpha']*direction
                require(np.allclose(result['parameter_vector'],expected,rtol=0,atol=1e-14),
                        'FD endpoint has wrong parameter direction or step')
                require(result['material'] == material and emeta['code_sha256'] == meta['code_sha256'],
                        'FD material or frozen core changed')
                require(emeta['pseudopotentials'] == meta['pseudopotentials'], 'FD pseudopotentials changed')
                for key in ['grid_sizes','cutoff_energy','k_grid_sizes','empty_bands','xc','spin']:
                    require(emeta['config'][key] == meta['config'][key], 'FD configuration changed: '+key)
                for key in ['strain_step','asr_force_floor']:
                    require(emeta['arguments'].get(key,0) == meta['arguments'].get(key,0), 'FD estimator changed')
                require(emeta['config']['d3ft'].get('xc_grid_factor',1) == meta['config']['d3ft'].get('xc_grid_factor',1),
                        'FD XC grid changed')
                require(np.array_equal(np.load(base/'mask.npy'),np.load(path/'mask.npy')), 'FD Miller basis changed')
                require(abs(result['equilibrium_pressure_GPa'])/(3*result['B_GPa']) <= 1e-7,
                        'FD pressure root is not sufficiently tight')
                derive(path)
                values.append(result)
                sources.update({str(path/name):sha(path/name) for name in FILES})
            components = []
            for key in sorted(required):
                require(gradients.get(key) is not None, 'Missing FD observable')
                predicted = float(np.asarray(gradients[key]) @ (2*pair['alpha']*direction))
                actual = values[1][key]-values[0][key]
                components.append(component(key,predicted,actual,summary['a0_A'],.02))
            require(all(c['joint_gate_passed'] for c in components), 'FD agreement or original signal budget failed')
            derivatives.append({c['observable']:c['actual_difference']/(2*pair['alpha']) for c in components})
        for key in required:
            a,b = [d[key] for d in derivatives]
            floor = 1e-7 if key == 'a0_A' else 1e-3
            tolerance = .02*max(abs(a),abs(b))+floor/(2*case['small_alpha'])
            require(abs(a-b) <= tolerance, 'FD two-step inconsistency')
        if case['role'] == 'implementation_control':
            scaled = direction/SCALE
            directions.append(scaled/np.linalg.norm(scaled))
        tests.append(dict(material=material,role=case['role'],observables=sorted(required),passed=True))
        sources.update({str(base/name):sha(base/name) for name in FILES})
    require(covered == set(work), 'Directional derivative checks do not cover every training material')
    singular_values = np.linalg.svd(np.asarray(directions),compute_uv=False) if directions else np.array([])
    rank = int(np.sum(singular_values > 1e-8*singular_values[0])) if len(singular_values) else 0
    require(rank == 7, 'Implementation controls do not span seven parameter directions')
    return dict(registered_controls_rank=int(rank),control_direction_singular_values=singular_values.tolist(),
        training_material_direction_checks=sorted(covered),
        tests=tests,entrywise_jacobian_fd_certified=False), sources


def audit(trial_path, before_numerical, after_numerical, fd_index):
    trial_path = Path(trial_path); trial = read(trial_path); proposal = read(trial['proposal'])
    verify_hashes(trial['preparation_source_hashes'])
    before, after = [read(trial[k]) for k in ['baseline_manifest','candidate']]
    train = set(proposal['input_fingerprints'])
    require(set(after) == set(trial['training_materials']) == train, 'Incomplete training set')
    require(not train.intersection(proposal['holdouts']), 'Held material included in fitting')
    for material, files in proposal['input_fingerprints'].items():
        verify_hashes({str(Path(before[material])/name):digest for name,digest in files.items()})
    validate_trial(trial['baseline_manifest'],trial['candidate'],proposal['holdouts'])
    v, old_rows, r, j = dataset(trial['baseline_manifest'],proposal['holdouts'])
    cv, new_rows, cr, _ = dataset(trial['candidate'],proposal['holdouts'])
    identity = lambda rr:[(x['material'],x['observable'],x['reference'],x['sigma']) for x in rr]
    require(identity(old_rows) == identity(new_rows) == identity(proposal['training_rows']), 'Training targets changed')
    require(v.tolist() == proposal['starting_vector'] and cv.tolist() == proposal['candidate_vector'], 'LM vector mismatch')
    for material,path in after.items():
        require(read(Path(path)/'metadata.json')['code_sha256'] == read(Path(before[material])/'metadata.json')['code_sha256'],
                'Candidate changed the frozen physical model')
        audit_spectrum(Path(path),Path(read(trial['spectra'])[material]))
    actual = float(r@r-cr@cr)
    predicted = float(r@r-(r+j@(cv-v))@(r+j@(cv-v)))
    rho = actual/predicted if predicted > 0 else None
    sources = {str(trial_path):sha(trial_path),trial['proposal']:sha(trial['proposal']),
               trial['baseline_manifest']:sha(trial['baseline_manifest']),trial['candidate']:sha(trial['candidate'])}
    sources.update({str(Path(path)/name):sha(Path(path)/name) for path in after.values() for name in FILES})
    if actual <= 0 or rho is None or rho <= .1:
        return dict(accepted=False,decision='rejected_actual_loss',actual_reduction=actual,
                    predicted_reduction=predicted,rho=rho,source_sha256=sources)
    # Only training rows contribute to intervals, even for a shared baseline audit.
    # numerical_envelope takes a manifest; the original may contain held materials.
    # Its comparisons are filtered by the authoritative training rows.
    be, bs = numerical_envelope(before_numerical,trial['baseline_manifest'],old_rows)
    ae, aps = numerical_envelope(after_numerical,trial['candidate'],new_rows)
    controls, ds = derivative_evidence(fd_index,trial['baseline_manifest'],old_rows,v.tolist())
    low, high = loss_interval(old_rows,be), loss_interval(new_rows,ae)
    reduction_lower = low['lower']-high['upper']
    lower_rho = reduction_lower/predicted
    accepted = bool(reduction_lower > 0 and lower_rho > .1)
    sources.update(bs);sources.update(aps);sources.update(ds)
    return dict(accepted=accepted,decision='accepted_real_lm_step' if accepted else 'needs_finer_numerical_resolution',
        loss_before=float(r@r),loss_after=float(cr@cr),actual_reduction=actual,predicted_reduction=predicted,rho=rho,
        empirical_loss_interval_before=low,empirical_loss_interval_after=high,
        empirical_reduction_lower=reduction_lower,empirical_rho_lower=lower_rho,derivatives=controls,
        all_registered_projected_jacobian_resolution_gates_passed=True,
        entrywise_finite_difference_validated=False,weak_jacobian_responses_claimed_resolved=False,
        candidate_vector=cv.tolist(),holdouts=proposal['holdouts'],candidate_manifest=trial['candidate'],
        source_sha256=sources,certified_continuum_error_bounds=False,refit_complete=False,
        whole_material_holdout_evaluated=False,
        scope='One real LM step resolved across the registered numerical refinements and derivative checks. No converged fit, statistical uncertainty or entrywise Jacobian FD claim.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trial',type=Path)
    for name in ['before-numerical','after-numerical','fd-index','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    try:
        result = audit(args.trial,args.before_numerical,args.after_numerical,args.fd_index)
    except (ValueError,KeyError,OSError,AssertionError) as exc:
        result = dict(accepted=False,decision='missing_or_invalid_acceptance_evidence',error=str(exc),refit_complete=False)
    result['audit_script_sha256'] = sha(__file__)
    result['audit_inputs'] = {key:str(getattr(args,key).resolve())
        for key in ['trial','before_numerical','after_numerical','fd_index']}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
