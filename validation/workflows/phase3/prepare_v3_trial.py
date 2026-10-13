"""Prepare full frozen-model nonlinear evaluations, never linear acceptance.

Independent fold proposals retain their original training rows and initialization.
This prepares one trial in a refit, not a completed fit or holdout evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign', type=Path)
    ap.add_argument('--fold', choices=['all', 'leave_diamond_out', 'leave_BN_out'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spectrum-executor', choices=['guarded', 'serial'], default='guarded',
        help='Guarded compares serial/sharded points at every density before full k sharding; serial retains the original execution.')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    campaign = args.campaign.resolve()
    protocol = json.loads((campaign/'protocol.json').read_text())
    manifest_path = Path(protocol['manifest'])
    if digest(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('Frozen baseline manifest changed')
    jacobian_path = Path(protocol['jacobian'])
    if digest(jacobian_path) != protocol['jacobian_sha256']:
        raise ValueError('Frozen v3 Jacobian changed')
    original = json.loads(manifest_path.read_text())
    proposal_path = campaign/(args.fold+'_proposal.json')
    proposal = json.loads(proposal_path.read_text())
    if proposal['protocol_sha256'] != digest(campaign/'protocol.json'):
        raise ValueError('Proposal belongs to a different protocol')
    held = protocol['folds'][args.fold]
    train = set(original)-set(held)
    if set(proposal['input_fingerprints']) != train or proposal['holdouts'] != held:
        raise ValueError('Whole-material split changed')
    provenance = json.loads((campaign/'full_input_provenance.json').read_text())
    for material in original:
        for file, sha in provenance[material]['files'].items():
            if digest(Path(original[material])/file) != sha:
                raise ValueError(f'Frozen input changed: {material}/{file}')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    # Reuse the recorded GPU library environment, not another project's model.
    fd = json.loads((campaign/'Si_theta_p_v3_fd_plan.json').read_text())
    template = fd['experiments'][0]['command']
    pyindex = next(i for i, item in enumerate(template) if item.endswith('/envs/jrystal/bin/python'))
    python = template[pyindex]
    candidate, spectra, experiments = {}, {}, []
    sources = {str(manifest_path): digest(manifest_path), str(proposal_path): digest(proposal_path)}
    for material in original:
        for file in provenance[material]['files']:
            path = Path(original[material])/file
            sources[str(path)] = digest(path)
    for name in ['check_occupations.py', 'variational_ks.py', 'lowest_eigensystem.py',
                 'fit_step.py', 'equilibrium_jacobian.py', 'audit_trial.py']:
        path = root/'validation/workflows/phase3'/name
        sources[str(path)] = digest(path)
    spectral_driver = root/'validation/workflows/phase3'/(
        'guarded_sharded_spectrum.py' if args.spectrum_executor == 'guarded' else 'check_occupations.py')
    if args.spectrum_executor == 'guarded':
        for name in ['guarded_sharded_spectrum.py', 'sharded_spectrum.py', 'validate_sharded_spectrum.py']:
            path=root/'validation/workflows/phase3'/name
            sources[str(path)]=digest(path)
    for runtime in [campaign/'frozen_runtime', campaign/'frozen_runtime_GaP']:
        for path in (runtime/'d3ft').rglob('*.py'):
            sources[str(path)] = digest(path)
        for path in (runtime/'validation').rglob('*'):
            if path.is_file() and path.suffix in ['.py', '.json', '.yaml']:
                sources[str(path)] = digest(path)
    order = [m for m in ['Si', 'BN', 'SiC', 'AlP', 'diamond', 'AlAs', 'GaP'] if m in train]
    for material in order:
        base = Path(original[material]); meta = json.loads((base/'metadata.json').read_text())
        summary = json.loads((base/'summary.json').read_text())
        runtimes = [campaign/'frozen_runtime', campaign/'frozen_runtime_GaP']
        runtimes = [runtime for runtime in runtimes
            if digest(runtime/'validation/workflows/phase3/worker.py') == digest(base/'worker_snapshot.py')]
        if len(runtimes) != 1:
            raise ValueError('No uniquely matching archived worker for '+material)
        runtime = runtimes[0]
        actual_core = {str(p.relative_to(runtime)): digest(p) for p in (runtime/'d3ft').rglob('*.py')}
        if actual_core != meta['code_sha256']:
            raise ValueError('Runtime core differs from '+material+' baseline')
        for file, sha in meta['pseudopotentials'].items():
            if digest(Path(file)) != sha:
                raise ValueError('Pseudopotential changed')
            sources[file] = sha
        cfg, options = meta['config'], meta['arguments']
        prefix = template[:pyindex+1]
        prefix = [x for x in prefix if not x.startswith(('PYTHONPATH=', 'D3FT_PHASE3_RESULTS_ROOT=', 'JAX_COMPILATION_CACHE_DIR='))]
        for i, item in enumerate(prefix):
            if item.startswith('--jobid='): prefix[i] = '--jobid=250285'
            elif item.startswith('--cpus-per-task='): prefix[i] = '--cpus-per-task=64'
            elif item.startswith('--gres=gpu:'): prefix[i] = '--gres=gpu:8'
            elif item.startswith('CUDA_VISIBLE_DEVICES='): prefix[i] = 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'
            elif item == '7200s': prefix[i] = '172800s'
        extra = ['PYTHONPATH='+str(runtime)+':/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/Performance01/chenp/d3ft/software/jrystal',
            'D3FT_PHASE3_RESULTS_ROOT='+str(output/'runs'),
            'JAX_COMPILATION_CACHE_DIR=/tmp/phase3_v3_trial_250285']
        position = prefix.index('env')+1; prefix[position:position] = extra
        if not output.name.replace('_', '').isalnum():
            raise ValueError('Trial folder name must contain only letters, digits and underscores')
        tag = 'trialv3_'+output.name+'_'+material
        candidate[material] = str(output/'runs'/tag)
        spectra[material] = str(output/'spectra'/material)
        worker = runtime/'validation/workflows/phase3/worker.py'
        cmd = prefix+['-u', str(worker), '--material', material, '--mode', 'evaluate', '--tag', tag,
            '--vector', json.dumps(proposal['candidate_vector']), '--a0', str(summary['a0_A']),
            '--grid', str(cfg['grid_sizes']), '--cutoff', str(cfg['cutoff_energy']),
            '--k', str(cfg['k_grid_sizes']), '--empty', str(cfg['empty_bands']),
            '--epochs', str(cfg['epoch']), '--continuation-epochs', str(options.get('continuation_epochs', 0)),
            '--fixed-mask', str(base/'mask.npy'), '--strain-step', str(options['strain_step']),
            '--xc-grid-factor', str(cfg['d3ft'].get('xc_grid_factor', 1)),
            '--asr-force-floor', str(options.get('asr_force_floor', 0)),
            '--pressure-tol', '1e-7', '--fused-report', '--kpool']
        if 'C44_GPa' in summary: cmd.append('--shear')
        experiments.append(dict(name=tag, command=cmd))
        # Full-mesh block solver uses saved stationary density and the same frozen core.
        spectral = prefix+['-u', str(spectral_driver),
            candidate[material], '--out', spectra[material], '--solver', 'block']
        experiments.append(dict(name=tag+'_spectrum', command=spectral))
        sources[str(worker)] = digest(worker)
    def put(name, data):
        (output/name).write_text(json.dumps(data, indent=2)+'\n')
    put('candidate_manifest.json', candidate); put('spectra_manifest.json', spectra)
    put('proposal.json', proposal)
    put('sequence_plan.json', dict(jobid='250285', gpus=list(range(8)), stop_on_failure=True,
        experiments=experiments))
    registry = campaign/'fd_gate_registry.json'
    gates = [Path(path) for path in json.loads(registry.read_text())['acceptance_files']]
    sources[str(registry)] = digest(registry)
    sources[str(output/'sequence_plan.json')] = digest(output/'sequence_plan.json')
    sources[str(output/'proposal.json')] = digest(output/'proposal.json')
    put('trial.json', dict(fold=args.fold, holdouts=held, training_materials=order,
        baseline_manifest=str(manifest_path), proposal=str(output/'proposal.json'),
        sequence_plan=str(output/'sequence_plan.json'), candidate=str(output/'candidate_manifest.json'),
        spectra=str(output/'spectra_manifest.json'), fd_acceptance_files=[str(p) for p in gates],
        spectrum_executor=args.spectrum_executor,
        preparation_source_hashes=sources,
        preparation_script_sha256=digest(Path(__file__)), accepted=False,
        state='prepared_full_nonlinear_trial',
        remaining_acceptance=['candidate numerical budgets', 'derivative coverage beyond control checks as applicable',
            'actual loss decrease/rho', 'fresh full-mesh spectra'],
        scope='One independent proposal evaluation; not a converged refit. Held materials are not evaluated or used for stopping here.'))
    print(json.dumps(dict(fold=args.fold, materials=order, held=held, commands=len(experiments))))


if __name__ == '__main__':
    main()
