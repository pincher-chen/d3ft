"""Register candidate numerical comparisons before observing their results.

These are initial independent refinement pairs, not claimed continuum bounds.
No calculation is launched. Failed pairs require further controlled refinement.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from numerical_geometry import actual_fft_size, REFERENCE, sphere, density_support


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trial', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    trial = read(args.trial)
    proposal, base, candidate = [read(trial[k]) for k in ['proposal', 'baseline_manifest', 'candidate']]
    train = set(proposal['input_fingerprints'])
    if set(candidate) != train or set(trial['training_materials']) != train:
        raise ValueError('Candidate numerical protocol requires the exact training split')
    fingerprints = {str(p): sha(p) for p in [args.trial, Path(trial['proposal']),
        Path(trial['baseline_manifest']), Path(trial['candidate'])]}
    for source in [REFERENCE/'const.py', REFERENCE/'grid.py',
        Path(__file__).with_name('numerical_geometry.py')]:
        fingerprints[str(source.resolve())] = sha(source)
    comparisons = []
    for material in sorted(train):
        directory = Path(base[material])
        meta = read(directory/'metadata.json')
        for file, digest in proposal['input_fingerprints'][material].items():
            if sha(directory/file) != digest:
                raise ValueError('Frozen baseline changed')
        cfg, options = meta['config'], meta['arguments']
        setting = dict(cutoff=cfg['cutoff_energy'], grid=cfg['grid_sizes'],
            k=cfg['k_grid_sizes'], xc_grid=cfg['d3ft'].get('xc_grid_factor', 1),
            strain=options['strain_step'])
        refinements = dict(cutoff=math.ceil(1.5*setting['cutoff']),
            grid=actual_fft_size(4*math.ceil(1.25*setting['grid']/4)), k=setting['k']+2,
            xc_grid=min(8, math.ceil(1.5*setting['xc_grid'])), strain=setting['strain']/2)
        targets = [r['observable'] for r in proposal['training_rows'] if r['material'] == material]
        bridge = None
        if setting['xc_grid'] == 8:
            # 48*8 = 64*6 = 384: change the coarse FFT grid while holding
            # the physical XC quadrature fixed; then test 384 -> 512.
            bridge_grid = 4*setting['grid']//3
            if (3*bridge_grid == 4*setting['grid'] and bridge_grid % 4 == 0):
                bridge = {**setting, 'grid':bridge_grid, 'xc_grid':6}
        anchor_a0 = read(directory/'summary.json')['a0_A']
        cutoff_support = density_support(sphere(anchor_a0, setting['grid'], refinements['cutoff']))
        cutoff_bridge = None if cutoff_support['full_density_support'] else (
            bridge or {**setting, 'grid':refinements['grid']})
        grid_output = str(args.output.resolve()/'runs'/f'{material}_grid_bridge')
        for axis, value in refinements.items():
            unavailable = axis == 'xc_grid' and value == setting[axis]
            fine = None if unavailable else {**setting, axis:value}
            coarse_setting, source = setting, candidate[material]
            definition = 'one registered configuration axis'
            if bridge is not None and axis == 'grid':
                fine = bridge
                definition = 'coarse FFT grid with physical XC grid held constant'
            if bridge is not None and axis == 'xc_grid':
                coarse_setting = bridge
                source = str(args.output.resolve()/'runs'/f'{material}_grid_bridge')
                fine = {**bridge, 'xc_grid':8}
                unavailable = False
                definition = 'physical XC grid after independently screened coarse-grid bridge'
            if cutoff_bridge is not None and axis == 'cutoff':
                coarse_setting = cutoff_bridge
                source = grid_output
                fine = {**cutoff_bridge, 'cutoff':value}
                definition = 'cutoff after independently screened grid bridge with adequate density support'
            comparisons.append(dict(material=material, axis=axis,
                coarse_candidate=source, coarse_settings=coarse_setting, fine_settings=fine,
                axis_definition=definition,
                grid_bridge_output=(grid_output
                    if (bridge is not None or cutoff_bridge is not None) and axis=='grid' else None),
                cutoff_preflight_at_base_geometry=cutoff_support if axis=='cutoff' else None,
                parameter_vector=proposal['candidate_vector'], required_training_observables=targets,
                basis_policy=('exact remap of the candidate Miller set' if axis=='grid' else
                    'retain exact remapped bridge Miller mask' if bridge is not None and axis=='xc_grid' else
                    'retain candidate ordered Miller mask' if axis in ['k', 'xc_grid', 'strain'] else
                    'larger cutoff at the same coarse grid and geometry anchor; verify density support before solve'),
                spectrum_policy='fresh equilibrium density-bound complete refined SCF mesh',
                local_integral='frozen-v3 trapezoid',
                force_floor=options.get('asr_force_floor', 0),
                gate=dict(lattice_relative=1e-4, elastic_absolute_GPa=.1),
                state='requires_independent_grid_bridge' if unavailable else 'registered_unexecuted',
                refinement_limit_note=('The archived worker stops at factor 8. Do not count a no-op as a test. Require a controlled coarse-grid bridge at identical physical XC grid, then a fine-XC refinement, or a separately audited worker and measured memory feasibility.' if unavailable else None)))
    report = dict(frozen_at=datetime.datetime.now().isoformat(), comparisons=comparisons,
        candidate_vector=proposal['candidate_vector'], holdouts=proposal['holdouts'],
        input_sha256=fingerprints, preparation_script_sha256=sha(Path(__file__)),
        comparison_script_sha256=sha(Path(__file__).with_name('check_elastic_convergence.py')),
        execution_prerequisite='Complete actual trial with matching vector and nominal real-loss decrease; all selected devices idle. Preparation alone does not submit these calculations.',
        implementation_policy='Use matching archived worker/core. A memory-only equivalent implementation needs a separately verified exact source-pair certificate; all other source changes invalidate the pair.',
        failed_pair_policy='Record FAIL and refine the failing axis. Do not loosen tolerances, average away failure, or silently change another axis. Insufficient density support requires a separately controlled grid bridge.',
        acceptance_policy='All required observables and all registered axes must have actual valid comparisons. Each pair is empirical resolution screening only; loss interval error bounds and parameter-derivative bounds require separate justification.',
        budgets_are_not_experimental_uncertainties=True, certified_total_error_bounds=False,
        accepted=False, scope='Prospective complete candidate forward-resolution protocol. Not an execution result, parameter-derivative certificate, accepted nonlinear fit or whole-material holdout outcome.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(materials=len(train), registered_comparisons=len(comparisons),
        computations_launched=0, accepted=False)))


if __name__ == '__main__':
    main()
