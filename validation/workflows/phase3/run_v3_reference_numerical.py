"""Execute zero-parameter resolution pairs, independently of LM proposals.

Reference calculations may reuse exactly matching immutable fine endpoints.
Reused endpoints still require density-bound full spectra and raw re-comparison.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from audit_trial import audit_spectrum
from check_elastic_convergence import compare, miller_set
from embed_mask import remap
from numerical_geometry import sphere, density_support
from run_v3_numerical_validation import (read, sha, put, verify_hashes,
                                        check_registered_settings, clone_memory_runtime)
from run_v3_reference_spectra import dependency_state


def validate_reference(plan, protocol):
    if protocol['evidence_kind'] != 'frozen_reference_resolution' or protocol['candidate_vector'] != [0.]*7:
        raise ValueError('Not a registered zero-parameter reference protocol')
    if protocol['holdouts'] or len(protocol['comparisons']) != 35:
        raise ValueError('Reference resolution registration is incomplete')
    manifest = read(plan['manifest'])
    if len(manifest) != 7 or not set(plan['materials']).issubset(manifest):
        raise ValueError('Reference lane material split changed')
    axes = ['grid','cutoff','xc_grid','strain','k']
    if {(p['material'],p['axis']) for p in protocol['comparisons']} != {
        (m,axis) for m in manifest for axis in axes}:
        raise ValueError('Incomplete or duplicate full reference protocol')
    selected = [p for p in protocol['comparisons'] if p['material'] in plan['materials']]
    if {(p['material'], p['axis']) for p in selected} != {
        (m, axis) for m in plan['materials'] for axis in axes}:
        raise ValueError('Missing or duplicate registered reference axes')
    for m in plan['materials']:
        if read(Path(manifest[m])/'summary.json')['parameter_vector'] != [0.]*7:
            raise ValueError('Reference density has a nonzero parameter vector')
    return manifest, selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args(); plan = read(args.plan)
    record = args.record.resolve(); record.mkdir(exist_ok=False, parents=True)
    (record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    put(record/'plan.json', plan)

    def status(state, **kw):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(), accepted=False, **kw)
        temp = record/'status.tmp'; temp.write_text(json.dumps(obj, indent=2)+'\n')
        temp.replace(record/'status.json'); print(json.dumps(obj), flush=True)

    try:
        hashes = dict(plan['source_hashes']); verify_hashes(hashes)
        protocol_path = Path(plan['protocol']); protocol = read(protocol_path)
        manifest, pairs = validate_reference(plan, protocol)
        while True:
            states = [{**d, 'state':dependency_state(d)} for d in plan['dependencies']]
            if any(d['state'] == 'missing_controller' for d in states):
                status('dependency_controller_missing', dependencies=states); return
            if all(d['state'] == 'terminal' for d in states): break
            status('waiting_for_predecessors', dependencies=states); time.sleep(30)
        verify_hashes(hashes)
        for material in plan['materials']:
            audit_spectrum(Path(manifest[material]), Path(plan['baseline_spectra'][material]))
        runtimes = {}
        for name in plan['runtime_names']:
            runtime = clone_memory_runtime(Path(plan['campaign'])/name,
                record/(name+'_memory'), Path(plan['equivalence_report']))
            runtimes[name] = runtime
            for path in runtime.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    hashes[str(path)] = sha(path)
        put(record/'execution_source_hashes.json', hashes)
        evidence_root = Path(plan['evidence_record']); output = protocol_path.parent/'runs'
        spectra_root = protocol_path.parent/'spectra'
        masks = record/'masks'; masks.mkdir()
        order = {'grid':0,'cutoff':1,'xc_grid':2,'strain':3,'k':4}
        material_order = {m:i for i,m in enumerate(plan['materials'])}
        pairs.sort(key=lambda p:(material_order[p['material']],order[p['axis']]))
        results = []; bridge_gates = {}

        def save_result(row):
            results.append(row)
            put(evidence_root/(row['material']+'_'+row['axis']+'_result.json'), row)

        for pair in pairs:
            verify_hashes(hashes)
            material, axis = pair['material'], pair['axis']; key = material+'_'+axis
            coarse = Path(pair['coarse_candidate'])
            if (pair['fine_settings'] is None or not (coarse/'summary.json').exists() or
                (coarse != Path(manifest[material]) and not bridge_gates.get(str(coarse), False))):
                save_result(dict(material=material,axis=axis,passed=False,state='required_bridge_or_input_unavailable'))
                continue
            meta, summary = read(coarse/'metadata.json'), read(coarse/'summary.json')
            check_registered_settings(meta, pair['coarse_settings'])
            if summary['parameter_vector'] != [0.]*7 or any(
                name not in summary for name in pair['required_training_observables']):
                raise ValueError('Reference vector or observable scope changed')
            settings = pair['fine_settings']; template = list(plan['worker_templates'][material])
            old_worker = next(x for x in template if x.endswith('/worker.py'))
            old_runtime = Path(old_worker).parents[3]; runtime = runtimes[old_runtime.name]
            provenance = runtime/'numerical_runtime_provenance.json'
            expected_core = read(provenance)['core_sha256']
            original_core = read(old_runtime/'runtime_provenance.json')['core_sha256']
            if meta['code_sha256'] not in [original_core, expected_core]:
                raise ValueError('Unverified coarse implementation')
            reused = pair.get('reuse_fine')
            target = Path(reused) if reused else Path(pair.get('grid_bridge_output') or output/key)
            coarse_hashes = {str(coarse/f):sha(coarse/f) for f in ['metadata.json','summary.json','points.json','mask.npy']}
            spectrum = spectra_root/key
            command = copy.deepcopy(template)
            command[command.index(old_worker)] = str(runtime/'validation/workflows/phase3/worker.py')
            for i, item in enumerate(command):
                if item.startswith('PYTHONPATH='): command[i] = item.replace(str(old_runtime), str(runtime))
                elif item.startswith('D3FT_PHASE3_RESULTS_ROOT='): command[i] = 'D3FT_PHASE3_RESULTS_ROOT='+str(output)
            experiments = []
            if not reused:
                if target.parent != output or target.exists():
                    raise ValueError('Occupied or unregistered reference output')
                mask = np.load(coarse/'mask.npy')
                if axis == 'grid': fine_mask, _ = remap(mask, settings['grid'])
                elif axis == 'cutoff':
                    fine_mask = sphere(summary['a0_A'], settings['grid'], settings['cutoff'])
                    if not miller_set(mask).issubset(miller_set(fine_mask)):
                        raise ValueError('Cutoff refinement removed existing Miller indices')
                else: fine_mask = mask
                support = density_support(fine_mask)
                if not support['full_density_support']:
                    save_result(dict(material=material,axis=axis,passed=False,state='density_support_requires_grid_bridge',preflight=support)); continue
                mask_path = masks/(key+'.npy')
                with mask_path.open('xb') as handle: np.save(handle, fine_mask)
                for flag, value in [('--tag',target.name),('--a0',summary['a0_A']),('--fixed-mask',mask_path),
                    ('--cutoff',settings['cutoff']),('--grid',settings['grid']),('--k',settings['k']),
                    ('--xc-grid-factor',settings['xc_grid']),('--strain-step',settings['strain'])]:
                    command[command.index(flag)+1] = str(value)
                experiments.append(dict(name=key, command=command))
            else:
                check_registered_settings(read(target/'metadata.json'), settings)
                hashes.update({str(target/f):sha(target/f) for f in ['metadata.json','summary.json','points.json','mask.npy']})
            pyindex = next(i for i,x in enumerate(command) if x.endswith('/envs/jrystal/bin/python'))
            spectrum_command = command[:pyindex+1]+['-u',
                str(runtime/'validation/workflows/phase3/check_occupations.py'), str(target),
                '--out',str(spectrum),'--solver','block']
            experiments.append(dict(name=key+'_spectrum',command=spectrum_command))
            sequence_plan = record/(key+'_plan.json'); sequence_record = record/(key+'_sequence')
            put(sequence_plan,dict(jobid=str(plan['job']),gpus=plan['gpus'],stop_on_failure=True,experiments=experiments))
            put(evidence_root/(key+'_inputs.json'),dict(coarse=coarse_hashes, protocol_sha256=sha(protocol_path),
                runtime_provenance=str(provenance), runtime_provenance_sha256=sha(provenance), reused_fine=bool(reused)))
            status('executing_reference_pair',material=material,axis=axis,reused_fine=bool(reused))
            with (record/(key+'_controller.log')).open('x') as log:
                process = subprocess.run([sys.executable,'-u',plan['sequence_driver'],str(sequence_plan),
                    '--record',str(sequence_record)],stdout=log,stderr=subprocess.STDOUT)
            terminal = read(sequence_record/'status.json')
            verify_hashes(hashes); verify_hashes(coarse_hashes)
            if (process.returncode or terminal.get('state') != 'sequence_exited' or
                len(terminal.get('results',[])) != len(experiments) or
                any(x['returncode'] for x in terminal.get('results',[]))):
                save_result(dict(material=material,axis=axis,passed=False,state='worker_or_spectrum_failed',details=terminal)); continue
            try:
                fine_meta = read(target/'metadata.json'); fine_summary = read(target/'summary.json')
                check_registered_settings(fine_meta, settings)
                if fine_summary['parameter_vector'] != [0.]*7 or fine_meta['code_sha256'] not in [original_core,expected_core]:
                    raise ValueError('Fine reference implementation or vector changed')
                if fine_meta['workflow_sha256']['worker.py'] != meta['workflow_sha256']['worker.py']:
                    raise ValueError('Numerical pair worker changed')
                certificate = compare(coarse,target,axis,pair['gate']['lattice_relative'],
                    pair['gate']['elastic_absolute_GPa'],
                    pair['axis_definition']=='coarse FFT grid with physical XC grid held constant')
                spectrum_evidence = audit_spectrum(target,spectrum)
            except (ValueError,KeyError,OSError,AssertionError) as exc:
                save_result(dict(material=material,axis=axis,passed=False,state='scientific_audit_invalid',error=str(exc))); continue
            put(evidence_root/(key+'_certificate.json'),dict(**certificate,spectrum=spectrum_evidence,
                runtime_provenance=str(provenance), runtime_provenance_sha256=sha(provenance),
                equivalence_report_sha256=sha(plan['equivalence_report'])))
            bridge_gates[str(target)] = certificate['passed']
            save_result(dict(material=material,axis=axis,passed=certificate['passed'],
                state='pair_passed' if certificate['passed'] else 'pair_failed',fine=str(target),rows=certificate['rows'],reused_fine=bool(reused)))
            status('reference_pair_scientific_result',material=material,axis=axis,passed=certificate['passed'])
        put(record/'audit.json',dict(evidence_kind='frozen_reference_resolution',rows=results,
            protocol=str(protocol_path),protocol_sha256=sha(protocol_path),accepted=False,
            all_registered_pairs_passed=len(results)==len(pairs) and all(x['passed'] for x in results),
            scope='This resource lane only; all seven materials and five axes are required for complete reference resolution evidence.'))
        status('completed_reference_numerical_lane',passed=sum(x['passed'] for x in results),total=len(pairs))
    except Exception as exc:
        status('reference_numerical_controller_failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__ == '__main__':
    main()
