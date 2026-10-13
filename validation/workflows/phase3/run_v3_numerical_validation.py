"""Execute prospective candidate resolution pairs after real trial evaluation.

Preserves the registered vector, full material split and eight-GPU hardware.
Every refined equilibrium receives a fresh full-mesh spectrum. Scientific pair
failures stay visible; dependent bridges cannot silently pass or be skipped.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

from audit_trial import audit_spectrum
from check_elastic_convergence import compare, miller_set
from embed_mask import remap
from numerical_geometry import sphere, density_support

ACTIVE_RECORD = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)


def validate_complete_split(trial):
    proposal = read(trial['proposal'])
    candidate = read(trial['candidate'])
    if set(candidate) != set(proposal['input_fingerprints']) or set(trial['training_materials']) != set(candidate):
        raise ValueError('Incomplete numerical validation material split')
    return proposal, candidate


def verify_hashes(fingerprints):
    for path, digest in fingerprints.items():
        if sha(path) != digest:
            raise ValueError('Frozen numerical dependency changed: '+path)


def check_registered_settings(meta, expected):
    cfg, args = meta['config'], meta['arguments']
    actual = dict(cutoff=cfg['cutoff_energy'], grid=cfg['grid_sizes'],
                  k=cfg['k_grid_sizes'], xc_grid=cfg['d3ft'].get('xc_grid_factor', 1),
                  strain=args['strain_step'])
    if actual != expected:
        raise ValueError(f'Numerical settings differ from registration: {actual} != {expected}')


def check_protocol_registration(registered, executable):
    for key in ['candidate_vector', 'holdouts']:
        if registered[key] != executable[key]:
            raise ValueError('Recovered numerical protocol changed '+key)
    keys = ['material', 'axis', 'coarse_settings', 'fine_settings', 'axis_definition',
            'required_training_observables', 'gate', 'basis_policy', 'local_integral', 'force_floor']
    def descriptors(protocol):
        return [{k: p[k] for k in keys} for p in protocol['comparisons']]
    if descriptors(registered) != descriptors(executable):
        raise ValueError('Numerical protocol changed after observing coarse results')


def clone_memory_runtime(old, new, evidence_path):
    evidence = read(evidence_path)
    if not evidence['passed'] or not all(c['exact_elementwise_equal'] for c in evidence['cases']):
        raise ValueError('Memory implementation equivalence is not verified')
    verify_hashes(evidence['input_files'])
    source = next(Path(p) for p,h in evidence['input_files'].items() if h == evidence['new_sha256'])
    old_hashes = {str(p.relative_to(old)):sha(p) for p in (old/'d3ft').rglob('*.py')}
    if old_hashes != read(old/'runtime_provenance.json')['core_sha256']:
        raise ValueError('Frozen-v3 core differs from its provenance')
    if old_hashes[evidence['path']] != evidence['old_sha256']:
        raise ValueError('Unexpected frozen preconditioner version')
    shutil.copytree(old, new, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(source, new/evidence['path'])
    new_hashes = {str(p.relative_to(new)):sha(p) for p in (new/'d3ft').rglob('*.py')}
    if set(old_hashes) != set(new_hashes) or {p for p in old_hashes if old_hashes[p] != new_hashes[p]} != {evidence['path']}:
        raise ValueError('Unexpected memory-runtime core change')
    provenance = dict(parent=str(old), parent_provenance_sha256=sha(old/'runtime_provenance.json'),
        core_exactly_matches_v3=False, core_sha256=new_hashes, changed_source=evidence['path'],
        equivalence_report=str(evidence_path), equivalence_sha256=sha(evidence_path),
        scope='Only the independently compared preconditioner value/order preserving change. Frozen-v3 physical model retained; full-solver equivalence is not claimed.')
    # Replace the inherited exact-core claim in this derived copy only.
    (new/'runtime_provenance.json').write_text(json.dumps(provenance, indent=2))
    put(new/'numerical_runtime_provenance.json', provenance)
    return new


def main():
    global ACTIVE_RECORD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trial_directory', type=Path)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args()
    campaign = args.campaign.resolve(); record = args.record.resolve()
    record.mkdir(parents=True, exist_ok=False)
    ACTIVE_RECORD = record
    (record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dependencies = ['check_elastic_convergence.py','numerical_geometry.py','embed_mask.py',
        'audit_trial.py','equilibrium_jacobian.py','force_acceptance.py','fit_step.py',
        'materials.json','prepare_v3_numerical_protocol.py','run_gpu_sequence.py']
    fingerprints = {str(Path(__file__).resolve()):sha(Path(__file__))}
    for name in dependencies:
        (record/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        fingerprints[str(Path(__file__).with_name(name).resolve())] = sha(Path(__file__).with_name(name))
    evidence_path=campaign/'preconditioner_equivalence_v2/report.json'
    fingerprints[str(evidence_path)] = sha(evidence_path)
    fingerprints.update(read(evidence_path)['input_files'])
    for name in ['frozen_runtime','frozen_runtime_GaP']:
        for path in (campaign/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                fingerprints[str(path)] = sha(path)
    put(record/'preparation_source_hashes.json', fingerprints)
    def status(state, **kw):
        obj = dict(state=state, time=datetime.datetime.now().isoformat(), **kw)
        tmp=record/'status.tmp'; tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(record/'status.json')
        print(json.dumps(obj),flush=True)
    directory = args.trial_directory.resolve()
    prepare=Path(__file__).with_name('prepare_v3_numerical_protocol.py')
    registration_directory=record/'registered_protocol'
    subprocess.run([sys.executable,str(prepare),str(directory/'trial.json'),
        '--output',str(registration_directory)],check=True)
    registration=read(registration_directory/'protocol.json')
    fingerprints.update(registration['input_sha256'])
    put(record/'registered_input_hashes.json',fingerprints)
    status('waiting_for_complete_actual_trial', trial=str(directory))
    while True:
        verify_hashes(fingerprints)
        trial = read(directory/'trial.json')
        validate_complete_split(trial)
        execution = directory/'execution'
        state = read(execution/'status.json') if (execution/'status.json').exists() else {}
        if (execution/'actual_trial.json').exists():
            actual=read(execution/'actual_trial.json')
            if not actual.get('actual_evaluation_complete') or not actual.get('nominal_actual_loss_gate'):
                status('actual_trial_not_eligible', actual=actual); return
            break
        if state.get('state') == 'fd_dependency_failed':
            recovery = campaign/'Si_theta_p_timeout_recovery/status.json'
            if recovery.exists():
                recovered = read(recovery)
                if recovered.get('state') == 'fd_recovery_passed_and_conditional_trial_rewired':
                    directory = Path(recovered['new_trial'])
                    status('following_registered_recovered_trial', trial=str(directory))
                    continue
                if recovered.get('state') in ['waiting_for_original_terminal_result','running_timeout_retry']:
                    launch=read(campaign/'Si_theta_p_timeout_recovery_launch.json')
                    try: os.kill(launch['controller_pid'],0)
                    except ProcessLookupError:
                        status('recovery_controller_missing_no_restart'); return
                    time.sleep(15); continue
            status('upstream_fd_dependency_failed', details=state); return
        if state.get('state') in ['fd_scientific_gate_failed','frozen_source_changed','candidate_worker_failed']:
            status('upstream_scientific_or_worker_failure',details=state); return
        launch_path=directory/'launch.json'
        if launch_path.exists():
            try: os.kill(read(launch_path)['controller_pid'],0)
            except ProcessLookupError:
                status('actual_trial_controller_missing_without_complete_result'); return
        time.sleep(15)
    proposal, candidate = validate_complete_split(trial)
    verify_hashes(trial['preparation_source_hashes'])
    protocol_directory = directory/'numerical_protocol_executable'
    subprocess.run([sys.executable,str(prepare),
        str(directory/'trial.json'),'--output',str(protocol_directory)],check=True)
    protocol_path=protocol_directory/'protocol.json'; protocol=read(protocol_path)
    verify_hashes(protocol['input_sha256'])
    check_protocol_registration(registration,protocol)
    for material,path in candidate.items():
        if read(Path(path)/'summary.json')['parameter_vector'] != protocol['candidate_vector']:
            raise ValueError('Candidate numerical protocol vector mismatch')
        audit_spectrum(Path(path),Path(read(trial['spectra'])[material]))
    # Only the independently tested mask-before-broadcast implementation differs.
    runtimes={}
    for name in ['frozen_runtime','frozen_runtime_GaP']:
        old=campaign/name; new=record/(name+'_memory')
        runtimes[name]=clone_memory_runtime(old,new,evidence_path)
        for path in new.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                fingerprints[str(path)]=sha(path)
    templates={}
    for exp in read(trial['sequence_plan'])['experiments']:
        cmd=exp['command']
        if '--material' in cmd:
            templates[cmd[cmd.index('--material')+1]]=cmd
    output=protocol_directory/'runs'; output.mkdir()
    spectra_root=protocol_directory/'spectra'; spectra_root.mkdir()
    masks=record/'masks'; masks.mkdir()
    driver=Path(__file__).with_name('run_gpu_sequence.py'); rows=[]; bridge_gates={}
    order={'grid':0,'cutoff':1,'xc_grid':2,'strain':3,'k':4}
    material_order={m:i for i,m in enumerate(['Si','BN','SiC','AlP','diamond','AlAs','GaP'])}
    pairs=sorted(protocol['comparisons'],key=lambda p:(material_order[p['material']],order[p['axis']]))
    for pair in pairs:
        verify_hashes(fingerprints)
        material,axis=pair['material'],pair['axis']; key=material+'_'+axis
        coarse=Path(pair['coarse_candidate'])
        if pair['fine_settings'] is None or not (coarse/'summary.json').exists() or (
            coarse not in [Path(p) for p in candidate.values()] and not bridge_gates.get(str(coarse),False)):
            rows.append(dict(material=material,axis=axis,passed=False,state='required_bridge_or_input_unavailable'))
            status('pair_pending_missing_scientific_bridge',material=material,axis=axis); continue
        summary,meta=read(coarse/'summary.json'),read(coarse/'metadata.json')
        check_registered_settings(meta,pair['coarse_settings'])
        if any(k not in summary for k in pair['required_training_observables']):
            raise ValueError('Missing registered training observable')
        if summary['parameter_vector']!=protocol['candidate_vector']:
            raise ValueError('Numerical pair uses wrong vector')
        setting=pair['fine_settings']; mask=np.load(coarse/'mask.npy')
        if axis=='grid':
            fine_mask,_=remap(mask,setting['grid'])
        elif axis=='cutoff':
            fine_mask=sphere(summary['a0_A'],setting['grid'],setting['cutoff'])
            if not miller_set(mask).issubset(miller_set(fine_mask)):
                raise ValueError('Refined cutoff removed existing physical Miller indices')
        else: fine_mask=mask
        support=density_support(fine_mask)
        if not support['full_density_support']:
            rows.append(dict(material=material,axis=axis,passed=False,state='density_support_requires_grid_bridge',preflight=support))
            status('pair_not_launched_density_support_failed',material=material,axis=axis); continue
        mask_path=masks/(key+'.npy')
        with mask_path.open('xb') as f:np.save(f,fine_mask)
        target=Path(pair['grid_bridge_output']) if pair.get('grid_bridge_output') else output/key
        if target.parent!=output or target.exists(): raise ValueError('Unexpected or occupied numerical output path')
        cmd=copy.deepcopy(templates[material]); old_worker=next(x for x in cmd if x.endswith('/worker.py'))
        old_root=Path(old_worker).parents[3]; runtime=runtimes[old_root.name]
        if sha(old_worker)!=sha(runtime/'validation/workflows/phase3/worker.py'):
            raise ValueError('Numerical worker differs from its frozen source')
        worker=str(runtime/'validation/workflows/phase3/worker.py')
        cmd[cmd.index(old_worker)]=worker
        for i,item in enumerate(cmd):
            if item.startswith('PYTHONPATH='):
                cmd[i]=item.replace(str(old_root),str(runtime))
            elif item.startswith('D3FT_PHASE3_RESULTS_ROOT='):
                cmd[i]='D3FT_PHASE3_RESULTS_ROOT='+str(output)
        for flag,value in [('--tag',target.name),('--a0',summary['a0_A']),('--fixed-mask',mask_path),
            ('--cutoff',setting['cutoff']),('--grid',setting['grid']),('--k',setting['k']),
            ('--xc-grid-factor',setting['xc_grid']),('--strain-step',setting['strain'])]:
            cmd[cmd.index(flag)+1]=str(value)
        pyindex=next(i for i,x in enumerate(cmd) if x.endswith('/envs/jrystal/bin/python'))
        spectral=spectra_root/key
        spectrum_cmd=cmd[:pyindex+1]+['-u',str(runtime/'validation/workflows/phase3/check_occupations.py'),
            str(target),'--out',str(spectral),'--solver','block']
        seq_plan=record/(key+'_plan.json');seq_record=record/(key+'_sequence')
        put(seq_plan,dict(jobid='250285',gpus=list(range(8)),stop_on_failure=True,
            experiments=[dict(name=key,command=cmd),dict(name=key+'_spectrum',command=spectrum_cmd)]))
        provenance={str(coarse/file):sha(coarse/file) for file in ['metadata.json','summary.json','points.json','mask.npy']}
        original_core=read(old_root/'runtime_provenance.json')['core_sha256']
        expected_core=read(runtime/'numerical_runtime_provenance.json')['core_sha256']
        if meta['code_sha256'] not in [original_core,expected_core]:
            raise ValueError('Unverified coarse numerical implementation')
        put(record/(key+'_inputs.json'),dict(coarse=provenance,protocol_sha256=sha(protocol_path),
            memory_runtime_provenance_sha256=sha(runtime/'numerical_runtime_provenance.json'),
            mask_sha256=sha(mask_path),density_support=support))
        status('executing_registered_numerical_pair',material=material,axis=axis)
        with (record/(key+'_controller.log')).open('x') as log:
            proc=subprocess.run([sys.executable,'-u',str(driver),str(seq_plan),'--record',str(seq_record)],
                stdout=log,stderr=subprocess.STDOUT)
        terminal=read(seq_record/'status.json')
        verify_hashes(fingerprints)
        verify_hashes(provenance)
        verify_hashes(protocol['input_sha256'])
        if (proc.returncode or terminal.get('state')!='sequence_exited' or len(terminal.get('results',[]))!=2 or
            any(x['returncode'] for x in terminal.get('results',[]))):
            rows.append(dict(material=material,axis=axis,passed=False,state='worker_or_spectrum_failed',details=terminal))
            status('pair_worker_failed',material=material,axis=axis); continue
        fine_meta=read(target/'metadata.json')
        check_registered_settings(fine_meta,setting)
        if any(k not in read(target/'summary.json') for k in pair['required_training_observables']):
            raise ValueError('Fine result is missing registered training observable')
        expected=read(runtime/'numerical_runtime_provenance.json')['core_sha256']
        if fine_meta['code_sha256']!=expected: raise ValueError('Numerical core changed')
        if sha(Path(__file__).with_name('check_elastic_convergence.py'))!=protocol['comparison_script_sha256']:
            raise ValueError('Registered comparison code changed')
        try:
            certificate=compare(coarse,target,axis,pair['gate']['lattice_relative'],pair['gate']['elastic_absolute_GPa'],
                pair['axis_definition']=='coarse FFT grid with physical XC grid held constant')
            spec=audit_spectrum(target,spectral)
        except (ValueError,KeyError,OSError,AssertionError) as exc:
            rows.append(dict(material=material,axis=axis,passed=False,state='scientific_audit_invalid',error=str(exc)))
            put(record/(key+'_result.json'),rows[-1])
            status('pair_scientific_audit_invalid',material=material,axis=axis,error=str(exc))
            continue
        put(record/(key+'_certificate.json'),dict(**certificate,spectrum=spec,
            equivalence_report_sha256=sha(evidence_path)))
        bridge_gates[str(target)]=certificate['passed']
        rows.append(dict(material=material,axis=axis,passed=certificate['passed'],
            state='pair_passed' if certificate['passed'] else 'pair_failed',fine=str(target),rows=certificate['rows']))
        put(record/(key+'_result.json'),rows[-1])
        status('pair_scientific_result',material=material,axis=axis,passed=certificate['passed'])
    result=dict(rows=rows,all_registered_pairs_passed=len(rows)==len(pairs) and all(x['passed'] for x in rows),
        accepted=False,certified_total_error_bounds=False,nonlinear_refit_complete=False,
        trial=str(directory),protocol_sha256=sha(protocol_path),script_sha256=sha(Path(__file__)),
        scope='Actual empirical candidate resolution pairs and fresh spectra only. Failures and missing bridges remain explicit. Total-loss and derivative error justification plus full refit/holdout results remain separate.')
    put(record/'audit.json',result)
    status('completed_registered_numerical_validation',all_registered_pairs_passed=result['all_registered_pairs_passed'],accepted=False)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if ACTIVE_RECORD is not None:
            tmp=ACTIVE_RECORD/'status.tmp'
            tmp.write_text(json.dumps(dict(state='controller_failed',time=datetime.datetime.now().isoformat(),
                error_type=type(exc).__name__,error=str(exc),accepted=False),indent=2))
            tmp.replace(ACTIVE_RECORD/'status.json')
        raise
