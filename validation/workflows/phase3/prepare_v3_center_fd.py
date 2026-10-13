"""Register FD controls at a replayably accepted center, using training materials.

All seven coordinate controls are recomputed at the new vector. Entire held
materials are excluded even from control selection. Preparations are not FD
certificates or completed nonlinear fits; this command does not launch jobs.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import numpy as np

from audit_v3_lm_step import audit, read, sha, verify_hashes
from check_jacobian_resolution import physical_gradients
from fit_step import dataset


FILES = ['summary.json', 'metadata.json', 'points.json', 'mask.npy']
POLICY = dict(step_factors=[1., 2., 4.], required_predicted_signal_ratio=1.,
    relative_agreement_tolerance=.02, a0_relative_signal_budget=1e-4, elastic_signal_budget_GPa=.1,
    scope='Choose before new FD results, using retained training derivatives only. '
          'Two real FD steps remain mandatory; predicted signal never certifies agreement.')
PREFERRED = {0:'Si', 1:'Si', 2:'GaP', 3:'BN', 4:'BN', 5:'BN', 6:'diamond'}
LANES = {'Si':(250212,list(range(4))), 'SiC':(250212,list(range(4))),
         'BN':(247061,list(range(4))), 'AlP':(247061,list(range(4))),
         'diamond':(250285,list(range(8))), 'AlAs':(250285,list(range(8))),
         'GaP':(250285,list(range(8)))}


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)


def accepted_center(path):
    recorded = read(path)
    if recorded.get('accepted') is not True or recorded.get('decision') != 'accepted_real_lm_step':
        raise ValueError('A real fully audited accepted LM center is required')
    if recorded['audit_script_sha256'] != sha(Path(__file__).with_name('audit_v3_lm_step.py')):
        raise ValueError('Acceptance auditor changed')
    verify_hashes(recorded['source_sha256']); inputs = recorded['audit_inputs']
    replay = audit(inputs['trial'], inputs['before_numerical'], inputs['after_numerical'], inputs['fd_index'])
    if replay.get('accepted') is not True or any(recorded.get(k) != v for k,v in replay.items()):
        raise ValueError('Accepted center does not replay from raw evidence')
    return recorded, read(inputs['trial']), read(inputs['fd_index'])


def choose_step(vector, direction, alpha, gradients, observables, a0):
    vector,direction=np.asarray(vector,float),np.asarray(direction,float)
    if (vector.shape!=(7,) or direction.shape!=(7,) or not np.isfinite(vector).all() or
        not np.isfinite(direction).all() or np.count_nonzero(direction)!=1 or alpha<=0 or a0<=0):
        raise ValueError('Invalid center, coordinate or FD step design')
    constants = np.array([.06672455060314922,.031090690869654894,.804,.2195149727645171])
    bounds = np.r_[np.full(3,.5),.5*constants]
    for factor in POLICY['step_factors']:
        small = alpha*factor; large = 2*small
        if np.any(np.abs(vector)+large*np.abs(direction)>bounds):
            continue
        ratios = {key:abs(float(np.asarray(gradients[key])@(2*small*direction)))/
                  (POLICY['a0_relative_signal_budget']*a0 if key=='a0_A' else POLICY['elastic_signal_budget_GPa'])
                  for key in observables}
        if ratios and min(ratios.values())>POLICY['required_predicted_signal_ratio']:
            return dict(small_alpha=small,large_alpha=large,step_factor=factor,signal_ratios=ratios)
    return None


def select_controls(work, vector, original, targets):
    gradients = {m:physical_gradients(path) for m,path in work.items()}
    summaries = {m:read(Path(path)/'summary.json') for m,path in work.items()}
    controls = {}; material_checks = {}
    for case in original['cases']:
        if case['role']!='implementation_control':
            continue
        direction = np.asarray(case['physical_direction'],float)
        if direction.shape!=(7,) or np.count_nonzero(direction)!=1:
            raise ValueError('Original controls must be seven distinct coordinate directions')
        index = int(np.flatnonzero(direction)[0])
        if index in controls:
            raise ValueError('Duplicate original coordinate control')
        options = []
        for material in sorted(work):
            available = sorted(gradients[material])
            # Preserve broad method controls. AlP's diagnostic B remains a
            # diagnostic in its material check, rather than replacing C44 tests.
            if len(available)<3:
                continue
            design = choose_step(vector,direction,case['small_alpha'],gradients[material],available,summaries[material]['a0_A'])
            if design:
                options.append((material,design,available))
        if not options:
            raise ValueError(f'No resolved bounded training-only control design for coordinate {index}')
        preferred = [item for item in options if item[0]==PREFERRED[index]]
        material, design, available = preferred[0] if preferred else max(options,key=lambda item:min(item[1]['signal_ratios'].values()))
        controls[index] = dict(material=material,physical_direction=direction.tolist(),observables=available,**design)
    if set(controls)!=set(range(7)):
        raise ValueError('Original controls do not cover all seven coordinates')
    for material in sorted(work):
        shared = [v for _,v in sorted(controls.items()) if v['material']==material and set(targets[material]).issubset(v['observables'])]
        if shared:
            material_checks[material] = shared[0]
            continue
        old = [case for case in original['cases'] if case['role']=='training_material' and
               read(Path(case['base'])/'summary.json')['material']==material]
        if len(old)!=1:
            raise ValueError('No unique original training-material direction')
        direction = np.asarray(old[0]['physical_direction'],float)
        design = choose_step(vector,direction,old[0]['small_alpha'],gradients[material],targets[material],summaries[material]['a0_A'])
        if design is None:
            raise ValueError('Original material direction cannot resolve the new center within registered step factors')
        material_checks[material] = dict(material=material,physical_direction=direction.tolist(),observables=targets[material],**design)
    return controls,material_checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('acceptance',type=Path);ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    accepted,trial,original = accepted_center(args.acceptance)
    verify_hashes(trial['preparation_source_hashes'])
    campaign=args.campaign.resolve();proposal=read(trial['proposal']);held=proposal['holdouts']
    if read(campaign/'protocol.json')['folds'][trial['fold']]!=held:
        raise ValueError('Whole-material split changed')
    work=read(trial['candidate']);train=set(proposal['input_fingerprints'])
    if set(work)!=train or train.intersection(held):
        raise ValueError('New center must contain every retained material and no held material')
    args.out.mkdir(parents=True,exist_ok=False);out=args.out.resolve()
    if not out.name.replace('_','').isalnum():
        raise ValueError('Output folder must use letters, digits and underscores')
    manifest=out/'center_manifest.json';put(manifest,work)
    vector,rows,_,_=dataset(manifest,held)
    if vector.tolist()!=accepted['candidate_vector']:
        raise ValueError('New FD center differs from accepted vector')
    targets={m:sorted(r['observable'] for r in rows if r['material']==m) for m in sorted(train)}
    controls,material_checks=select_controls(work,vector,original,targets)
    put(out/'policy.json',POLICY);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    templates={e['command'][e['command'].index('--material')+1]:e['command']
               for e in read(trial['sequence_plan'])['experiments'] if '--material' in e['command']}
    unique={};cases=[];descriptors=[]
    for role,designs in [('implementation_control',[controls[i] for i in range(7)]),
                         ('training_material',[material_checks[m] for m in sorted(train)])]:
        for design in designs:
            material=design['material'];base=Path(work[material]);direction=design['physical_direction']
            key=(material,tuple(direction),design['small_alpha'],design['large_alpha'])
            if key not in unique:
                index=int(np.flatnonzero(direction)[0]);job,gpus=LANES[material];experiments=[];pairs=[]
                meta=read(base/'metadata.json');cmd_base=templates[material]
                worker=Path(next(x for x in cmd_base if x.endswith('/worker.py')));runtime=worker.parents[3]
                core={str(p.relative_to(runtime)):sha(p) for p in (runtime/'d3ft').rglob('*.py')}
                if core!=meta['code_sha256'] or sha(worker)!=meta['workflow_sha256']['worker.py']:
                    raise ValueError('Accepted-center worker/core identity changed')
                for source,digest in meta['pseudopotentials'].items():
                    if sha(source)!=digest:raise ValueError('Pseudopotential changed')
                for step,alpha in enumerate([design['small_alpha'],design['large_alpha']]):
                    pair=dict(alpha=alpha)
                    for name,sign in [('plus',1),('minus',-1)]:
                        tag=f'centerfd_{out.name}_{material}_p{index}_s{step}_{name}';cmd=copy.deepcopy(cmd_base)
                        for i,item in enumerate(cmd):
                            if item.startswith('--jobid='):cmd[i]='--jobid='+str(job)
                            elif item.startswith('--cpus-per-task='):cmd[i]='--cpus-per-task='+str(8*len(gpus))
                            elif item.startswith('--gres=gpu:'):cmd[i]='--gres=gpu:'+str(len(gpus))
                            elif item.startswith('CUDA_VISIBLE_DEVICES='):cmd[i]='CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus))
                            elif item.startswith('D3FT_PHASE3_RESULTS_ROOT='):cmd[i]='D3FT_PHASE3_RESULTS_ROOT='+str(out/'runs')
                        for flag,value in [('--tag',tag),('--vector',json.dumps((vector+sign*alpha*np.asarray(direction)).tolist())),
                                           ('--a0',read(base/'summary.json')['a0_A']),('--fixed-mask',base/'mask.npy')]:
                            cmd[cmd.index(flag)+1]=str(value)
                        pair[name]=str(out/'runs'/tag);experiments.append(dict(name=tag,command=cmd))
                    pairs.append(pair)
                name=f'{material}_p{index}';path=out/(name+'_plan.json')
                observable_scope=sorted(set(design['observables'])|set(targets[material]))
                fd_protocol=dict(base=str(base),base_metadata_sha256=sha(base/'metadata.json'),
                    parameter_index=index,parameter_name=read(campaign/'protocol.json')['parameter_names'][index],
                    parameter_scale=direction[index],pairs=pairs,alphas=[design['small_alpha'],design['large_alpha']],
                    required_observables=observable_scope,required_observables_source=str(out/'target_rows.json'),
                    relative_agreement_tolerance=.02,scope='Prospective new-center FD; no held material or scientific PASS.')
                unique[key]=dict(path=path,pairs=pairs,protocol=fd_protocol,experiments=experiments,job=job,gpus=gpus)
                descriptors.append(dict(material=material,parameter_index=index,plan=str(path),
                    record=str(out/(name+'_sequence')),audit_record=str(out/(name+'_audit')),job=job,gpus=gpus))
            case=dict(role=role,base=str(base),physical_direction=direction,small_alpha=design['small_alpha'],
                large_alpha=design['large_alpha'],pairs=unique[key]['pairs'],observables=design['observables'],
                base_input_sha256={str(base/n):sha(base/n) for n in FILES},
                source_protocol=str(unique[key]['path']))
            cases.append(case)
    # One target scope per base/control plan; distinct coordinates at a material
    # can require different diagnostic scopes. Each plan receives its own row file.
    sources={str(args.acceptance.resolve()):sha(args.acceptance),str(Path(__file__).resolve()):sha(__file__)}
    for key,item in unique.items():
        scope=item['protocol']['required_observables'];row_source=item['path'].with_name(item['path'].stem+'_rows.json')
        put(row_source,dict(rows=[dict(material=key[0],observable=obs) for obs in scope]))
        item['protocol'].update(required_observables_source=str(row_source),required_observables_source_sha256=sha(row_source))
        put(item['path'],dict(jobid=str(item['job']),gpus=item['gpus'],stop_on_failure=True,
            experiments=item['experiments'],protocol=item['protocol']))
        sources[str(item['path'])]=sha(item['path']);sources[str(row_source)]=sha(row_source)
    for case in cases:case['source_protocol_sha256']=sha(case['source_protocol'])
    put(out/'fd_evidence_index.json',dict(starting_vector=vector.tolist(),fold=trial['fold'],holdouts=held,
        cases=cases,registered_source_sha256=sources,entrywise_jacobian_fd_certified=False,
        scope='All controls at the accepted center, training-only selection and coverage. No old-center FD reuse or held evaluation.'))
    put(out/'fd_gate_registry.json',dict(starting_vector=vector.tolist(),acceptance_files=[str(Path(d['audit_record'])/'acceptance.json') for d in descriptors],
        scope='Prospective all-coordinate and every-training-material new-center gates; none passed yet.'))
    put(out/'preparation.json',dict(time=datetime.datetime.now().isoformat(),experiments=descriptors,
        registered_endpoints=4*len(unique),computations_launched=0,refit_complete=False,held_materials_evaluated=False))
    print(json.dumps(dict(registered_endpoints=4*len(unique),computations_launched=0,held=held)))


if __name__=='__main__':main()
