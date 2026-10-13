"""Fill the three remaining materials' frozen-v3 two-step FD checks.

Each plan waits for an existing lane, keeps the original worker and physical
settings, and checks only the already frozen target rows. No allocation is made.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--launch',action='store_true')
    args=ap.parse_args();c=args.campaign.resolve();out=args.out.resolve()
    protocol=read(c/'protocol.json');manifest=read(protocol['manifest']);jac=read(protocol['jacobian'])
    if sha(protocol['manifest'])!=protocol['manifest_sha256'] or sha(protocol['jacobian'])!=protocol['jacobian_sha256']:
        raise ValueError('Frozen-v3 baseline changed')
    source_plan=c/'actual_trials/all_t00_v3/sequence_plan.json'
    templates={e['command'][e['command'].index('--material')+1]:e['command']
        for e in read(source_plan)['experiments'] if '--material' in e['command']}
    launches=read(c/'additional_fd_launches.json')
    specifications=[('SiC',[.16,.32],250212,[0,1],'BN_beta_v3_fd_sequence'),
        ('AlP',[.04,.08],247061,[0,1],'BN_kappa_v3_fd_sequence'),
        ('AlAs',[.04,.08],250285,list(range(8)),'GaP_theta_d_v3_fd_warm2000_sequence')]
    out.mkdir(parents=True,exist_ok=False);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    descriptors=[]
    for material,alphas,job,gpus,parent_name in specifications:
        parent=next(e for e in launches if Path(e['record']).name==parent_name and not e.get('superseded_by'))
        os.kill(parent['controller_pid'],0)
        base=Path(manifest[material]);meta=read(base/'metadata.json');summary=read(base/'summary.json')
        targets=sorted(r['observable'] for r in jac['rows'] if r['material']==material)
        gradients={r['observable']:r for r in jac['rows'] if r['material']==material}
        signals=[]
        for key in targets:
            gradient=gradients[key]['gradient'][0]*gradients[key]['reference']
            budget=1e-4*summary['a0_A'] if key=='a0_A' else .1
            ratio=abs(2*alphas[0]*.1*gradient)/budget
            if ratio<=1:raise ValueError('Registered smaller step cannot resolve '+material+'/'+key)
            signals.append(dict(observable=key,linear_small_step_signal_to_budget=ratio))
        experiments=[];pairs=[]
        for alpha in alphas:
            pair={'alpha':alpha}
            for name,sign in [('plus',1),('minus',-1)]:
                tag=f'{material}_v3_theta_s_a{round(alpha*1000):03d}_{name}_{job}g'+''.join(map(str,gpus))
                vector=list(summary['parameter_vector']);vector[0]+=sign*alpha*.1
                cmd=copy.deepcopy(templates[material])
                for i,item in enumerate(cmd):
                    if item.startswith('--jobid='):cmd[i]='--jobid='+str(job)
                    elif item.startswith('--cpus-per-task='):cmd[i]='--cpus-per-task='+str(8*len(gpus))
                    elif item.startswith('--gres=gpu:'):cmd[i]='--gres=gpu:'+str(8 if len(gpus)==8 else 4)
                    elif item.startswith('CUDA_VISIBLE_DEVICES='):cmd[i]='CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus))
                    elif item.startswith('D3FT_PHASE3_RESULTS_ROOT='):cmd[i]='D3FT_PHASE3_RESULTS_ROOT='+str(c/'runs')
                    elif item.startswith('JAX_COMPILATION_CACHE_DIR='):cmd[i]='JAX_COMPILATION_CACHE_DIR=/tmp/phase3_v3_exact_'+str(job)
                for flag,value in [('--tag',tag),('--vector',json.dumps(vector)),('--a0',summary['a0_A'])]:
                    cmd[cmd.index(flag)+1]=str(value)
                worker=Path(next(x for x in cmd if x.endswith('/worker.py')));runtime=worker.parents[3]
                hashes={str(p.relative_to(runtime)):sha(p) for p in (runtime/'d3ft').rglob('*.py')}
                if hashes!=meta['code_sha256'] or sha(worker)!=sha(base/'worker_snapshot.py'):
                    raise ValueError('Original worker or frozen core mismatch')
                target=c/'runs'/tag
                if target.exists():raise ValueError('FD tag already occupied')
                pair[name]=str(target);experiments.append(dict(name=tag,command=cmd))
            pairs.append(pair)
        plan=dict(jobid=str(job),gpus=gpus,stop_on_failure=True,after_record=parent['record'],after_pid=parent['controller_pid'],
            experiments=experiments,protocol=dict(frozen_at=datetime.datetime.now().isoformat(),base=str(base),
                base_metadata_sha256=sha(base/'metadata.json'),parameter_index=0,parameter_name='theta_s',parameter_scale=.1,
                alphas=alphas,pairs=pairs,relative_agreement_tolerance=.02,
                original_resolution_budgets=dict(a0_relative=1e-4,elastic_GPa=.1),required_observables=targets,
                required_observables_source=protocol['jacobian'],required_observables_source_sha256=protocol['jacobian_sha256'],
                linear_signal_design=signals,runtime_provenance=str(runtime/'runtime_provenance.json'),
                scope=f'Two-step theta_s check of every existing {material} training target. Diagnostic B for AlP is retained but not certified; no missing C44 target or full matrix FD claim.'))
        plan_path=out/(material+'_theta_s_plan.json');put(plan_path,plan)
        descriptors.append(dict(material=material,job=job,gpus=gpus,plan=str(plan_path),plan_sha256=sha(plan_path),
            record=str(out/(material+'_theta_s_sequence')),audit_record=str(out/(material+'_theta_s_audit'))))
    put(out/'preparation.json',dict(input_sha256={str(source_plan):sha(source_plan),protocol['manifest']:protocol['manifest_sha256'],
        protocol['jacobian']:protocol['jacobian_sha256']},experiments=descriptors,registered_endpoints=12,
        computations_launched=0,scope='Prospective raw-worker plans; linear signal predictions are not FD results.'))
    if not args.launch:
        print(json.dumps(dict(registered_endpoints=12,computations_launched=0)));return
    # There must be no active candidate whose frozen registry would be changed.
    old=read(c/'actual_trials/all_t00_v3/execution/status.json')
    recovery=read(c/'Si_theta_p_timeout_recovery/status.json')
    if old.get('state')!='fd_dependency_failed' or recovery.get('state')!='running_timeout_retry':
        raise ValueError('Candidate registry cannot be extended in this state; no new controllers launched')
    for d in descriptors:
        driver=Path(__file__).with_name('run_gpu_sequence.py')
        with (out/(d['material']+'_controller.log')).open('x') as log:
            p=subprocess.Popen([sys.executable,'-u',str(driver),d['plan'],'--record',d['record']],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d.update(controller_pid=p.pid,time=datetime.datetime.now().isoformat())
        launch=out/(d['material']+'_launch.json');put(launch,d)
        checker=Path(__file__).with_name('audit_frozen_targets_fd_sequence.py')
        with (out/(d['material']+'_audit_controller.log')).open('x') as log:
            a=subprocess.Popen([sys.executable,'-u',str(checker),str(launch),'--record',d['audit_record']],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d.update(audit_controller_pid=a.pid)
        launches=read(c/'additional_fd_launches.json');launches.append(d)
        tmp=c/'additional_fd_launches.tmp';tmp.write_text(json.dumps(launches,indent=2));tmp.replace(c/'additional_fd_launches.json')
    registry_path=c/'fd_gate_registry.json';registry=read(registry_path)
    put(out/'prior_fd_gate_registry.json',registry)
    registry['acceptance_files'] += [str(Path(d['audit_record'])/'acceptance.json') for d in descriptors]
    registry['updated']=datetime.datetime.now().isoformat()
    registry['scope']+=' Three additional two-step material checks are required before the recovered initial trial.'
    tmp=c/'fd_gate_registry.tmp';tmp.write_text(json.dumps(registry,indent=2));tmp.replace(registry_path)
    put(out/'launches.json',descriptors)
    print(json.dumps(dict(conditional_sequences_launched=3,registered_new_endpoints=12,
        scientific_results_completed=0)))


if __name__=='__main__':
    main()
