"""Run actual-loss LM iterations and final whole-material holdout evaluations.

Requires completed baseline inputs. This driver does not certify numerical
convergence or experimental uncertainty. It never accepts a linear prediction.
Each evaluation owns a new worker tag and preserves its complete stdout/stderr.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from fit_step import dataset, SCALE, MATERIALS
from equilibrium_jacobian import derive

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RESULT=ROOT/'validation/results/phase3_20260908'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def require_saved_density(folder):
    folder=Path(folder)
    points=json.loads((folder/'points.json').read_text())
    equilibrium=[p for p in points if p['label'].startswith('eq_')]
    if not equilibrium:raise ValueError(f'{folder}: missing equilibrium density record')
    point=equilibrium[-1]
    if not np.isfinite(point['residual']) or point['residual']>1e-6:
        raise ValueError(f'{folder}: nonstationary equilibrium density')
    if 'density_file' not in point or 'density_sha256' not in point:
        raise ValueError(f'{folder}: baseline needs a saved stationary density')
    if hashlib.sha256((folder/point['density_file']).read_bytes()).hexdigest()!=point['density_sha256']:
        raise ValueError(f'{folder}: saved density hash mismatch')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest',type=Path);ap.add_argument('--tag',required=True)
    ap.add_argument('--jobid',required=True)
    ap.add_argument('--gpu-groups',default='0,1;2,3')
    ap.add_argument('--holdout',action='append',default=[])
    ap.add_argument('--external',action='append',default=['GaAs'])
    ap.add_argument('--max-trials',type=int,default=20)
    ap.add_argument('--continuation-epochs',type=int,default=1000)
    ap.add_argument('--loss-rtol',type=float,default=1e-3)
    ap.add_argument('--skip-spectra-for-diagnostics',action='store_true')
    a=ap.parse_args()
    if not a.tag.replace('_','').isalnum():raise ValueError('Use an alphanumeric/underscore tag')
    groups=[list(map(int,s.split(','))) for s in a.gpu_groups.split(';')]
    flat=[x for g in groups for x in g]
    if not groups or any(not g for g in groups) or len(set(flat))!=len(flat):
        raise ValueError('GPU groups must be nonempty and disjoint')
    if a.max_trials<1 or a.loss_rtol<=0 or a.continuation_epochs<0:
        raise ValueError('Invalid iteration controls')
    original={m:str(Path(p).resolve()) for m,p in json.loads(a.manifest.read_text()).items()}
    if (set(a.holdout)|set(a.external))-set(MATERIALS):raise ValueError('Unknown material')
    train=sorted({m for m,c in MATERIALS.items() if c['role'].startswith('train')}-set(a.holdout))
    if set(a.external)&set(train):raise ValueError('External validation materials cannot be in training')
    held=sorted((set(a.holdout)|set(a.external))-set(train))
    if set(held)-set(original):raise ValueError('Missing holdout baselines')
    initial_vector,_,_,_=dataset(a.manifest,a.holdout)
    for material in held:
        folder=Path(original[material]);derive(folder)
        summary=json.loads((folder/'summary.json').read_text())
        if summary['material']!=material or not np.array_equal(summary['parameter_vector'],initial_vector):
            raise ValueError('Holdout baseline uses different material or parameters')
    if not a.skip_spectra_for_diagnostics:
        for material in train+held:require_saved_density(original[material])
    # Check the allocation's current device state; never share an occupied GPU
    # with an unrelated process merely because --overlap permits another step.
    query=subprocess.check_output(['srun','--jobid='+a.jobid,'--overlap','-N1','-n1',
        '--cpus-per-task=1','nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in query.strip().splitlines()}
    if any(g not in memory or memory[g]>100 for g in flat):
        raise RuntimeError('Requested GPUs are unavailable or occupied; no campaign jobs launched')
    work=RESULT/'campaigns'/a.tag;work.mkdir(parents=True,exist_ok=False)
    write(work/'initial_manifest.json',original)
    write(work/'arguments.json',vars(a)|{'manifest':str(a.manifest.resolve())})
    write(work/'workflow_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    current=original.copy();manifest=work/'initial_manifest.json'
    damping=1.;radius=1.;history=[];accepted_count=0;stop='trial_limit'

    def spectrum(folder,tag,group):
        spectral=work/'spectra'/tag
        command=['srun','--jobid='+a.jobid,'--overlap','-N1','-n1',
            '--cpus-per-task='+str(8*len(group)),
            'env','CUDA_VISIBLE_DEVICES='+','.join(map(str,group)),'JAX_PLATFORMS=cuda',
            'bash',str(ROOT/'validation/phase3_audit_20260908/run_env.sh'),sys.executable,'-u',
            str(HERE/'check_occupations.py'),str(folder),'--out',str(spectral),'--solver','block']
        write(work/(tag+'_spectrum_command.json'),command)
        with (work/(tag+'_spectrum.log')).open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((spectral/'summary.json').read_text())

    def evaluate(material,vector,source,tag,group):
        folder=Path(source);meta=json.loads((folder/'metadata.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        cfg=meta['config'];args=meta['arguments']
        cmd=['srun','--jobid='+a.jobid,'--overlap','-N1','-n1','--cpus-per-task='+str(8*len(group)),
            'env','CUDA_VISIBLE_DEVICES='+','.join(map(str,group)),'JAX_PLATFORMS=cuda',
            'bash',str(ROOT/'validation/phase3_audit_20260908/run_env.sh'),sys.executable,'-u',
            str(HERE/'worker.py'),'--material',material,'--mode','evaluate','--tag',tag,
            '--vector',json.dumps(vector),'--a0',str(summary['a0_A']),
            '--grid',str(cfg['grid_sizes']),'--cutoff',str(cfg['cutoff_energy']),
            '--k',str(cfg['k_grid_sizes']),'--empty',str(cfg['empty_bands']),
            '--epochs',str(cfg['epoch']),'--fixed-mask',str(folder/'mask.npy'),
            '--strain-step',str(args.get('strain_step',.01)),
            '--xc-grid-factor',str(cfg['d3ft'].get('xc_grid_factor',1)),
            '--asr-force-floor',str(args.get('asr_force_floor',0.)),
            '--pressure-tol','1e-7','--fused-report']
        if len(group)>1:cmd.append('--kpool')
        if 'C44_GPa' in summary:cmd.append('--shear')
        if a.continuation_epochs:
            cmd+=['--continuation-epochs',str(min(a.continuation_epochs,cfg['epoch']-1))]
        write(work/(tag+'_command.json'),cmd)
        with (work/(tag+'.log')).open('x') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        dest=RESULT/tag
        if not (dest/'summary.json').exists():raise RuntimeError('Worker exited without a final result')
        if not a.skip_spectra_for_diagnostics:
            spectrum(dest,tag,group)
        return material,str(dest)

    def batch(materials,vector,sources,prefix):
        # One serial lane per disjoint GPU group; separate materials run concurrently.
        def lane(index):
            rows=[]
            for material in materials[index::len(groups)]:
                rows.append(evaluate(material,vector,sources[material],prefix+'_'+material,groups[index]))
            return rows
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures=[pool.submit(lane,i) for i in range(len(groups))]
            rows=[row for future in futures for row in future.result()]
        return dict(rows)

    initial_spectra={}
    if not a.skip_spectra_for_diagnostics:
        materials=sorted(set(train+held))
        def initial_lane(index):
            return {m:spectrum(original[m],a.tag+'_initial_'+m,groups[index])
                    for m in materials[index::len(groups)]}
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures=[pool.submit(initial_lane,i) for i in range(len(groups))]
            for future in futures:initial_spectra.update(future.result())
        write(work/'initial_spectra.json',initial_spectra)
        invalid=[m for m,s in initial_spectra.items() if not s['mesh_fixed_occupations_passed']]
        if invalid:
            raise RuntimeError(f'Initial fixed occupations invalid; no proposals evaluated: {invalid}')

    for trial in range(a.max_trials):
        prefix=f'{a.tag}_t{trial:02d}'
        proposal=work/(prefix+'_proposal.json')
        options=[x for h in a.holdout for x in ['--holdout',h]]
        subprocess.run([sys.executable,str(HERE/'fit_step.py'),str(manifest),*options,
            '--damping',str(damping),'--radius',str(radius),'--out',str(proposal)],check=True)
        prop=json.loads(proposal.read_text());step=np.array(prop['candidate_vector'])-prop['starting_vector']
        if np.linalg.norm(step/SCALE)<1e-5:
            stop='small_trial_step_requires_gradient_review';break
        start=time.time()
        candidate=current|batch(train,prop['candidate_vector'],current,prefix)
        candidate_path=work/(prefix+'_manifest.json');write(candidate_path,candidate)
        acceptance=work/(prefix+'_acceptance.json')
        invalid=[]
        if not a.skip_spectra_for_diagnostics:
            for material in train:
                spectral=json.loads((work/'spectra'/(prefix+'_'+material)/'summary.json').read_text())
                if not spectral['mesh_fixed_occupations_passed']:invalid.append(material)
        if invalid:
            write(acceptance,dict(accepted=False,reason='fixed_occupations_invalid_on_SCF_mesh',materials=invalid))
        else:
            subprocess.run([sys.executable,str(HERE/'fit_step.py'),str(manifest),*options,
                '--candidate',str(candidate_path),'--proposal',str(proposal),'--out',str(acceptance)],check=True)
        acc=json.loads(acceptance.read_text());acc.update(trial=trial,seconds=time.time()-start)
        history.append(acc)
        if acc['accepted']:
            current=candidate;manifest=candidate_path;accepted_count+=1
            damping=max(damping/2,1e-6);radius=min(radius*1.5,2.)
            if acc['actual_reduction']/max(acc['loss_before'],1e-30)<a.loss_rtol:
                stop='small_actual_loss_reduction';break
        else:
            damping*=4;radius*=.5
    vector,rows,residual,jacobian=dataset(manifest,a.holdout)
    held_outputs=batch(held,vector.tolist(),original,a.tag+'_held') if held else {}
    held_spectra={m:json.loads((work/'spectra'/(a.tag+'_held_'+m)/'summary.json').read_text())
                 for m in held} if not a.skip_spectra_for_diagnostics else {}
    write(work/'final_manifest.json',current|held_outputs)
    write(work/'summary.json',dict(stop_reason=stop,accepted_steps=accepted_count,
        trial_history=history,final_parameter_vector=vector.tolist(),training_loss=float(residual@residual),
        training_rows=rows,holdout_materials=held,holdout_outputs=held_outputs,
        spectra_checked=not a.skip_spectra_for_diagnostics,holdout_spectra=held_spectra,
        initial_spectra=initial_spectra,
        optimization_stopped_by_tolerance=stop=='small_actual_loss_reduction',
        scope='Actual nonlinear fit and independent holdout evaluations; numerical convergence and uncertainty require separate acceptance'))
    print('CAMPAIGN_DONE',str(work),flush=True)


if __name__=='__main__':main()
