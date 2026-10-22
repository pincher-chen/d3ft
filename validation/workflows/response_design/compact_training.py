"""Bounded two-parameter XC calibration with actual full DFT re-solves.

No proposal is accepted from a linear prediction. Training is Si/BN a0 and B;
SiC is evaluated after stopping and never used for proposal or acceptance.
Run manifests freeze hardware, basis, references and software before launch.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def proposal(residual, jacobian, vector, scales, radius, damping, bounds):
    """LM in scaled coordinates, projected onto registered physical bounds."""
    r, j = np.asarray(residual), np.asarray(jacobian)
    scale = np.asarray(scales)
    js = j * scale
    z = np.linalg.solve(js.T @ js + damping * np.eye(len(scale)), -js.T @ r)
    z *= min(1., radius / max(np.linalg.norm(z), 1e-30))
    new = np.clip(np.asarray(vector) + scale*z, -np.asarray(bounds), np.asarray(bounds))
    pred = r + j @ (new - vector)
    return new, float(r@r - pred@pred)


def decision(before, after, predicted_reduction, ratio_min=.1):
    actual = before - after
    ratio = actual / predicted_reduction if predicted_reduction > 0 else None
    accepted = bool(np.isfinite(after) and actual > 0 and ratio is not None and ratio > ratio_min)
    return dict(accepted=accepted, actual_reduction=float(actual), rho=ratio)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('protocol',type=Path)
    a=ap.parse_args(); cfg=read(a.protocol); root=Path(cfg['output'])
    root.mkdir(exist_ok=False)
    sys.path.insert(0,str(Path(cfg['runtime'])/'validation/workflows/phase3'))
    from equilibrium_jacobian import derive
    status=root/'status.json'; started=time.monotonic()
    def stage(name,**kwargs):
        row=dict(stage=name,time=datetime.datetime.now().isoformat(),pid=os.getpid(),**kwargs)
        dump(status,row); print('TRAIN_STAGE',json.dumps(row),flush=True)
    def vector7(x): return [0.,0.,0.,0.,0.,float(x[0]),float(x[1])]
    def execute(material,x,tag,devices,parent=None):
        spec=cfg['materials'][material]
        command=list(cfg['worker_command_prefix'])
        assignments={'CUDA_VISIBLE_DEVICES':','.join(map(str,devices)),
                     'D3FT_PHASE3_RESULTS_ROOT':str(root/'runs'),
                     'TMPDIR':f'/dev/shm/d3ft_response_design_20261001_compact_{tag}',
                     'JAX_COMPILATION_CACHE_DIR':f'/dev/shm/d3ft_response_design_20261001_compact_{tag}/jax_cache',
                     'PYTHONPYCACHEPREFIX':f'/dev/shm/d3ft_response_design_20261001_compact_{tag}/python_cache'}
        # env is part of the prefix; append task overrides immediately before bash.
        split=command.index('/bin/bash'); command[split:split]=[f'{k}={v}' for k,v in assignments.items()]
        command += [str(Path(cfg['runtime'])/'validation/workflows/phase3/worker.py'),
            '--material',material,'--vector',json.dumps(vector7(x)), '--tag',tag,
            '--mode','evaluate','--grid',str(spec['grid']),'--cutoff',str(spec['cutoff']),
            '--k',str(spec['k']),'--epochs',str(cfg['epochs']),
            '--xc-grid-factor',str(spec['xc_grid_factor']),
            '--strain-step',str(spec['strain_step']),'--a0',str(parent['a0_A'] if parent else spec['initial_a0']),
            '--asr-force-floor','1e-6','--fused-report']
        if len(devices)>1: command.append('--kpool')
        if parent: command += ['--fixed-mask',str(root/'runs'/parent['tag']/'mask.npy')]
        dump(root/(tag+'_command.json'),command)
        with (root/(tag+'.log')).open('x') as logfile:
            run=subprocess.run(command,stdout=logfile,stderr=subprocess.STDOUT,timeout=cfg['point_timeout_seconds'])
        if run.returncode: raise RuntimeError(f'{tag}: worker RC {run.returncode}; inspect immutable log')
        folder=root/'runs'/tag
        result=read(folder/'summary.json'); derived=derive(folder)
        points=read(folder/'points.json')
        if result['equilibrium_residual']>1e-6 or max(row['residual'] for row in points)>1e-6:
            raise ValueError('Nonstationary training candidate')
        if not np.allclose(result['parameter_vector'],vector7(x),rtol=0,atol=0):
            raise ValueError('Parameter mismatch')
        if parent:
            old=root/'runs'/parent['tag']
            if not np.array_equal(np.load(folder/'mask.npy'),np.load(old/'mask.npy')):
                raise ValueError('Candidate changed fixed Miller basis')
            ma,mb=read(folder/'metadata.json'),read(old/'metadata.json')
            if ma['code_sha256']!=mb['code_sha256'] or ma['pseudopotentials']!=mb['pseudopotentials']:
                raise ValueError('Candidate changed source or pseudopotentials')
        grad_a=np.asarray(derived['dlogV_equilibrium_dp'])*result['a0_A']/3
        result.update(tag=tag,derivatives={'a0_A':grad_a[[5,6]].tolist(),
            'B_GPa':np.asarray(derived['dB_equilibrium_dp'])[[5,6]].tolist()},
            path=str(folder),hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest()
                                   for n in ['summary.json','metadata.json','mask.npy','points.json']})
        dump(root/(tag+'_validated.json'),result)
        return result
    def training(x,index,parents=None):
        stage('training_evaluation',trial=index,delta_kappa_mu=np.asarray(x).tolist())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={mat:pool.submit(execute,mat,x,f'trial{index}_{mat}',cfg['training_gpu_map'][mat],
                                    parents[mat] if parents else None) for mat in cfg['train_materials']}
            return {mat:f.result() for mat,f in futures.items()}
    def dataset(rows):
        residual=[];jac=[]
        for mat in cfg['train_materials']:
            for key in cfg['observables']:
                target=cfg['materials'][mat]['references'][key]
                weight=cfg['materials'][mat]['relative_weight']*abs(target)
                residual.append((rows[mat][key]-target)/weight)
                jac.append(np.asarray(rows[mat]['derivatives'][key])/weight)
        return np.asarray(residual),np.asarray(jac)
    try:
        stage('baseline_start')
        vector=np.zeros(2); rows=training(vector,0); baseline=rows
        r,j=dataset(rows); loss=float(r@r); radius=cfg['initial_scaled_radius']; damping=cfg['initial_damping']
        history=[dict(trial=0,accepted=True,parameter_delta=vector.tolist(),actual_loss=loss,role='untrained_baseline')]
        dump(root/'history.json',history)
        accepted=0
        for index in range(1,cfg['maximum_trials']+1):
            new,pred=proposal(r,j,vector,cfg['scales'],radius,damping,cfg['absolute_delta_bounds'])
            if pred<=0 or np.max(abs(new-vector))<cfg['minimum_parameter_step']:
                stage('proposal_stopping',trial=index,reason='no positive predicted decrease or tiny update'); break
            candidate=training(new,index,rows); cr,cj=dataset(candidate); actual=float(cr@cr)
            check=decision(loss,actual,pred,cfg['minimum_acceptance_ratio'])
            item=dict(trial=index,parameter_delta=new.tolist(),actual_loss=actual,
                      predicted_reduction=pred,**check)
            history.append(item);dump(root/'history.json',history)
            print('TRAIN_DECISION',json.dumps(item),flush=True)
            if check['accepted']:
                vector,rows,r,j,loss=new,candidate,cr,cj,actual;accepted+=1
                radius=min(cfg['maximum_scaled_radius'],radius*1.5);damping=max(.01,damping/2)
            else: radius/=2;damping*=4
            dump(root/'accepted_model.json',dict(delta_kappa_mu=vector.tolist(),loss=loss,
                accepted_updates=accepted,train_paths={m:rows[m]['path'] for m in rows}))
        # Fit is frozen before any held-out reference or result enters this branch.
        stage('holdout_after_stopping',accepted_updates=accepted,delta_kappa_mu=vector.tolist())
        held=cfg['holdout_material']
        oldheld=execute(held,np.zeros(2),'holdout_PBE',cfg['holdout_gpus'])
        newheld=execute(held,vector,'holdout_learned',cfg['holdout_gpus'],oldheld) if accepted else None
        result=dict(completed=True,accepted_updates=accepted,initial_loss=history[0]['actual_loss'],
            final_loss=loss,delta_kappa_mu=vector.tolist(),training_baseline=baseline,
            training_final=rows,holdout_material=held,holdout_PBE=oldheld,holdout_learned=newheld,
            training_success=accepted>0 and loss<history[0]['actual_loss'],
            total_seconds=time.monotonic()-started,
            scope='Small numerical-model calibration; no wide transfer or full-Jacobian convergence claim; original experimental reference qualifications retained')
        dump(root/'summary.json',result);stage('completed',accepted_updates=accepted)
    except Exception as exc:
        stage('failed',error=str(exc));raise

if __name__=='__main__':main()
