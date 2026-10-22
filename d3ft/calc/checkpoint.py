"""Opt-in atomic physical SCF checkpoints, outside all compiled kernels.

These preserve orbitals/occupations and the numerical model. Optimizer moments
are deliberately not serialized: a partial restart is an explicit warm start,
never advertised as an exact continuation of the optimizer trajectory.
"""
import datetime as dt
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np


def _dump(path,value):
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:
        x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    tmp.replace(path)


class _HashingWriter:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def write(self,data):
        n=self.stream.write(data);self.digest.update(memoryview(data)[:n]);return n


class CheckpointBudgetStop(RuntimeError):pass


class SCFCheckpoint:
    def __init__(self,root,config,crystal,mask,k_fractional,ewald,interval=500,max_wall_seconds=None):
        self.started=time.monotonic();self.max_wall_seconds=max_wall_seconds
        self.interval=int(interval)
        if self.interval<1:raise ValueError('Checkpoint interval must be positive')
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.folder=Path(root)/('SCF_'+stamp+'_'+str(os.getpid()))
        self.folder.mkdir(parents=True,exist_ok=False)
        self.last_step=None;self.last=None
        np.save(self.folder/'mask.npy',np.asarray(mask))
        self.model=dict(config=config.to_dict(),cell_bohr=np.asarray(crystal.cell_vectors).tolist(),
            positions_bohr=np.asarray(crystal.positions).tolist(),k_fractional=np.asarray(k_fractional).tolist(),
            mask_sha256=hashlib.sha256((self.folder/'mask.npy').read_bytes()).hexdigest(),
            ewald_Ha=float(ewald),optimizer_state_saved=False,restart_mode='physical warm start; optimizer reset',
            jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'))
        _dump(self.folder/'model.json',self.model)

    def write(self,params,step,energy_history,eps,temp,*,reason):
        if step==self.last_step:
            self.last['reason']=reason;_dump(self.folder/'latest.json',self.last);return
        import jax
        file=self.folder/f'state_{step:06d}.pkl'
        intent=dict(iteration=int(step),epochs_requested=int(self.model['config']['epoch']),
            energy_history_Ha=[float(x) for x in energy_history],eps=float(eps),temperature=float(temp),
            reason=reason,stationarity_certified=False,physical_acceptance_pending=True,
            model='model.json',state_file=file.name,time=dt.datetime.now().astimezone().isoformat())
        _dump(self.folder/f'state_{step:06d}.intent.json',intent)
        host=jax.tree_util.tree_map(np.asarray,jax.device_get(params))
        tmp=file.with_name(file.name+'.tmp.'+str(os.getpid()))
        with tmp.open('xb') as stream:
            writer=_HashingWriter(stream);pickle.dump(host,writer,protocol=5)
            stream.flush();os.fsync(stream.fileno());digest=writer.digest.hexdigest()
        tmp.replace(file)
        self.last=dict(intent,state_sha256=digest,complete=True)
        _dump(self.folder/f'state_{step:06d}.json',self.last)
        _dump(self.folder/'latest.json',self.last)
        self.last_step=step
        print('D3FT_CHECKPOINT '+json.dumps(dict(iteration=step,path=str(file),reason=reason)),flush=True)
        if self.max_wall_seconds is not None and time.monotonic()-self.started>=self.max_wall_seconds:
            raise CheckpointBudgetStop('Infrastructure budget pause after complete atomic checkpoint; not physical failure')

    def density(self,density):
        target=self.folder/'density.npy';tmp=target.with_name(target.name+'.tmp.'+str(os.getpid()))
        with tmp.open('xb') as stream:
            np.save(stream,np.asarray(density));stream.flush();os.fsync(stream.fileno())
        tmp.replace(target)
        _dump(self.folder/'density_complete.json',dict(complete=True,iteration=self.last_step,
            file=target.name,parameters_written_before_density=True))


def create_checkpoint(config,crystal,mask,k_fractional,ewald):
    cfg=config.d3ft.get('checkpoint',{})
    root=cfg.get('root') or os.environ.get('D3FT_CHECKPOINT_ROOT')
    if not root:return None
    return SCFCheckpoint(root,config,crystal,mask,k_fractional,ewald,
        cfg.get('interval',int(os.environ.get('D3FT_CHECKPOINT_INTERVAL','500'))),cfg.get('max_wall_seconds'))
