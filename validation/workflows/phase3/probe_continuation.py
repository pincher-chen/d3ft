"""Real cold/continued strained-state parity check on a saved small configuration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from ml_collections import ConfigDict
from ase.io import read,write
from d3ft.calc.ground_state import calc
from d3ft.calc.forces import force_report
from d3ft.calc.screening import sensitivity
from d3ft.parallel import plain_fft_patch
from continuation import SameBasisContinuation
from residual_report import fused_report

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('source',type=Path)
ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--epochs',type=int,default=1000)
ap.add_argument('--short',type=int,default=500)
a=ap.parse_args()
a.out.mkdir(exist_ok=False)
meta=json.loads((a.source/'metadata.json').read_text())
cfg=ConfigDict(meta['config']);cfg.epoch=a.epochs;cfg.d3ft.parallel.k_pool=False
cfg.crystal_file_path_path=str((a.source/'sample.xyz').resolve())
mask=np.load(a.source/'mask.npy')
cache=SameBasisContinuation(a.short)
report=lambda out:fused_report(out)

def observables(out,r):
    return dict(energy_Ha=float(out.total_energy),pressure_GPa=float(r['pressure_GPa']),
                residual=float(r['variational_residual']),
                forces=np.asarray(force_report(out)['forces_Ha_Bohr']).tolist(),
                dE_dp=np.asarray(sensitivity(out,cfg.d3ft.xc_params.to_dict(),3)).tolist())

with plain_fft_patch():
    initial,r,initial_info=cache.calculate(cfg,None,'initial',a.out,calc,report)
    initial_residual=float(r['variational_residual'])
    del initial,r
    jax.clear_caches()
    atoms=read(a.source/'sample.xyz')
    atoms.set_cell(atoms.cell.array*1.005,scale_atoms=True)
    write(a.out/'strained.xyz',atoms,format='extxyz')
    strained=copy.deepcopy(cfg);strained.crystal_file_path_path=str((a.out/'strained.xyz').resolve())
    warm,r,info=cache.calculate(strained,mask,'strained',a.out,calc,report)
    w=observables(warm,r)
    del warm,r
    jax.clear_caches()
    cold=calc(strained,freq_mask=mask);c=observables(cold,report(cold))
errors={k:float(np.max(np.abs(np.asarray(w[k])-np.asarray(c[k])))) for k in ['energy_Ha','pressure_GPa','forces','dE_dp']}
limits=dict(energy_Ha=1e-9,pressure_GPa=1e-5,forces=1e-8,dE_dp=1e-7)
attempted=info['method'] in ['same_basis_continuation','cold_fallback_after_continuation']
passed=attempted and all(errors[k]<=limits[k] for k in errors) and max(w['residual'],c['residual'])<=1e-10
result=dict(source=str(a.source),initial_residual=initial_residual,continuation=info,warm=w,cold=c,
            errors=errors,limits=limits,passed=passed,
            short_schedule_passed=info['method']=='same_basis_continuation',
            continuation_attempted=attempted,
            source_metadata_sha256=hashlib.sha256((a.source/'metadata.json').read_bytes()).hexdigest(),
            scope='One small strained-cell control; other materials still require per-point stationarity and independent convergence checks')
with (a.out/'parity.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
if not passed:raise SystemExit(1)
