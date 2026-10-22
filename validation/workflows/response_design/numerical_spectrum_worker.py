"""Source-bound full-mesh NSCF spectra for the numerical robustness check."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict

P=Path('${D3FT_ROOT}')
F=P/'validation/results/response_design_20261002_resource_fill/comparison_runtime'
sys.path.insert(0,str(F/'validation/workflows/phase3'))
sys.path.insert(0,str(F/'validation/workflows/response_design'))
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from d3ft import pseudopotential as pp
from d3ft.grid.kpoints import k_vectors
from jrystal._src.grid import g_vectors,proper_grid_size
from variational_ks import operator_factory
from lowest_eigensystem import lowest_block
from radial_quadrature import local_simpson_patch

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('state')
    ap.add_argument('output',type=Path);ap.add_argument('indices');a=ap.parse_args()
    protocol=read(a.protocol);state=protocol['states'][a.state];source=Path(state['source'])
    for n,h in state['source_sha256'].items():
        if sha(source/n)!=h:raise ValueError('Immutable SCF source changed: '+n)
    point=read(source/'point.json');meta=read(source/'metadata.json')
    if not point['stationarity_pass'] or point['residual']>1e-6 or point['orthogonality_max_error']>1e-8:
        raise ValueError('SCF stationarity/orthogonality gate failed')
    if abs(point['integrated_valence_density']-96)>.00096:raise ValueError('Charge gate failed')
    if len(jax.devices())!=1:raise ValueError('One GPU per spectrum shard')
    a.output.mkdir(exist_ok=False)
    cfg=ConfigDict(meta['config']);cfg.epoch=3;cfg.crystal_file_path_path=str(source/'input.xyz')
    cfg.d3ft.parallel.k_pool=False;kgrid=cfg.k_grid_sizes;cfg.k_grid_sizes=1
    mask=np.load(source/'mask.npy');indices=json.loads(a.indices)
    with local_simpson_patch(),plain_fft_patch(k_pool=False):
        out=calc(cfg,freq_mask=mask)
        target=k_vectors(out.crystal.cell_vectors,proper_grid_size(kgrid),'gamma')
        g=g_vectors(out.crystal.cell_vectors,np.asarray(mask.shape))
        out.beta_kmax=pp.kmax_for_grid(g,target)
        out.beta_tables=pp.build_tables(out.pseudopot,out.beta_kmax)
        out.k_frac=np.asarray(protocol['k_fractional']);cfg.k_grid_sizes=kgrid
        out.density_grid=jnp.asarray(np.load(source/'density.npy'))
        charge=float(np.asarray(out.density_grid).mean()*float(out.crystal.vol))
        if abs(charge-96)>.00096:raise ValueError('Reconstructed charge gate failed')
        factory=operator_factory(out)
        dump(a.output/'metadata.json',dict(protocol_sha256=sha(a.protocol),worker_sha256=sha(__file__),
             source=state,indices=indices,occupied_bands=48,empty_bands=2,
             solver_sha256=sha(F/'validation/workflows/phase3/lowest_eigensystem.py'),
             scope='All registered k6 points at own stationary density; no force/stress/ASR or continuous-BZ certificate'))
        rows=[]
        for i in indices:
            t=time.monotonic();operator,dim=factory(i)
            values,residuals=lowest_block(operator,dim,50,i)
            if not np.isfinite(values).all() or not np.isfinite(residuals).all() or max(residuals)>1e-7:
                raise ValueError('Eigenpair gate failed at point '+str(i))
            row=dict(index=i,k_frac=out.k_frac[i].tolist(),eigenvalues_Ha=values.tolist(),
              residuals_Ha=residuals.tolist(),vbm_Ha=float(values[47]),cbm_Ha=float(values[48]),
              seconds=time.monotonic()-t)
            dump(a.output/f'k_{i:04d}.json',row);rows.append(row)
            print('NUMERICAL_BAND_POINT',json.dumps(dict(index=i,seconds=row['seconds'],max_residual_Ha=float(max(residuals)))),flush=True)
            del operator;jax.clear_caches();gc.collect()
        dump(a.output/'summary.json',dict(completed=True,total_points=len(rows),indices=indices,
             max_residual_Ha=max(max(r['residuals_Ha']) for r in rows)))

if __name__=='__main__':main()
