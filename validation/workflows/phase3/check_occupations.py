"""Resolve occupied/unoccupied KS edges on the SCF mesh at a saved density.

The operator uses the actual tunable discrete XC energy and scaled KB matrix.
Positive mesh gap is necessary for the fixed occupations used by Phase 3; this
does not certify a gap between mesh points or accuracy versus experiment.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator, eigsh
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from variational_ks import operator_factory
from lowest_eigensystem import lowest_block as lowest


def lowest_arpack(op, dim, count, seed, tolerance=1e-10):
    action=jax.jit(op)
    rng=np.random.default_rng(seed)
    start=rng.normal(size=dim)+1j*rng.normal(size=dim)
    matrix=LinearOperator((dim,dim),matvec=lambda x:np.asarray(action(jnp.asarray(x))),dtype=np.complex128)
    if count>=dim-1:
        raise ValueError('Too few plane waves for an unoccupied-band check')
    vals,vecs=eigsh(matrix,k=count,which='SA',tol=tolerance,v0=start,
                    ncv=min(dim,max(2*count+8,40)),maxiter=10000)
    order=np.argsort(vals);vals=vals[order];vecs=vecs[:,order]
    errors=np.array([np.linalg.norm(matrix@vecs[:,i]-vals[i]*vecs[:,i]) for i in range(count)])
    if not np.isfinite(vals).all() or not np.isfinite(errors).all() or errors.max()>1e-7:
        raise ValueError(f'Unconverged eigensystem: max residual {errors.max()} Ha')
    return vals,errors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--indices',help='JSON list for a diagnostic subset; default checks every SCF k point')
    ap.add_argument('--lightweight-reconstruction',action='store_true',help='Initialize geometry on Gamma only, then restore full-mesh projector tables; always use saved density')
    ap.add_argument('--solver',choices=['block','arpack'],default='block',help='ARPACK retained for diagnostics only: may miss exact degeneracies')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.source/'metadata.json').read_text())
    points=json.loads((a.source/'points.json').read_text())
    eq=[p for p in points if p['label'].startswith('eq_')]
    record=eq[-1] if eq else points[-1]
    if not np.isfinite(record['residual']) or record['residual']>1e-6:
        raise ValueError('Source density is not stationary')
    path=a.source/record['density_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['density_sha256']:
        raise ValueError('Saved density hash mismatch')
    cfg=ConfigDict(meta['config']);cfg.epoch=3
    cfg.crystal_file_path_path=str((a.source/(record['label']+'.xyz')).resolve())
    cfg.d3ft.parallel.k_pool=len(jax.devices())>1
    original_kgrid=cfg.k_grid_sizes
    if a.lightweight_reconstruction:
        if len(jax.devices())!=1:raise ValueError('Lightweight reconstruction requires one visible GPU')
        if cfg.d3ft.kpoints.mesh!='gamma':raise ValueError('Only verified Gamma meshes supported')
        cfg.k_grid_sizes=1
    mask=np.load(a.source/'mask.npy')
    with ExitStack() as stack:
        stack.enter_context(plain_fft_patch(k_pool=cfg.d3ft.parallel.k_pool))
        out=calc(cfg,freq_mask=mask)
        if a.lightweight_reconstruction:
            from d3ft.grid.kpoints import k_vectors
            from jrystal._src.grid import g_vectors, proper_grid_size
            from d3ft import pseudopotential as d3pp
            # Match the original Cartesian-to-fractional round trip and SBT
            # table extent, not the smaller Gamma reconstruction table.
            target_k=k_vectors(out.crystal.cell_vectors,proper_grid_size(original_kgrid),'gamma')
            out.k_frac=np.asarray(target_k)@np.linalg.inv(np.asarray(out.crystal.reciprocal_vectors))
            g=g_vectors(out.crystal.cell_vectors,np.asarray(mask.shape))
            out.beta_kmax=d3pp.kmax_for_grid(g,target_k)
            out.beta_tables=d3pp.build_tables(out.pseudopot,out.beta_kmax)
            cfg.k_grid_sizes=original_kgrid
        # Reconstruction only supplies geometry, PP tables and k mesh. Never
        # use its deliberately unconverged three-step density for spectroscopy.
        density=np.load(path)
        if density.shape!=out.density_grid.shape:raise ValueError('Density grid mismatch')
        out.density_grid=jnp.asarray(density)
        density_electrons=float(density.sum(axis=0).mean()*abs(np.linalg.det(out.crystal.cell_vectors)))
        electrons=float(np.asarray(out.get_occupation(out.params_occ,out.final_eps)).sum())
        nocc=int(round(electrons/2))
        if abs(electrons-2*nocc)>1e-6:raise ValueError('Noninteger restricted electron count')
        # The stored density includes the SCF positivity floor. Count bands from
        # occupations; record its small integrated charge excess without renormalizing.
        charge_budget=1e-5*max(electrons,1.)
        if abs(density_electrons-electrons)>charge_budget:
            raise ValueError('Saved density charge disagrees with electron occupations')
        factory=operator_factory(out)
        indices=json.loads(a.indices) if a.indices else list(range(len(out.k_frac)))
        if not indices or len(set(indices))!=len(indices) or any(i<0 or i>=len(out.k_frac) for i in indices):
            raise ValueError('Invalid k-point subset')
        provenance=dict(source=str(a.source.resolve()),density_sha256=record['density_sha256'],
            source_metadata_sha256=hashlib.sha256((a.source/'metadata.json').read_bytes()).hexdigest(),
            workflow_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),Path(__file__).with_name('variational_ks.py'),Path(__file__).with_name('lowest_eigensystem.py')]},
            solver=a.solver,
            electrons=electrons,density_electrons=density_electrons,charge_budget=charge_budget,
            occupied_bands=nocc,indices=indices,
            mesh=out.k_frac.tolist(),devices=[str(d) for d in jax.devices()])
        provenance['lightweight_reconstruction']=a.lightweight_reconstruction
        (a.out/'metadata.json').write_text(json.dumps(provenance,indent=2))
        rows=[]
        for i in indices:
            t=time.time();op,dim=factory(i)
            vals,errors=(lowest if a.solver=='block' else lowest_arpack)(op,dim,nocc+2,i)
            row=dict(index=i,k_frac=np.asarray(out.k_frac[i]).tolist(),eigenvalues_Ha=vals.tolist(),
                residuals_Ha=errors.tolist(),vbm_Ha=float(vals[nocc-1]),cbm_Ha=float(vals[nocc]),seconds=time.time()-t)
            rows.append(row)
            (a.out/f'k_{i:04d}.json').write_text(json.dumps(row,indent=2))
            print(json.dumps(row),flush=True)
            del op; jax.clear_caches();gc.collect()
        vbm=max(r['vbm_Ha'] for r in rows);cbm=min(r['cbm_Ha'] for r in rows)
        full=set(indices)==set(range(len(out.k_frac)))
        result=dict(vbm_Ha=vbm,cbm_Ha=cbm,gap_eV=(cbm-vbm)*27.211386245988,
            full_mesh=full,solver=a.solver,mesh_fixed_occupations_passed=bool(full and a.solver=='block' and cbm-vbm>2e-7),
            sampled_gap_positive=bool(cbm-vbm>2e-7),max_eigen_residual_Ha=max(max(r['residuals_Ha']) for r in rows),
            scope='SCF mesh only; neither continuous-Brillouin-zone nor experimental gap certificate')
        (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
