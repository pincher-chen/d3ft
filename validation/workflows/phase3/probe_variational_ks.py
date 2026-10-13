"""Validate the actual KS action against the unconstrained SCF energy gradient."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.calc.implicit import _kwargs_from
from d3ft.calc.stress import free_energy_at_strain
from d3ft.parallel import plain_fft_patch
from variational_ks import operator_factory

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--factors',help='JSON list of XC grid factors; default checks 1 and 2')
ap.add_argument('--base-only',action='store_true',help='Keep source parameters at every chosen grid factor')
a=ap.parse_args();a.out.mkdir(exist_ok=False)
m=json.loads((a.source/'metadata.json').read_text());cfg=ConfigDict(m['config'])
cfg.epoch=3;cfg.d3ft.parallel.k_pool=False
points=json.loads((a.source/'points.json').read_text())
eq=[p for p in points if p['label'].startswith('eq_')]
record=eq[-1] if eq else points[-1]
cfg.crystal_file_path_path=str((a.source/(record['label']+'.xyz')).resolve())
mask=np.load(a.source/'mask.npy')
rows=[]
factors=json.loads(a.factors) if a.factors else [1,2]
if not factors or any(type(f) is not int or not 1<=f<=8 for f in factors):raise ValueError('Invalid XC factors')
with plain_fft_patch():
    for index,factor in enumerate(factors):
        modified=not a.base_only and index>0
        c=copy.deepcopy(cfg);c.d3ft.xc_grid_factor=factor
        if modified:
            c.d3ft.theta_nonlocal_l=[.02,-.01,.015]
            for name,entries in c.d3ft.xc_params.items():
                for key in entries:entries[key]*=1.03
        out=calc(c,freq_mask=mask)
        factory=operator_factory(out)
        kw=_kwargs_from(out);pw=kw.pop('params_pw');occ=kw.pop('params_occ')
        grad=jnp.conj(jax.grad(lambda p:free_energy_at_strain(jnp.zeros((3,3)),p,occ,**kw))(pw))
        occupations=np.asarray(out.get_occupation(occ,out.final_eps))
        errors=[]
        for ik in [0,len(out.k_frac)-1]:
            indices=np.where(occupations[0,ik]>0)[0]
            # np indexing uses a separate band axis to avoid advanced-index axis reordering.
            coefficients=pw[0,ik][:,indices]
            op,_=factory(ik)
            action=jax.jit(jax.vmap(op,in_axes=1,out_axes=1))(coefficients)
            expected=grad[0,ik][:,indices]/(2*occupations[0,ik,indices][None,:])
            errors.append(float(jnp.max(jnp.abs(action-expected))))
        rows.append(dict(xc_factor=factor,modified_parameters=modified,max_operator_error=max(errors)))
        del out,grad,factory,op
        jax.clear_caches()
result=dict(rows=rows,passed=all(x['max_operator_error']<1e-9 for x in rows),
            source=str(a.source.resolve()),geometry_label=record['label'],reconstruction_epochs=3,
            source_metadata_sha256=hashlib.sha256((a.source/'metadata.json').read_bytes()).hexdigest(),
            checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='KS action versus unconstrained SCF energy gradient at the reconstructed three-epoch state; '
                  'not a saved-density gap or global ground-state certificate')
with (a.out/'identity.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(1)
