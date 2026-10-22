#!/usr/bin/env python3
"""Fixed-density low KS spectrum; independently check occupied subspace and density."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.optimize import brentq
from scipy.special import expit, xlogy
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from d3ft.spectrum.ks_matvec import make_ks_hvp
from d3ft import pseudopotential as pp_module
from jrystal._src import pw, energy
from streamed_components import components as streamed_components
from jrystal._src.grid import g_vectors
from jrystal.calc.opt_utils import create_freq_mask, get_ewald_coulomb_repulsion

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--state',type=Path,required=True)
p.add_argument('--bands',type=int,default=10)
p.add_argument('--tag',default='')
p.add_argument('--spectrum-cache',type=Path)
a=p.parse_args()
state=a.state.resolve()
assert jax.devices()[0].platform=='gpu'
provenance=json.loads((state/'provenance.json').read_text())
assert provenance['arguments']['orbital'] in ('postkp', 'qr')
assert 'raw_v2' in provenance['source']
result=ROOT/'results'/(f'ks_audit_{state.name}_eigsh{a.bands}'+(f'_{a.tag}' if a.tag else ''))
result.mkdir(exist_ok=False)
(result/'audit_snapshot.py').write_bytes(Path(__file__).read_bytes())
cfg=ConfigDict(json.loads((state/'config.json').read_text()))
source_summary=json.loads((state/'summary.json').read_text())
source_temperature=float(source_summary['temperature_Ha'])
if not np.isfinite(source_temperature) or source_temperature <= 0:
    raise ValueError('Invalid physical endpoint temperature')
cfg.smearing=source_temperature
cfg.d3ft.free_energy.temp_init=source_temperature
cfg.epoch=1
cfg.verbose=False
cfg.optimizer_args.learning_rate=0.0
cfg.d3ft.optim.lr_schedule='constant'
cfg.d3ft.occupation.learning_rate=0.0
cfg.convergence_condition=0.0
occ_state={}
pw_state={}
with np.load(state/'final_state.npz') as saved:
    density_saved=saved['density_grid'].copy()
    for key in saved.files:
        if key.startswith(('occ__', 'pw__')):
            target=occ_state if key.startswith('occ__') else pw_state
            path=key.split('__')[1:]
            for component in path[:-1]:
                target=target.setdefault(component,{})
            target[path[-1]]=jnp.asarray(saved[key])
    init={'pw':pw_state if pw_state else jnp.asarray(saved['params_pw']),'occ':occ_state}
mask=create_freq_mask(cfg)
with plain_fft_patch():
    out=calc(cfg,init_params=init,freq_mask=mask)
replay_error=float(np.max(abs(np.asarray(out.density_grid)-density_saved)))
if replay_error>1e-10:
    raise RuntimeError(f'Zero-LR reconstruction changed saved density: {replay_error}')
crystal=out.crystal
pp=out.pseudopot
gvec=g_vectors(crystal.cell_vectors,np.asarray(mask.shape))
kvec=jnp.asarray(out.k_frac)@jnp.asarray(crystal.reciprocal_vectors)
vloc=pp_module.potential_local_reciprocal(crystal.positions,gvec,pp.r_grid,
    pp.local_potential_grid,pp.local_potential_charge,crystal.vol,kmax=out.grid_kmax)
gk=pp_module.gk_vectors(gvec,kvec,mask)
beta=pp_module.beta_grid(pp.r_grid,pp.nonlocal_beta_grid,
    pp.nonlocal_angular_momentum,gk,out.beta_kmax,tables=out.beta_tables)
projectors=pp_module.projectors_reciprocal(crystal.positions,gk,pp.nonlocal_angular_momentum,beta)
coeff=out.make_coeff(out.params_pw)
occ=np.asarray(out.get_occupation(out.params_occ,out.final_eps))
cap=2/occ.shape[1]
T=float(out.temperature)
if abs(T-source_temperature)>1e-14:
    raise RuntimeError(f'Physical temperature mismatch: {T} vs {source_temperature}')
ne=float(occ.sum())
energies=[]
vectors=[]
checks=[]
t0=time.perf_counter()
if a.spectrum_cache:
    cached=json.loads((a.spectrum_cache/'summary.json').read_text())
    assert cached['source_checkpoint_sha256']==hashlib.sha256((state/'final_state.npz').read_bytes()).hexdigest()
    assert cached['audit_temperature_Ha']==source_temperature and cached['bands']==a.bands
    with np.load(a.spectrum_cache/'low_spectrum.npz') as data:
        energies=data['energies_Ha'].copy();vectors=list(data['compact_vectors'].copy())
    checks=cached['checks']
else:
    for ik in range(occ.shape[1]):
        pn=jax.tree.map(lambda x:x[ik],projectors)
        hop,ng=make_ks_hvp(out.density_grid,vloc,pn,gvec,kvec[ik],crystal.vol,mask,
            xc=cfg.xc,core_density_grid=out.core_density_grid,nl_d_matrix=out.nl_d_matrix)
        apply=jax.jit(hop)
        calls=[0]
        def action(x):
            calls[0]+=1
            return np.asarray(apply(jnp.asarray(x,dtype=jnp.complex128)))
        rng=np.random.default_rng(92000+ik)
        x=rng.normal(size=ng)+1j*rng.normal(size=ng)
        y=rng.normal(size=ng)+1j*rng.normal(size=ng)
        x/=np.linalg.norm(x);y/=np.linalg.norm(y)
        herm=float(abs(np.vdot(x,action(y))-np.vdot(action(x),y)))
        if herm>1e-9:
            raise RuntimeError(f'Non-Hermitian operator at k={ik}: {herm}')
        linear=LinearOperator((ng,ng),matvec=action,dtype=np.complex128)
        vals,vecs=eigsh(linear,k=a.bands,which='SA',v0=x,ncv=max(40,3*a.bands),tol=1e-11,maxiter=2000)
        # Complex ARPACK can lose orthogonality inside degenerate eigenspaces.
        # Reorthogonalise the returned subspace and perform a small Ritz solve.
        q=np.linalg.qr(vecs)[0]
        hq=np.asarray(jax.jit(jax.vmap(hop,in_axes=1,out_axes=1))(jnp.asarray(q)))
        projected=q.conj().T@hq
        vals,rotation=np.linalg.eigh((projected+projected.conj().T)/2)
        vecs=q@rotation
        hv=hq@rotation
        residuals=np.linalg.norm(hv-vecs*vals[None,:],axis=0)
        orth=float(np.max(abs(vecs.conj().T@vecs-np.eye(a.bands))))
        if np.max(residuals)>1e-8 or orth>1e-8:
            raise RuntimeError(f'Unresolved eigenspectrum at k={ik}: {residuals}, orth={orth}')
        energies.append(vals);vectors.append(vecs)
        checks.append(dict(k=ik,ng=ng,matvec_calls=calls[0],hermiticity_probe_error_Ha=herm,
            eigen_residuals_Ha=residuals.tolist(),orthonormality_max=orth))
        print(json.dumps(checks[-1]),flush=True)
energies=np.asarray(energies)
# Retain solved compact vectors before any density/energy postprocessing.
np.savez_compressed(result/'eigenspectrum_before_density.npz',energies_Ha=energies,compact_vectors=np.stack(vectors))
(result/'eigenspectrum_before_density.json').write_text(json.dumps(dict(source_checkpoint_sha256=hashlib.sha256((state/'final_state.npz').read_bytes()).hexdigest(),bands=a.bands,temperature_Ha=T,checks=checks),indent=2)+'\n')
mu=brentq(lambda m:np.sum(cap*expit((m-energies)/T))-ne,
    energies.min()-100*T,energies.max()+100*T,xtol=1e-14)
f_new=cap*expit((mu-energies)/T)
rdm_squared=0.0
old_vectors=[]
for ik,u in enumerate(vectors):
    old=np.asarray(coeff[0,ik][:,np.asarray(mask,dtype=bool)].T)
    old_vectors.append(old)
    cross=np.abs(old.conj().T@u)**2
    rdm_squared+=np.sum(occ[0,ik]**2)+np.sum(f_new[ik]**2)-2*np.sum(occ[0,ik,:,None]*f_new[ik,None,:]*cross)
# Avoid full216-k FFT tensors; preserve every k weight and energy expression.
del coeff
old_components, old_density = streamed_components(old_vectors,occ,mask=mask,gvec=gvec,kvec=kvec,
    projectors=projectors,crystal=crystal,out=out,cfg=cfg,vloc=vloc)
new_components, new_density = streamed_components(vectors,f_new[None,...],mask=mask,gvec=gvec,kvec=kvec,
    projectors=projectors,crystal=crystal,out=out,cfg=cfg,vloc=vloc)
stream_density_replay_error=float(np.max(abs(np.asarray(old_density)-density_saved)))
if stream_density_replay_error>1e-10:
    raise RuntimeError(f'Streaming density replay changed saved density: {stream_density_replay_error}')
ew=get_ewald_coulomb_repulsion(cfg,charges=np.asarray(pp.valence_charges,dtype=float))
old_components['ewald_Ha']=float(ew);new_components['ewald_Ha']=float(ew)
e_old=sum(old_components.values());e_new=sum(new_components.values())
def S(f):
    return float(-np.sum(xlogy(f,f/cap)+xlogy(cap-f,(cap-f)/cap)))
F_old=e_old-T*S(occ);F_new=e_new-T*S(f_new)
reported_F=float(source_summary['free_energy_physical_Ha'])
F_replay_error=abs(F_old-reported_F)
F_replay_bound=2*occ.size*float(cfg.eps)*T+1e-8
if F_replay_error>F_replay_bound:
    raise RuntimeError(f'Exact/regularized endpoint replay mismatch: {F_replay_error} > {F_replay_bound}')
rho_difference=np.asarray(new_density)-np.asarray(out.density_grid)
summary=dict(scope='Independently seeded matrix-free ARPACK low spectrum at saved density; one spectral/FD update is a diagnostic, not another converged SCF run.',
    reused_spectrum=str(a.spectrum_cache) if a.spectrum_cache else None,
    source_state=str(state),source_checkpoint_sha256=hashlib.sha256((state/'final_state.npz').read_bytes()).hexdigest(),
    source_tree_sha256=os.environ['JCTC_CANDIDATE_TREE_SHA256'],
    audit_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_provenance_sha256=hashlib.sha256((state/'provenance.json').read_bytes()).hexdigest(),
    source_temperature_Ha=source_temperature,audit_temperature_Ha=T,
    exact_vs_reported_regularized_F_error_Ha=F_replay_error,
    exact_vs_reported_regularized_F_allowed_Ha=F_replay_bound,
    density_replay_max_error=replay_error,stream_density_replay_max_error=stream_density_replay_error,
    density_energy_reduction="One-k blocks with original2/K occupation capacity; complete216-k sum unchanged",bands=a.bands,chemical_potential_Ha=mu,
    maximum_ritz_residual_Ha=max(max(x['eigen_residuals_Ha']) for x in checks),
    top_band_occupation_max=float(np.max(f_new[:,-1])),
    retained_band_tail_indicator_only=True,
    density_matrix_difference_frobenius=float(np.sqrt(max(rdm_squared,0))),
    density_matrix_difference_squared_unclipped=float(rdm_squared),
    spectral_update_density_L1_electrons=float(np.sum(abs(rho_difference))*crystal.vol/np.prod(mask.shape)),
    spectral_update_density_max=float(np.max(abs(rho_difference))),
    old_exact_free_energy_Ha=F_old,spectral_update_exact_free_energy_Ha=F_new,
    old_energy_components_Ha=old_components,new_energy_components_Ha=new_components,
    spectral_update_free_energy_change_Ha=F_new-F_old,
    wall_seconds=time.perf_counter()-t0,device=str(jax.devices()[0]),
    job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),checks=checks)
np.savez_compressed(result/'low_spectrum.npz',energies_Ha=energies,occupation=f_new,
    compact_vectors=np.stack(vectors),density_new=np.asarray(new_density))
(result/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
