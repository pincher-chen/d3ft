"""Independent matrix-free eigsh on the accepted density; no optimizer calls."""
import hashlib,json,os,sys,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,eigsh
from d3ft import pseudopotential as pp_module
from d3ft.pseudopotential.quadrature import attach_upf_metadata
from d3ft.spectrum.ks_matvec import make_ks_hvp
from d3ft.grid.kpoints import k_vectors
from d3ft.parallel import plain_fft_patch
from jrystal.calc.opt_utils import create_crystal,create_freq_mask,create_grids,create_pseudopotential
from jrystal._src.grid import proper_grid_size

jax.config.update('jax_enable_x64',True)
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',choices=['Al','Na','Si','SiC','GaAs'],required=True)
p.add_argument('--state',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--lane',type=int,default=0)
p.add_argument('--lanes',type=int,default=1)
a=p.parse_args()
assert 0<=a.lane<a.lanes
REPO=Path(__file__).resolve().parents[2]
material=a.case
C=json.loads((REPO/'examples/cases.json').read_text())[material]
M=dict(grid=C['grid'],charge_e={'Al':12,'Na':2,'Si':8,'SiC':8,'GaAs':18}[material],bands_solved=40,bands_compared=C['represented_bands'],NLCC=material=='Na')
GLOBAL=dict(ncv=120,eigsh_tol=1e-11,maxiter=3000,maximum_Ritz_residual_Ha=1e-8)
P=json.loads((REPO/'Figures/Figure_ground_state_bands/data'/material/'path.json').read_text())
lane=a.lane;points=list(range(lane,P['points'],a.lanes))
assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
R=a.output.resolve()
if a.lanes==1:R.mkdir(parents=True,exist_ok=False)
else:R.mkdir(parents=True,exist_ok=True)
(R/'raw'/material/'candidate').mkdir(parents=True,exist_ok=True)
(R/'metadata').mkdir(exist_ok=True)
state=a.state.resolve()
cfg=ConfigDict(json.loads((state/'config.json').read_text()))
assert cfg.cutoff_energy==C['cutoff_Ha'] and cfg.grid_sizes==C['grid'] and cfg.k_grid_sizes==C['kmesh'] and cfg.xc==C['xc']
# Retain exact SCF geometry, frequency mask, PP tables and Hamiltonian settings.
crystal=create_crystal(cfg)
mask=np.asarray(create_freq_mask(cfg),dtype=bool)
gvec,_,_=create_grids(cfg)
scf_k=jnp.asarray(k_vectors(crystal.cell_vectors,proper_grid_size(cfg.k_grid_sizes),'gamma'))
pp=create_pseudopotential(cfg)
attach_upf_metadata(pp,cfg.pseudopotential_file_dir)
assert float(np.sum(pp.valence_charges))==float(M['charge_e'])
cores=getattr(pp,'core_charge_grid',None)
core=None
if cores is not None and any(c is not None for c in cores):
    assert material=='Na' and M['NLCC']
    idx=[i for i,c in enumerate(cores) if c is not None]
    core=pp_module.core_charge_real(jnp.asarray(np.asarray(crystal.positions)[idx]),gvec,
        [pp.r_grid[i] for i in idx],[cores[i] for i in idx],crystal.vol,
        r_ab=[pp.r_ab[i] for i in idx],density_cutoff_Ha=4.0*float(cfg.cutoff_energy))
with np.load(state/'final_state.npz',allow_pickle=False) as z:rho=jnp.asarray(z['density_grid'])
assert rho.shape==(1,M['grid'],M['grid'],M['grid'])
charge=float(jnp.sum(rho)*crystal.vol/M['grid']**3)
assert abs(charge-M['charge_e'])<1e-8
grid_kmax=float(np.max(np.linalg.norm(np.asarray(gvec),axis=-1)))
vloc=pp_module.potential_local_reciprocal(crystal.positions,gvec,pp.r_grid,
           pp.local_potential_grid,pp.local_potential_charge,crystal.vol,kmax=grid_kmax)
beta_kmax=pp_module.kmax_for_grid(gvec,scf_k)
tables=pp_module.build_tables(pp,beta_kmax)
nl=pp_module.scaled_d_matrix(pp,theta_nonlocal=float(cfg.d3ft.get('theta_nonlocal',0.0)),
                           theta_nonlocal_l=cfg.d3ft.get('theta_nonlocal_l',None))
ng=int(mask.sum());nb=M['bands_solved']

def raw_action(x,k,pn):
    hop,_=make_ks_hvp(rho,vloc,pn,gvec,k,crystal.vol,mask,xc=cfg.xc,
                      core_density_grid=core,nl_d_matrix=nl)
    return hop(x)

# Only the postprocessing operator is compiled. Reuse the executable at each k.
action_gpu=jax.jit(raw_action)
batch_gpu=jax.jit(jax.vmap(raw_action,in_axes=(1,None,None),out_axes=1))
records=[]
with plain_fft_patch():
 for ik in points:
    start=time.perf_counter()
    k=jnp.asarray(P['fractional_calculation'][ik])@jnp.asarray(crystal.reciprocal_vectors)
    gk=pp_module.gk_vectors(gvec,k[None,:],mask)
    beta=pp_module.beta_grid(pp.r_grid,pp.nonlocal_beta_grid,pp.nonlocal_angular_momentum,
                             gk,beta_kmax,tables=tables)
    pr=pp_module.projectors_reciprocal(crystal.positions,gk,pp.nonlocal_angular_momentum,beta)
    pn=jax.tree.map(lambda x:x[0],pr)
    calls=[0]
    def action(x):
        calls[0]+=1
        return np.asarray(action_gpu(jnp.asarray(x,dtype=jnp.complex128),k,pn))
    rng=np.random.default_rng(92000+ik)
    x=rng.normal(size=ng)+1j*rng.normal(size=ng)
    y=rng.normal(size=ng)+1j*rng.normal(size=ng)
    x/=np.linalg.norm(x);y/=np.linalg.norm(y)
    herm=float(abs(np.vdot(x,action(y))-np.vdot(action(x),y)))
    assert herm<1e-9,('Non-Hermitian operator',ik,herm)
    operator=LinearOperator((ng,ng),matvec=action,dtype=np.complex128)
    vals,vecs=eigsh(operator,k=nb,which='SA',v0=x,ncv=GLOBAL['ncv'],
                     tol=GLOBAL['eigsh_tol'],maxiter=GLOBAL['maxiter'])
    # Same Na FigureS1 recipe: restore degenerate-subspace orthogonality and Ritz.
    q=np.linalg.qr(vecs)[0]
    hq=np.asarray(batch_gpu(jnp.asarray(q),k,pn))
    projected=q.conj().T@hq
    vals,rotation=np.linalg.eigh((projected+projected.conj().T)/2)
    vecs=q@rotation;hv=hq@rotation
    residuals=np.linalg.norm(hv-vecs*vals[None,:],axis=0)
    orth=float(np.max(abs(vecs.conj().T@vecs-np.eye(nb))))
    assert np.max(residuals)<=GLOBAL['maximum_Ritz_residual_Ha'] and orth<=1e-8,('Unresolved spectrum',ik,float(max(residuals)),orth)
    assert len(vals)==40 and vals[-1]-vals[M['bands_compared']-1]>1e-4,('Insufficient empty-state buffer',ik)
    record=dict(material=material,point=ik,ng=ng,bands_solved=40,matvec_calls=calls[0],
                 hermiticity_probe_error_Ha=herm,Ritz_residuals_Ha=residuals.tolist(),
                 orthonormality_max=orth,buffer_above_highest_compared_Ha=float(vals[-1]-vals[M['bands_compared']-1]),
                 seconds=time.perf_counter()-start,source_charge_e=charge,
                 source_tree_sha256=os.environ['JCTC_CANDIDATE_TREE_SHA256'])
    np.savez_compressed(R/'raw'/material/'candidate'/f'point_{ik:03d}.npz',energies_Ha=vals,
          Ritz_residuals_Ha=residuals,k_fractional=P['fractional_calculation'][ik],
          cell_bohr=np.asarray(crystal.cell_vectors))
    (R/'raw'/material/'candidate'/f'point_{ik:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
    records.append(record)
    print('BANDS_CANDIDATE_POINT',json.dumps({k:v for k,v in record.items() if k!='Ritz_residuals_Ha'}),flush=True)
(R/'metadata'/f'{material}_lane{lane}_completed.json').write_text(json.dumps(dict(complete=True,material=material,lane=lane,points=points,records=records),indent=2)+'\n')
