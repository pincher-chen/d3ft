"""Stream low spectra of an owned saved density, with the exact SCF functional.

No SCF updates, no imported data, no charge fitting. Original optimizer orbital
commutators are unavailable; newly solved Ritz vectors and density/free-energy
fixed-point checks are explicitly reported separately.
"""
import argparse,datetime,hashlib,json,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse.linalg import eigsh,LinearOperator
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
P=Path(__file__).resolve().parents[3];sys.path.insert(0,str(P/'validation/workflows/phase3'));sys.path.insert(0,str(P/'validation/workflows/response_design'))
from d3ft import pseudopotential as pp
from d3ft.calc.xc_tunable import xc_energy_tunable
from d3ft.grid.kpoints import k_vectors
from jrystal.calc.opt_utils import create_crystal,create_pseudopotential,get_ewald_coulomb_repulsion
from jrystal._src.grid import g_vectors
from jrystal._src import energy
from verification.endpoint_residuals import orbital_residuals,occupation_residual,ritz_residuals
from verification.physical_acceptance import check
from variational_ks import operator_factory

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--start',type=int,default=0);ap.add_argument('--merge-from',type=Path);a=ap.parse_args();a.output.mkdir(exist_ok=False);(a.output/'blocks').mkdir()
 start=time.monotonic()
 def stage(s,**kw):dump(a.output/'status.json',dict(time=datetime.datetime.now().isoformat(),stage=s,**kw));print('E2_DENSITY_STAGE',s,kw,flush=True)
 try:
  stage('input_identity');meta=json.loads((a.source/'metadata.json').read_text());points=json.loads((a.source/'points.json').read_text());point=[r for r in points if r['label'].startswith('eq_')][-1];density_path=a.source/point['density_file']
  if sha(density_path)!=point['density_sha256']:raise ValueError('Stored density identity mismatch')
  c=ConfigDict(meta['config']);c.crystal_file_path_path=str(a.source/(point['label']+'.xyz'));c.d3ft.parallel.k_pool=False;cr=create_crystal(c);ps=create_pseudopotential(c)
  if any(x is not None for x in ps.core_charge_grid):raise ValueError('This saved-density lane is registered for non-NLCC calibration states')
  mask=np.load(a.source/'mask.npy');rho=np.load(density_path);shape=mask.shape;N=int(np.prod(shape));V=float(abs(np.linalg.det(np.asarray(cr.cell_vectors))));Ne=float(sum(ps.valence_charges));charge=float(rho.sum(axis=0).mean()*V)
  if abs(charge-Ne)>1e-10:raise ValueError(f'Raw stored charge not conserved on actual cell: {charge}, {Ne}; no normalization allowed')
  if c.d3ft.kpoints.mesh!='gamma':raise ValueError('Unregistered k-mesh convention')
  grid=np.repeat(c.k_grid_sizes,3) if np.ndim(c.k_grid_sizes)==0 else np.asarray(c.k_grid_sizes);kc=k_vectors(cr.cell_vectors,grid,'gamma');Nk=len(kc);kf=np.asarray(kc)@np.linalg.inv(np.asarray(cr.reciprocal_vectors));g=g_vectors(cr.cell_vectors,np.asarray(shape));gridmax=float(np.max(np.linalg.norm(np.asarray(g),axis=-1)));betamax=pp.kmax_for_grid(g,jnp.asarray(kc));tables=pp.build_tables(ps,betamax);D=pp.scaled_d_matrix(ps,theta_nonlocal=float(c.d3ft.get('theta_nonlocal',0.)),theta_nonlocal_l=c.d3ft.get('theta_nonlocal_l'))
  out=SimpleNamespace(config=c,crystal=cr,freq_mask=mask,density_grid=jnp.asarray(rho),core_density_grid=None,pseudopot=ps,nl_d_matrix=D,beta_kmax=betamax,beta_tables=tables,grid_kmax=gridmax,k_frac=kf)
  idx=jnp.asarray(np.flatnonzero(mask.reshape(-1)));ng=len(idx);nocc=int(round(Ne/2));nb=nocc+4;occupation=np.r_[np.full(nocc,2./Nk),np.zeros(nb-nocc)];rec=jnp.fft.fftn(jnp.asarray(rho),axes=(-3,-2,-1));g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.);vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*rec[0]/g2).at[0,0,0].set(0.)))
  local=pp.potential_local_reciprocal(cr.positions,g,ps.r_grid,ps.local_potential_grid,ps.local_potential_charge,V,kmax=gridmax);local_real=jnp.real(jnp.fft.ifftn(local));vx=jax.grad(lambda den:xc_energy_tunable(den,g,V,c.xc,xc_params=c.d3ft.get('xc_params'),grid_factor=c.d3ft.get('xc_grid_factor',1)))(jnp.asarray(rho))[0]*N/V;veff=local_real+vh+vx
  def wave(q):
   full=jnp.zeros((N,q.shape[1]),dtype=q.dtype).at[idx].set(q);return jnp.fft.ifftn(full.reshape(shape+(q.shape[1],)),axes=(0,1,2))
  def action(q,diagonal,projectors):
   q=q.reshape(ng,-1);w=wave(q);l=jnp.fft.fftn(veff[...,None]*w,axes=(0,1,2)).reshape(N,q.shape[1])[idx];inside=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),q);nl=jnp.einsum('bmg,bmi->gi',projectors,jnp.einsum('bc,cmi->bmi',D,inside))/V;return diagonal[:,None]*q+nl+l
  single=jax.jit(lambda x,d,b:action(x[:,None],d,b)[:,0]);batch=jax.jit(action)
  @jax.jit
  def contribution(q,diagonal,projectors):
   den=jnp.sum(abs(wave(q))**2*jnp.asarray(occupation)[None,None,None,:],axis=-1)*N**2/V;kin=jnp.sum(abs(q)**2*diagonal[:,None]*jnp.asarray(occupation)[None,:]);inside=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),q);nl=jnp.real(jnp.einsum('bmi,bc,cmi,i->',jnp.conj(inside),D,inside,jnp.asarray(occupation)))/V;return den,kin,nl
  total_density=np.zeros(shape);kinetic=0.;nonlocal_=0.;records=[];operator_error=None;run_indices=range(a.start,min(a.start+a.limit,Nk) if a.limit else Nk)
  if (not run_indices and not a.merge_from) or a.start<0:raise ValueError('Invalid k range')
  if a.merge_from:run_indices=range(0)
  stage('low_spectrum',Nk=Nk,planned=len(run_indices),basis=ng)
  for ik in run_indices:
   tick=time.monotonic();gk=pp.gk_vectors(g,jnp.asarray(kc[ik:ik+1]),mask);beta=pp.beta_grid(ps.r_grid,ps.nonlocal_beta_grid,ps.nonlocal_angular_momentum,gk,betamax,tables=tables);pn=pp.projectors_reciprocal(cr.positions,gk,ps.nonlocal_angular_momentum,beta)[0];diagonal=jnp.sum(gk[0]**2,axis=-1)/2;rng=np.random.default_rng(20261014+ik);x=rng.normal(size=ng)+1j*rng.normal(size=ng);x/=np.linalg.norm(x)
   if ik==run_indices[0]:
    reference,_=operator_factory(out)(ik);expected=np.asarray(jax.jit(reference)(jnp.asarray(x)));actual=np.asarray(single(jnp.asarray(x),diagonal,pn));operator_error=float(np.max(abs(actual-expected)))
    if operator_error>1e-10:raise ValueError(f'Compact operator disagrees with exact-functional factory: {operator_error}')
   calls=[0]
   def hop(x):calls[0]+=1;return np.asarray(single(jnp.asarray(x),diagonal,pn))
   operator=LinearOperator((ng,ng),matvec=hop,dtype=np.complex128);values,q=eigsh(operator,k=nb,which='SA',ncv=min(ng,max(48,6*nb)),tol=1e-11,maxiter=3000,v0=x);q=np.linalg.qr(q)[0];hq=np.asarray(batch(jnp.asarray(q),diagonal,pn));h=q.conj().T@hq;values,u=np.linalg.eigh((h+h.conj().T)/2);q=q@u;hq=hq@u;ritz=ritz_residuals(q,hq,values)
   if ritz['maximum_ritz_residual_Ha']>1e-8:raise ValueError(f'Ritz gate failed at k{ik}: {ritz}')
   orb=orbital_residuals(q,hq,occupation);part,kin,nl=contribution(jnp.asarray(q),diagonal,pn);total_density+=np.asarray(part);kinetic+=float(kin);nonlocal_+=float(nl);np.savez_compressed(a.output/'blocks'/f'{ik:04d}.npz',vectors=q,actions=hq,eigenvalues_Ha=values,k_fractional=kf[ik],occupations=occupation)
   records.append(dict(index=ik,seconds=time.monotonic()-tick,matvec_calls=calls[0],ritz=ritz,orbital=orb,eigenvalues_Ha=values.tolist(),gap_Ha=float(values[nocc]-values[nocc-1])))
   if ik%10==0 or ik==run_indices[-1]:dump(a.output/'progress.json',dict(done=len(records),start_k=a.start,Nk=Nk,elapsed_seconds=time.monotonic()-start));print('E2_K',ik,'seconds',records[-1]['seconds'],'calls',calls[0],flush=True)
  if a.merge_from:
   stage('merge_full_mesh');seen=set()
   for partfile in sorted(a.merge_from.glob('*/partial.json')):
    part=json.loads(partfile.read_text());folder=partfile.parent
    if part['source']!=str(a.source) or part['Nk']!=Nk or part['density_sha256']!=point['density_sha256']:raise ValueError('Partial physical/source identity mismatch')
    if seen.intersection(part['indices']):raise ValueError('Duplicate k coverage')
    seen.update(part['indices']);total_density+=np.load(folder/part['density']);kinetic+=part['kinetic_Ha'];nonlocal_+=part['nonlocal_Ha'];blockmeta=json.loads((folder/'blocks.json').read_text())
    if blockmeta['operator_identity_max_error_Ha']>1e-10:raise ValueError('Shard operator identity failed')
    records+=blockmeta['records']
   if seen!=set(range(Nk)):raise ValueError(f'Incomplete k coverage: {len(seen)} of {Nk}')
   records.sort(key=lambda r:r['index'])
  np.save(a.output/'reoccupied_density.npy',total_density);dump(a.output/'blocks.json',dict(records=records,operator_identity_max_error_Ha=operator_error))
  if not a.merge_from and len(run_indices)<Nk:
   dump(a.output/'pilot.json',dict(completed=True,passed_operator_and_Ritz=True,blocks=len(records),Nk=Nk,elapsed_seconds=time.monotonic()-start,warm_seconds_per_k=float(np.median([r['seconds'] for r in records[1:]])) if len(records)>1 else None,estimate_full_GPU_hours=Nk*float(np.median([r['seconds'] for r in records[1:]]))/3600 if len(records)>1 else None,accepted=False,scope='Cost pilot only, incomplete mesh'));dump(a.output/'partial.json',dict(source=str(a.source),geometry_label=point['label'],density_sha256=point['density_sha256'],Nk=Nk,indices=list(run_indices),kinetic_Ha=kinetic,nonlocal_Ha=nonlocal_,actual_cell_volume_Bohr3=V,electrons=Ne,recorded_free_energy_Ha=point['energy_Ha'],density='reoccupied_density.npy',passed_operator_and_Ritz=True));stage('completed_partial_spectrum');return
  recnew=jnp.fft.fftn(jnp.asarray(total_density)[None,...],axes=(-3,-2,-1));hart=float(energy.hartree(recnew,g,V));loc=float(jnp.sum(jnp.asarray(total_density)*local_real)*V/N);xc=float(xc_energy_tunable(jnp.asarray(total_density)[None,...],g,V,c.xc,xc_params=c.d3ft.get('xc_params'),grid_factor=c.d3ft.get('xc_grid_factor',1)));ew=float(get_ewald_coulomb_repulsion(c,charges=np.asarray(ps.valence_charges,float)));F=kinetic+nonlocal_+hart+loc+xc+ew
  metrics=dict(charge_error=abs(charge-Ne),orbital_orthonormality_max=max(r['orbital']['orbital_orthonormality_max'] for r in records),orbital_outside_max_Ha=max(r['orbital']['orbital_outside_max_Ha'] for r in records),orbital_commutator_max_Ha=max(r['orbital']['orbital_commutator_max_Ha'] for r in records),occupation_max_error=0.,maximum_ritz_residual_Ha=max(r['ritz']['maximum_ritz_residual_Ha'] for r in records),spectral_update_density_L1_per_electron=float(abs(total_density-rho[0]).mean()*V/Ne),spectral_update_free_energy_change_abs_Ha=abs(F-point['energy_Ha']))
  mesh_gap=min(r['eigenvalues_Ha'][nocc] for r in records)-max(r['eigenvalues_Ha'][nocc-1] for r in records)
  if mesh_gap<=0:raise ValueError('Fixed integer occupations inconsistent with a nonpositive global mesh gap')
  acceptance=check(metrics);dump(a.output/'audit.json',dict(source=str(a.source),geometry_label=point['label'],density_sha256=point['density_sha256'],actual_cell_volume_Bohr3=V,recorded_legacy_volume_Bohr3=point['volume_bohr3'],fresh_free_energy_Ha=F,old_free_energy_Ha=point['energy_Ha'],metrics=metrics,acceptance=acceptance,full_mesh=True,Nk=Nk,elapsed_seconds=time.monotonic()-start,original_optimizer_orbital_commutator=None,scope='Fresh independently solved full-mesh low spectra and density/free-energy fixed-point audit; orbital metrics belong to new Ritz vectors, original optimizer vectors were not saved. No renormalization or refit.'))
  stage('completed',passed=acceptance['passed'])
 except Exception as e:stage('failed',error=str(e));raise
if __name__=='__main__':main()
