"""Post-hoc exact-functional audit of an owned MgSiN2 Gamma low-spectrum state.

Uses stored independently solved low eigenpairs; performs new Hamiltonian actions,
Ritz diagonalization, integer reoccupation, and full physical energy evaluation.
Does not claim ionic equilibrium or a mathematical global-minimum certificate.
"""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
P=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(P/'validation/workflows/response_design/reverse_response'))
sys.path.insert(0,str(P/'validation/workflows/response_design'))
from gamma_response import GammaResponse,validated_radial_splines
from bounded_xc import memory_bounded_xc
from verification.endpoint_residuals import orbital_residuals,occupation_residual,ritz_residuals
from verification.audit_low_spectrum_streamed import spectral_update
from verification.physical_acceptance import check
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.calc.xc_tunable import xc_energy_tunable
from d3ft.parallel import plain_fft_patch
from jrystal._src import energy
from jrystal.calc.opt_utils import get_ewald_coulomb_repulsion

def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--orbitals',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 def stage(s,**extra):
  dump(a.output/'status.json',dict(stage=s,time=datetime.datetime.now().isoformat(),**extra));print('E2_STAGE',s,extra,flush=True)
 try:
  stage('source_identity');m=json.loads((a.source/'metadata.json').read_text());om=json.loads((a.orbitals/'metadata.json').read_text());sm=json.loads((a.orbitals/'summary.json').read_text())
  for name,h in om['source_hashes'].items():
   if sha(a.source/name)!=h:raise ValueError(f'Source changed: {name}')
  if sha(a.orbitals/'orbitals.npy')!=sm['saved_orbitals_sha256']:raise ValueError('Orbitals changed')
  c=ConfigDict(m['config']);c.epoch=1;c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(a.source/'input.xyz');q=jnp.asarray(np.load(a.orbitals/'orbitals.npy'));values=np.asarray(sm['eigenvalues_Ha']);occ=np.r_[np.full(48,2.),np.zeros(q.shape[1]-48)]
  with local_simpson_patch(),plain_fft_patch(k_pool=False),validated_radial_splines(),memory_bounded_xc():
   stage('geometry_only_reconstruction');out=calc(c,freq_mask=np.load(a.source/'mask.npy'));r=GammaResponse(out,np.asarray(q),values,np.zeros(6));V=float(r.volume)
   stage('exact_Hamiltonian_actions');hq=jnp.concatenate([r.ham(q[:,i:i+4]) for i in range(0,q.shape[1],4)],axis=1);jax.block_until_ready(hq);metrics=orbital_residuals(np.asarray(q[:,:48]),np.asarray(hq[:,:48]),occ[:48]);metrics.update(occupation_residual(values,occ,96.,2.,temperature=0.));metrics.update(ritz_residuals(q,hq,values))
   matrix=np.asarray(q.conj().T@hq);ev,u=np.linalg.eigh((matrix+matrix.conj().T)/2);qn=q@jnp.asarray(u);hqn=hq@jnp.asarray(u);pn=r.projectors
   def state_block(index,vectors,f):
    vectors=jnp.asarray(vectors);f=jnp.asarray(f)
    den=jnp.zeros(r.shape)
    for i in range(0,vectors.shape[1],4):den+=jnp.sum(abs(r.wave(vectors[:,i:i+4]))**2*f[i:i+4],axis=-1)*r.n**2/V
    inner=jnp.einsum('bmg,gi->bmi',jnp.conj(pn),vectors)
    nl=jnp.real(jnp.einsum('bmi,bc,cmi,i->',jnp.conj(inner),r.nl,inner,f))/V
    kin=jnp.sum(abs(vectors)**2*r.diagonal[:,None]*f[None,:])
    return dict(density=np.asarray(den),kinetic_Ha=float(kin),nonlocal_Ha=float(nl),entropy_dimensionless=0.)
   ew=float(get_ewald_coulomb_repulsion(c,charges=np.asarray(out.pseudopot.valence_charges,float)));local=r.geometry_base(jnp.zeros(6))[2]
   def physical_free_energy(den,reduction):
    den=jnp.asarray(den);rec=jnp.fft.fftn(den[None,...],axes=(-3,-2,-1))
    hart=energy.hartree(rec,r.g,V);loc=jnp.sum(den*local)*V/r.n
    xc=xc_energy_tunable(den[None,...],r.g,V,c.xc,xc_params=c.d3ft.get('xc_params'),grid_factor=c.d3ft.get('xc_grid_factor',1))
    return float(hart+loc+xc+ew)+reduction['kinetic_Ha']+reduction['nonlocal_Ha']
   stage('spectral_reoccupation_full_energy');old=np.load(a.source/'density.npy')[0];point=json.loads((a.source/'point.json').read_text());spectral=spectral_update([(0,qn,hqn,ev,occ)],state_block=state_block,physical_free_energy=physical_free_energy,old_density=old,old_free_energy=point['energy_Ha'],volume=V,electrons=96.,expected_indices=[0]);np.save(a.output/'reoccupied_density.npy',spectral.pop('density'));metrics.update(spectral);metrics['charge_error']=abs(float(old.mean()*V)-96.)
   acceptance=check(metrics);dump(a.output/'audit.json',dict(source=str(a.source),orbitals=str(a.orbitals),metrics=metrics,acceptance=acceptance,prior_low_spectrum_source_hash=sha(a.orbitals/'summary.json'),scope='New exact-functional actions and spectral reoccupation of previously independently solved own response low states; no fresh global eigensolve, no structural-equilibrium claim'))
   stage('completed',passed=acceptance['passed']);print('E2_AUDIT',json.dumps(acceptance),flush=True)
 except Exception as e:stage('failed',error=str(e));raise
if __name__=='__main__':main()
