"""One physical tangent for a registered path, or an affine elastic response."""
import argparse,json,sys,time,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import GammaResponse,validated_radial_splines,strain_matrix,HA_EV,dump,sha
from bounded_xc import memory_bounded_xc
from projected_solver import real_inner

def main():
 ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('--mode',choices=['forward','elastic'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 cfg=json.loads(a.protocol.read_text());a.output.mkdir(exist_ok=False)
 def stage(name,**kw):
  row=dict(stage=name,time=datetime.datetime.now().isoformat(),**kw);dump(a.output/'status.json',row);print('DIRECTIONAL_STAGE',json.dumps(row),flush=True)
 try:
  src=Path(cfg['source']);orb=Path(cfg['orbitals']);meta=json.loads((src/'metadata.json').read_text());om=json.loads((orb/'metadata.json').read_text());bands=json.loads((orb/'summary.json').read_text())
  for n,h in om['source_hashes'].items():assert sha(src/n)==h
  assert sha(orb/'orbitals.npy')==bands['saved_orbitals_sha256']
  assert min(bands['vbm_adjacent_separation_Ha'],bands['cbm_adjacent_separation_Ha'])>1e-5
  sys.path.insert(0,str(Path(cfg['runtime'])/'validation/workflows/response_design'))
  from radial_quadrature import local_simpson_patch
  from d3ft.calc.ground_state import calc
  from d3ft.parallel import plain_fft_patch
  from d3ft.calc.xc_tunable import xc_energy_tunable
  from d3ft.calc.stress import ewald_translation_indices
  from jrystal._src.ewald import ewald_coulomb_repulsion
  c=ConfigDict(meta['config']);c.epoch=3;c.crystal_file_path_path=str(src/'input.xyz');c.d3ft.parallel.k_pool=False
  with local_simpson_patch(),plain_fft_patch(k_pool=False),validated_radial_splines(),memory_bounded_xc():
   stage('reconstruct');out=calc(c,freq_mask=np.load(src/'mask.npy'));r=GammaResponse(out,np.load(orb/'orbitals.npy'),bands['eigenvalues_Ha'],cfg['theta0'])
   primal=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[None,:],axis=0)))
   assert primal<=1e-7 and abs(float(r.rho.mean()*r.volume)-96)<.00096
   direction=jnp.asarray(cfg['direction']);external=r.external(direction);jax.block_until_ready(external)
   stage('tangent_solve');start=time.monotonic();u,record=r.solve(-r.project(external),lambda row:stage('iteration',**row))
   if a.mode=='forward':
    b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)))
    derivative=(jnp.dot(r.bare(jnp.zeros(6)),direction)+4*real_inner(b,u))*HA_EV
    dump(a.output/'result.json',dict(gap_eV=bands['gap_eV'],directional_gap_derivative_eV=float(derivative),response_record=record,
      primal_residual_Ha=primal,seconds=time.monotonic()-start,scope='One input path: one true self-consistent tangent, no six-direction cost claim'))
   else:
    inverse=jnp.linalg.inv(jnp.eye(3)+strain_matrix(jnp.asarray(cfg['theta0'])))
    images=jnp.asarray(ewald_translation_indices(out.crystal.cell_vectors,c.ewald_args['ewald_cutoff']))
    kinetic_weights=2*jnp.sum(jnp.abs(r.q)**2,axis=1)
    def energy_at(t):
     diagonal,projectors,local,volume,g=r.geometry_base(t*direction)
     rho=r.rho*r.volume/volume;g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
     vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*jnp.fft.fftn(rho)/g2).at[0,0,0].set(0.)))
     coefficients=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),r.q)
     nl=2*jnp.real(jnp.einsum('ami,ab,bmi->',jnp.conj(coefficients),r.nl,coefficients))/volume
     cell=r.cell@(jnp.eye(3)+strain_matrix(t*direction)@inverse).T
     positions=r.positions@(jnp.eye(3)+strain_matrix(t*direction)@inverse).T
     ionic=ewald_coulomb_repulsion(positions,jnp.asarray(out.pseudopot.valence_charges,dtype=float),g,volume,
       ewald_eta=c.ewald_args['ewald_eta'],ewald_grid=images@cell)
     xc=xc_energy_tunable(rho[None,...],g,volume,c.xc,xc_params=r.xc_params,grid_factor=r.xc_factor)
     return jnp.sum(kinetic_weights*diagonal)+nl+volume/r.n*jnp.sum(rho*(local+.5*vh))+xc+ionic
    def first(t):return jax.jvp(energy_at,(t,),(jnp.array(1.),))[1]
    second=jax.jit(lambda t:jax.jvp(first,(t,),(jnp.array(1.),))[1])
    stage('explicit_elastic_contraction');e0=float(jax.jit(energy_at)(jnp.array(0.)));e1=float(jax.jit(first)(jnp.array(0.)));bare=float(second(jnp.array(0.)))
    parent=json.loads((src/'point.json').read_text());assert abs(e0-parent['energy_Ha'])<1e-6
    pull=np.asarray(strain_matrix(direction)@inverse);stress_slope=parent['volume_bohr3']*float(np.sum(np.asarray(parent['stress_Ha_Bohr3'])*pull))
    assert abs(e1-stress_slope)/float(r.volume)*29421.015697<.02
    correction=4*float(real_inner(u,external));curvature=bare+correction
    plus=json.loads((Path(cfg['fd_plus'])/'point.json').read_text());minus=json.loads((Path(cfg['fd_minus'])/'point.json').read_text());h=cfg['fd_h']
    fd=(plus['energy_Ha']-2*parent['energy_Ha']+minus['energy_Ha'])/h**2
    scale=29421.015697/float(r.volume);ad_GPa=curvature*scale;fd_GPa=fd*scale
    passed=abs(ad_GPa-fd_GPa)<=cfg['elastic_absolute_tolerance_GPa']+cfg['elastic_relative_tolerance']*abs(fd_GPa)
    dump(a.output/'result.json',dict(passed=bool(passed),selfconsistent_modulus_GPa=ad_GPa,reoptimized_FD_modulus_GPa=fd_GPa,
      bare_modulus_GPa=bare*scale,electronic_response_correction_GPa=correction*scale,difference_GPa=abs(ad_GPa-fd_GPa),
      energy_check_difference_Ha=e0-parent['energy_Ha'],first_derivative_stress_difference_Ha=e1-stress_slope,
      response_record=record,primal_residual_Ha=primal,scope='One Γ clamped-ion absolute affine energy-curvature mode; full electronic response, not k4 analytic tensor'))
   stage('completed')
 except Exception as exc:stage('failed',error=str(exc));raise
if __name__=='__main__':main()
