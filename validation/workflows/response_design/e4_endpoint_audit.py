"""Fresh endpoint reoptimization + independent Ritz/filling audit, frozen E4 code."""
import argparse,datetime as dt,json,pickle,sys,gc,os
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.protocol.read_text());R=Path(p['runtime']);P=Path(p['project'])
for directory in reversed([R/'validation/workflows/response_design',R/'validation/workflows/response_design/reverse_response',R/'validation/workflows/phase3',Path(__file__).parent]):sys.path.insert(0,str(directory))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from joint_design_dft import GammaDFTBackend
from joint_design import physical_checks,dump
from gamma_response import sha,validated_radial_splines
from bounded_xc import memory_bounded_xc
from fixed_radial_program import fixed_radial_program
from radial_quadrature import local_simpson_patch
from memory_probe import MemoryProbe
from variational_ks import operator_factory
from verification.endpoint_residuals import orbital_residuals,occupation_residual,ritz_residuals
from verification.physical_acceptance import check
from d3ft.calc.ground_state import calc
from d3ft.calc.implicit import _kwargs_from
from d3ft.calc.stress import free_energy_at_strain,ewald_translation_indices
from d3ft.parallel import plain_fft_patch
from d3ft import pseudopotential as pp
from jrystal.calc.opt_utils import create_grids,create_pseudopotential
from compact_k_stream import compact_k_stream_patch

def main():
 a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output).start()
 def stage(name,**kw):
  v=dict(time=dt.datetime.now().astimezone().isoformat(),stage=name,jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),**kw);dump(a.output/'status.json',v);print('E4_AUDIT',json.dumps(v),flush=True)
 try:
  for name,digest in p['source_hashes'].items():
   if sha(name)!=digest:raise ValueError('Changed endpoint source '+name)
  root=Path(p['E4_root']);cfg=json.loads((root/'protocol.json').read_text());trajectory=json.loads((root/'design/trajectory.json').read_text());theta=trajectory[-1]['theta'];old=trajectory[-1]['readout'];reference=trajectory[0]['readout']
  cfg.update(epochs=p['epochs'],backend_status_file=str(a.output/'backend_status.json'));backend=GammaDFTBackend(cfg)
  # Reconstruct the original *initial-geometry* radial tables, as the design
  # retained them throughout. Never retabulate at the designed endpoint.
  with local_simpson_patch(),validated_radial_splines():
   c=ConfigDict(backend.config.to_dict());c.crystal_file_path_path=str(backend.source/'input.xyz');g,_,k=create_grids(c);ps=create_pseudopotential(c)
   backend.grid_kmax=float(np.max(np.linalg.norm(np.asarray(g),axis=-1)));backend.beta_kmax=pp.kmax_for_grid(g,k);backend.tables=pp.build_tables(ps,backend.beta_kmax)
  with (a.output/'beta_tables.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,backend.tables),f,protocol=5)
  dump(a.output/'radial_program.json',dict(initial_source=str(backend.source),grid_kmax=backend.grid_kmax,beta_kmax=float(backend.beta_kmax),tables_sha256=sha(a.output/'beta_tables.pkl'),regenerated_with_same_frozen_code=True,not_retabulated_at_endpoint=True))
  backend.last_vectors=np.load(Path(old['state_directory'])/'orbitals.npy')
  stage('fresh_endpoint_reoptimization_and_spectrum')
  new=backend.evaluate(theta,a.output/'reoptimized',old)
  gates=physical_checks(new,reference,cfg['limits']);gates['target']=abs(new['gap_eV']-cfg['target_eV'])<=cfg['limits']['target_tolerance_eV'];gates['independent_gap_drift']=abs(new['gap_eV']-old['gap_eV'])<=p['independent_gap_drift_gate_eV']
  dump(a.output/'reoptimization_acceptance.json',dict(passed=all(gates.values()),checks=gates,old_gap_eV=old['gap_eV'],new_gap_eV=new['gap_eV'],new_readout=new))
  # Evaluate fresh Hamiltonian actions on the actual optimizer orbitals and on
  # the newly solved Ritz vectors. The saved original values are not results.
  stage('independent_orbital_Ritz_and_reoccupation_audit')
  source=Path(new['state_directory']);c=ConfigDict(json.loads((source/'metadata.json').read_text())['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
  with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
  with plain_fft_patch(),local_simpson_patch(),memory_bounded_xc(),fixed_radial_program(backend.grid_kmax,backend.beta_kmax,backend.tables),validated_radial_splines():
   out=calc(c,init_params=state,freq_mask=backend.mask);out.grid_kmax=backend.grid_kmax;factory=operator_factory(out);op,ng=factory(0)
   @jax.jit
   def actions(x):return jax.lax.map(lambda b:op(b),x.T).T
   q=np.asarray(out.params_pw)[0,0];hq=np.asarray(actions(jnp.asarray(q)));vec=np.load(source/'orbitals.npy');values=np.asarray(new['eigenvalues_Ha']);hv=np.asarray(actions(jnp.asarray(vec)))
   metrics=orbital_residuals(q,hq,np.full(q.shape[1],2.));metrics.update(occupation_residual(values,np.r_[np.full(q.shape[1],2.),np.zeros(vec.shape[1]-q.shape[1])],96.,2.,0.));metrics.update(ritz_residuals(vec,hv,values))
   # Rediagonalize the fresh local subspace before integer reoccupation.
   h=vec.conj().T@hv;ev,u=np.linalg.eigh((h+h.conj().T)/2);qn=vec@u;nocc=q.shape[1]
   if ev[nocc]-ev[nocc-1]<=0:raise ValueError('Nonpositive audited gap under fixed integer filling')
   kw=_kwargs_from(out);kw.pop('params_pw');occ=kw.pop('params_occ');kw['ewald_indices']=ewald_translation_indices(out.crystal.cell_vectors,c.ewald_args['ewald_cutoff']);V=float(abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))
   with compact_k_stream_patch(backend.mask) as compact:
    kw['make_coeff']=lambda x:x
    F=jax.jit(lambda x:free_energy_at_strain(jnp.zeros((3,3)),x,occ,**kw))
    newF=float(F(jnp.asarray(qn[:,:nocc])[None,None,...]));optimizerF=float(F(out.params_pw))
    rho=np.asarray(jax.jit(lambda x:compact.density(x,V,jnp.full((1,1,nocc),2.)))(jnp.asarray(qn[:,:nocc])[None,None,...]))[0]
   oldrho=np.load(source/'density.npy')[0]
   metrics.update(charge_error=abs(float(oldrho.mean()*V)-96.),spectral_update_density_L1_per_electron=float(abs(rho-oldrho).mean()*V/96),spectral_update_free_energy_change_abs_Ha=abs(newF-new['energy_Ha']))
   np.save(a.output/'reoccupied_density.npy',rho);numerical=check(metrics)
   result=dict(time=dt.datetime.now().astimezone().isoformat(),passed=all(gates.values()) and numerical['passed'],physical_checks=gates,numerical_acceptance=numerical,metrics=metrics,new_readout=new,new_spectral_free_energy_Ha=newF,optimizer_free_energy_Ha=optimizerF,spectral_vs_optimizer_free_energy_Ha=newF-optimizerF,old_readout=old,structural_equilibrium_claim=False,structural_relaxation_performed=False,scope='Independent endpoint process: new SCF and Ritz solve, fixed Miller/initial radial tables; nonzero forces retained, not relaxed equilibrium')
   dump(a.output/'acceptance.json',result);stage('complete',passed=result['passed'],gap_eV=new['gap_eV'])
   if not result['passed']:raise ValueError('Endpoint numerical/physical gate failed; original result retained')
 except Exception as exc:stage('failed',error=str(exc));raise
 finally:probe.finish()
if __name__=='__main__':main()
