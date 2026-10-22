"""Real Gamma DFT backend for the bounded joint controller; fresh endpoints.

Frozen ordered Miller basis and radial tables, reoptimized electrons, force /
stress report and independently solved low spectrum at every trial. CPU Si
dry run deliberately uses a small cutoff; it is not paper-quality data.
"""
import argparse,gc,json,pickle,sys,datetime
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from joint_design import run_design,deform,nearest_distance,matrix,dump,HA_EV
P=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(P/'validation/workflows/response_design/reverse_response'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from gamma_response import GammaResponse,validated_radial_splines,sha
from bounded_xc import memory_bounded_xc
from fixed_radial_program import fixed_radial_program
from residual_report import fused_report
from variational_ks import operator_factory
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from jrystal._src.const import ANGSTROM2BOHR

class GammaDFTBackend:
 kind='dft'
 def stage(self,name,folder,**details):
  record=dict(time=datetime.datetime.now().isoformat(),stage=name,folder=str(folder),**details)
  if self.p.get('backend_status_file'):dump(Path(self.p['backend_status_file']),record)
  print('E4_STAGE',json.dumps(record),flush=True)
 def __init__(self,protocol):
  self.p=protocol;self.source=Path(protocol['source']);self.meta=json.loads((self.source/'metadata.json').read_text())
  for s,item in self.meta['physical']['pp'].items():
   if sha(item['path'])!=item['sha256']:raise ValueError('Changed PP '+s)
  if not protocol.get('CPU_dry_run',False):
   for f in protocol['required_acceptance_files']:
    if not json.loads(Path(f).read_text())['passed']:raise ValueError('Required E3 gate not passed')
  self.theta0=np.array(protocol['theta0']);physical=self.meta['physical'];self.symbols=physical['symbols'];factor=float(ANGSTROM2BOHR)
  f=np.eye(3)+matrix(self.theta0[:6]);inverse=np.linalg.inv(f).T
  self.basecell=np.array(physical['cell_angstrom'])*factor@inverse
  self.basepos=(np.array(physical['positions_angstrom'])*factor-self.theta0[6:].reshape(-1,3))@inverse
  self.config=ConfigDict(self.meta['config']);self.config.grid_sizes=protocol.get('grid',self.config.grid_sizes);self.config.cutoff_energy=protocol.get('cutoff_Ha',self.config.cutoff_energy)
  self.config.d3ft.parallel.k_pool=False;self.config.k_grid_sizes=1
  self.config.d3ft.optim.lr_schedule='cosine';self.config.d3ft.optim.lr_final=1e-6
  self.mask=None;self.tables=None;self.last_vectors=None
  if not protocol.get('CPU_dry_run',False):self.mask=np.load(self.source/'mask.npy')
  # New physical endpoints are recomputed here. The old source defines the
  # geometry/model, never an unrecorded replacement endpoint.
 def geometry(self,theta):return deform(self.basecell,self.basepos,theta)
 def evaluate(self,theta,folder,parent):
  folder=Path(folder);folder.mkdir(exist_ok=False);cell,pos=self.geometry(theta);c=ConfigDict(self.config.to_dict())
  self.stage('electronic_reoptimization_begin',folder)
  factor=.529177210903*float(ANGSTROM2BOHR)
  _write_xyz(folder/'input.xyz',cell/factor,pos/factor,self.symbols)
  c.crystal_file_path_path=str(folder/'input.xyz');c.epoch=self.p['epochs'];c.optimizer_args.learning_rate=.01
  state=None
  if parent:
   with (Path(parent['state_directory'])/'state.pkl').open('rb') as f:state=pickle.load(f)
  elif not self.p.get('CPU_dry_run',False):
   with (self.source/'state.pkl').open('rb') as f:state=pickle.load(f)
  program=nullcontext() if self.tables is None else fixed_radial_program(self.grid_kmax,self.beta_kmax,self.tables)
  with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),program,validated_radial_splines():
   out=calc(c,init_params=state,freq_mask=self.mask);jax.block_until_ready(out.params_pw)
   self.stage('electronic_reoptimization_complete',folder,energy_Ha=float(out.total_energy))
   if self.tables is None:
    self.mask=np.array(out.freq_mask);self.tables=out.beta_tables;self.grid_kmax=float(out.grid_kmax);self.beta_kmax=float(out.beta_kmax)
   else:out.grid_kmax=self.grid_kmax
   if max(np.max(abs(np.array(out.crystal.cell_vectors)-cell)),np.max(abs(np.array(out.crystal.positions)-pos)))>2e-11:raise ValueError('Cartesian Bohr boundary conversion')
   report=fused_report(out);force=force_report(out);factory=operator_factory(out);op,ng=factory(0)
   self.stage('force_stress_and_residual_complete',folder,residual_Ha=float(report['variational_residual']))
   nocc=int(round(sum(out.pseudopot.valence_charges)/2));nb=nocc+4
   self.stage('independent_low_spectrum_begin',folder,plane_waves=ng,bands=nb)
   @jax.jit
   def stream(x):return jax.lax.map(lambda b:op(b),x.T).T
   mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(stream(jnp.asarray(x.reshape(-1,1))))[:,0],matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128)
   if ng<1500:
    h=mat.matmat(np.eye(ng,dtype=complex));vals,vecs=np.linalg.eigh((h+h.conj().T)/2);vals=vals[:nb];vecs=vecs[:,:nb]
   else:
    occ=np.array(out.params_pw)[0,0];rng=np.random.default_rng(20261015);empty=(self.last_vectors[:,nocc:nb] if self.last_vectors is not None else rng.normal(size=(ng,4))+1j*rng.normal(size=(ng,4)))
    initial=np.linalg.qr(np.concatenate([occ,empty],axis=1))[0]
    diagonal=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((ng,ng),matvec=lambda x:diagonal*x.reshape(-1),matmat=lambda x:diagonal[:,None]*x,dtype=np.complex128)
    vals,vecs=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=400)
   vecs=np.linalg.qr(vecs)[0];hv=mat.matmat(vecs);h=vecs.conj().T@hv;vals,rotate=np.linalg.eigh((h+h.conj().T)/2);vecs=vecs@rotate;hv=hv@rotate
   residuals=np.linalg.norm(hv-vecs*vals[None,:],axis=0);self.last_vectors=vecs
   np.save(folder/'orbitals.npy',vecs);np.save(folder/'mask.npy',self.mask);np.save(folder/'density.npy',np.asarray(out.density_grid));np.save(folder/'energy_history.npy',np.asarray(out.energy_history))
   with (folder/'state.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5)
   meta=dict(time=datetime.datetime.now().isoformat(),source=str(self.source),source_metadata_sha256=sha(self.source/'metadata.json'),pp=self.meta['physical']['pp'],config=c.to_dict(),theta=np.asarray(theta).tolist(),
             fresh_electronic_optimization=True,parent_state=(str(self.source) if not self.p.get('CPU_dry_run',False) else None) if parent is None else parent['state_directory'],coordinate_unit='bohr',fixed_Miller_sha256=sha(folder/'mask.npy'),CPU_dry_run=self.p.get('CPU_dry_run',False),code_sha256={str(p.relative_to(P)):sha(p) for p in (P/'d3ft').rglob('*.py')})
   dump(folder/'metadata.json',meta)
   result=dict(energy_Ha=float(out.total_energy),gap_eV=float((vals[nocc]-vals[nocc-1])*HA_EV),residual_Ha=float(report['variational_residual']),
    forces_Ha_Bohr=np.asarray(force['forces_Ha_Bohr']).tolist(),ASR_Ha_Bohr=float(force['acoustic_sum_rule']),stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),
    charge=float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell))),electrons=float(sum(out.pseudopot.valence_charges)),ritz_max_Ha=float(max(residuals)),
    nearest_distance_bohr=nearest_distance(cell,pos),cell_determinant_bohr3=float(np.linalg.det(cell)),branch_margin_Ha=float(min(vals[nocc-1]-vals[nocc-2],vals[nocc+1]-vals[nocc])),
    electronic_reoptimized=True,state_directory=str(folder),eigenvalues_Ha=vals.tolist(),structural_relaxation_performed=False)
   dump(folder/'readout.json',result)
   self.stage('readout_complete',folder,gap_eV=result['gap_eV'],ritz_max_Ha=result['ritz_max_Ha'])
  jax.clear_caches();gc.collect();return result
 def gradient(self,theta,current,folder):
  folder=Path(folder);folder.mkdir(exist_ok=False);src=Path(current['state_directory']);c=ConfigDict(json.loads((src/'metadata.json').read_text())['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant'
  self.stage('adjoint_begin',folder)
  with (src/'state.pkl').open('rb') as f:state=pickle.load(f)
  with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),fixed_radial_program(self.grid_kmax,self.beta_kmax,self.tables),validated_radial_splines():
   out=calc(c,init_params=state,freq_mask=self.mask);out.grid_kmax=self.grid_kmax
   r=GammaResponse(out,np.load(src/'orbitals.npy'),np.array(current['eigenvalues_Ha']),theta,include_coordinates=True)
   primal=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[None,:],axis=0)))
   if primal>1e-8:raise ValueError('Response primal mismatch')
   def progress(record):
    with (folder/'PCG_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    self.stage('adjoint_iteration',folder,**record)
   b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)));lam,record=r.solve(b,progress);g=np.asarray(r.reverse_gradient(lam))*HA_EV
   dump(folder/'response.json',dict(gradient_eV=g.tolist(),adjoint_record=record,primal_residual_Ha=primal))
   self.stage('adjoint_complete',folder,iterations=record['iterations'],primal_residual_Ha=primal)
  jax.clear_caches();gc.collect();return g

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.protocol.read_text())
 if p.get('CPU_dry_run',False) and jax.default_backend()!='cpu':raise ValueError('CPU-only dry run')
 backend=GammaDFTBackend(p);target=p['target_eV'] if p['target_eV'] is not None else lambda initial:initial['gap_eV']+p['dry_run_target_offset_eV']
 result=run_design(a.output,p['theta0'],target,p['limits'],backend,max_updates=p.get('max_updates',3))
 print('E4 '+('CPU Si真实DFT干跑' if p.get('CPU_dry_run') else '联合设计')+'结束。\n接受更新数 '+str(result['completed_updates'])+'；每态有真实重优化与谱读出。\n'+('小网格干跑不作为MgSiN2论文结果。' if p.get('CPU_dry_run') else '原始轨迹已保存，端点独立审计另验。'),flush=True)
if __name__=='__main__':main()
