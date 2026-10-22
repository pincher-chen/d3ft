"""E9 energy/force evaluator and complete spectral readout; all outputs newly computed."""
import contextlib,gc,os,pickle,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from common import dump,read,sha
from joint_design_dft import GammaDFTBackend
from joint_design import nearest_distance,HA_EV
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from radial_quadrature import local_simpson_patch
from bounded_xc import memory_bounded_xc
from fixed_radial_program import fixed_radial_program
from gamma_response import validated_radial_splines
from residual_report import fused_report
from variational_ks import operator_factory
from jrystal._src.const import ANGSTROM2BOHR

class E9Backend(GammaDFTBackend):
 def __init__(self,p):
  super().__init__(p);rad=read(p['radial_program_json']);self.grid_kmax=rad['grid_kmax'];self.beta_kmax=rad['beta_kmax']
  with Path(p['radial_tables']).open('rb') as f:self.tables=pickle.load(f)
 def gradient(self,theta,current,folder):
  # Reconstruct the same accepted electronic state without writing checkpoints
  # inside its previously committed evaluation directory.
  import joint_design_dft as module
  original=module.calc
  def read_only(config,**kwargs):
   with config.d3ft.unlocked():
    for key in ('checkpoint','checkpoint_resume'):
     if key in config.d3ft:del config.d3ft[key]
   return original(config,**kwargs)
  module.calc=read_only
  try:return super().gradient(theta,current,folder)
  finally:module.calc=original
 def evaluate(self,theta,folder,parent,spectrum=True,resume=None):
  folder=Path(folder);folder.mkdir(exist_ok=False);cell,pos=self.geometry(theta);c=ConfigDict(self.config.to_dict());_write_xyz(folder/'input.xyz',cell/(.529177210903*float(ANGSTROM2BOHR)),pos/(.529177210903*float(ANGSTROM2BOHR)),self.symbols)
  c.crystal_file_path_path=str(folder/'input.xyz');c.epoch=self.p['epochs'];c.optimizer_args.learning_rate=.01;c.d3ft.optim.lr_schedule='cosine';c.d3ft.optim.lr_final=1e-6;c.d3ft.checkpoint=dict(root=str(folder/'checkpoints'),interval=500);c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(folder/'charge.jsonl'))
  source=Path(parent['state_directory']) if parent else self.source
  with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
  offset=0;patch=contextlib.nullcontext()
  if resume:
   cp=Path(resume);r=read(cp/'latest.json');model=read(cp/'model.json');assert r['complete'] and sha(cp/r['state_file'])==r['state_sha256'];assert np.max(abs(np.asarray(model['positions_bohr'])-pos))<2e-11 and np.max(abs(np.asarray(model['cell_bohr'])-cell))<2e-11
   with (cp/r['state_file']).open('rb') as f:state=pickle.load(f)
   offset=int(model['config'].get('d3ft',{}).get('checkpoint_resume',{}).get('completed_steps',0))+int(r['iteration']);offset=min(offset,self.p['epochs']);c.epoch=max(1,self.p['epochs']-offset);c.d3ft.checkpoint_resume=dict(completed_steps=offset,original_total_steps=self.p['epochs'])
   import importlib,copy
   ground=importlib.import_module('d3ft.calc.ground_state');orig=ground._learning_rate_schedule
   @contextlib.contextmanager
   def tail():
    def schedule(cfg,base=None):
     full=copy.deepcopy(cfg);full.epoch=self.p['epochs'];f=orig(full,base=base);return (lambda count:f(count+offset)) if callable(f) else f
    ground._learning_rate_schedule=schedule
    try:yield
    finally:ground._learning_rate_schedule=orig
   patch=tail()
   if offset>=self.p['epochs']:c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';patch=contextlib.nullcontext()
   dump(folder/'resume.json',dict(source=str(cp),offset=offset,optimizer_moments_restored=False,same_cosine_tail=offset<self.p['epochs']))
  with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),fixed_radial_program(self.grid_kmax,self.beta_kmax,self.tables),validated_radial_splines(),patch:
   out=calc(c,init_params=state,freq_mask=self.mask);jax.block_until_ready(out.params_pw);out.grid_kmax=self.grid_kmax
   if max(np.max(abs(np.asarray(out.crystal.cell_vectors)-cell)),np.max(abs(np.asarray(out.crystal.positions)-pos)))>2e-11:raise ValueError('E9 Cartesian Bohr conversion mismatch')
   # Commit physical state before potentially expensive final reports/spectrum.
   with (folder/'state.pkl.tmp').open('wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5);f.flush();os.fsync(f.fileno())
   (folder/'state.pkl.tmp').replace(folder/'state.pkl');np.save(folder/'mask.npy',self.mask);np.save(folder/'density.npy',np.asarray(out.density_grid));report=fused_report(out);force=force_report(out)
   meta=dict(config=c.to_dict(),pp=self.meta['physical']['pp'],theta=np.asarray(theta).tolist(),source=str(self.source),source_metadata_sha256=sha(self.source/'metadata.json'),fixed_Miller_sha256=sha(folder/'mask.npy'),state_sha256=sha(folder/'state.pkl'),coordinate_unit='bohr',fresh_electronic_optimization=True,parent_state=str(source),CPU_dry_run=False,epochs_completed=min(self.p['epochs'],offset+c.epoch))
   dump(folder/'metadata.json',meta)
   result=dict(energy_Ha=float(out.total_energy),residual_Ha=float(report['variational_residual']),forces_Ha_Bohr=np.asarray(force['forces_Ha_Bohr']).tolist(),ASR_Ha_Bohr=float(force['acoustic_sum_rule']),stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),charge=float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell))),electrons=96.,nearest_distance_bohr=nearest_distance(cell,pos),cell_determinant_bohr3=float(np.linalg.det(cell)),electronic_reoptimized=True,state_directory=str(folder),structural_relaxation_performed=True)
   dump(folder/'force_readout.json',result)
   if result['residual_Ha']>1e-8 or result['ASR_Ha_Bohr']>5e-8 or abs(result['charge']-96)>1e-6:raise ValueError('E9 electronic/ASR/charge condition failed; not an ionic step')
   if spectrum:
    op,ng=operator_factory(out)(0);nb=56
    @jax.jit
    def stream(x):
     pad=(-x.shape[1])%4;y=jnp.pad(x,((0,0),(0,pad))).reshape(ng,-1,4).transpose(1,0,2);z=jax.lax.map(jax.vmap(op,in_axes=1,out_axes=1),y);return z.transpose(1,0,2).reshape(ng,-1)[:,:x.shape[1]]
    mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(stream(jnp.asarray(x.reshape(-1,1))))[:,0],matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128);rng=np.random.default_rng(20261019);initial=np.linalg.qr(np.concatenate([np.asarray(out.params_pw)[0,0],rng.normal(size=(ng,8))+1j*rng.normal(size=(ng,8))],axis=1))[0];d=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((ng,ng),matvec=lambda x:d*x.reshape(-1),matmat=lambda x:d[:,None]*x,dtype=np.complex128)
    import warnings
    for k in range(12):
     with warnings.catch_warnings(record=True):vals,q=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=100)
     q=np.linalg.qr(q)[0];hq=mat.matmat(q);h=q.conj().T@hq;vals,u=np.linalg.eigh((h+h.conj().T)/2);q=q@u;hq=hq@u;res=np.linalg.norm(hq-q*vals,axis=0);np.save(folder/'spectrum_checkpoint.npy',q);dump(folder/'spectrum_checkpoint.json',dict(iteration=(k+1)*100,max_residual_Ha=float(max(res))))
     if max(res)<=1e-9:break
     initial=q
    np.save(folder/'orbitals.npy',q);self.last_vectors=q
    result.update(gap_eV=float((vals[48]-vals[47])*HA_EV),ritz_max_Ha=float(max(res)),branch_margin_Ha=float(min(vals[47]-vals[46],vals[49]-vals[48])),eigenvalues_Ha=vals.tolist())
    if result['ritz_max_Ha']>1e-9 or result['branch_margin_Ha']<=1e-5:raise ValueError('E9 spectral/branch check failed')
   dump(folder/'readout.json',result)
  jax.clear_caches();gc.collect();return result
