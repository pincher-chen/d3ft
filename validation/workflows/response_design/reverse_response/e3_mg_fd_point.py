"""Reoptimize one preregistered mixed Mg direction with frozen discrete tables.

Explicit adoption of an owned, E2-audited checkpoint: records old/new hashes,
and requires new-code 54-component primal gates. Does not bypass generic_point's
strict restart guard or relabel the old SCF as a new calculation.
"""
import argparse,datetime,gc,json,os,pickle,sys,time,warnings
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
from gamma_response import sha,dump,validated_radial_splines,HA_EV
from geometry_layout import GeometryLayout
from bounded_xc import memory_bounded_xc
from memory_probe import MemoryProbe
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'))
sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from variational_ks import operator_factory
from residual_report import fused_report
from d3ft import pseudopotential as pp
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from jrystal._src.const import ANGSTROM2BOHR
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--direction',type=int,required=True);ap.add_argument('--h',type=float,required=True)
ap.add_argument('--sign',type=int,choices=[-1,1],required=True);ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--epochs',type=int,default=2000);a=ap.parse_args();E=a.root
a.output.mkdir(exist_ok=False);start=time.monotonic();probe=MemoryProbe(a.output).start()
def stage(name,**kw):
 d=dict(stage=name,time=datetime.datetime.now().isoformat(),pid=os.getpid(),jobid=os.environ.get('SLURM_JOB_ID'),**kw)
 dump(a.output/'status.json',d);print('E3_MG_FD',json.dumps(d),flush=True)
try:
 mode=json.loads((E/'mg_mode_acceptance.json').read_text())
 if not mode['passed']:raise ValueError('Mg physical 54-component mode/translation gate required')
 rp=json.loads((E/'mg_response_protocol.json').read_text());directions=json.loads((E/'mg_fd_directions.json').read_text())
 if a.h not in directions['steps'] or not 0<=a.direction<len(directions['directions']):raise ValueError('Unregistered FD')
 source=Path(rp['source']);orb=Path(rp['orbitals']);sm=json.loads((source/'metadata.json').read_text())
 om=json.loads((orb/'metadata.json').read_text());bs=json.loads((orb/'summary.json').read_text())
 for name,h in om['source_hashes'].items():
  if sha(source/name)!=h:raise ValueError('Changed owned source '+name)
 if sha(source/'state.pkl')!=sm['state_sha256']:raise ValueError('Owned checkpoint hash mismatch')
 if sha(orb/'orbitals.npy')!=bs['saved_orbitals_sha256']:raise ValueError('Changed source orbitals')
 for symbol,item in sm['physical']['pp'].items():
  if sha(item['path'])!=item['sha256']:raise ValueError('UPF changed '+symbol)
 with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
 cfg=ConfigDict(sm['config']);cfg.epoch=1;cfg.crystal_file_path_path=str(source/'input.xyz')
 cfg.d3ft.parallel.k_pool=False;cfg.d3ft.optim.lr_schedule='constant';cfg.optimizer_args.learning_rate=0.
 mask=np.load(source/'mask.npy');d=np.asarray(directions['directions'][a.direction]);delta=a.sign*a.h*d
 with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc():
  stage('reconstruct_owned_parent_tables')
  parent=calc(cfg,init_params=state,freq_mask=mask)
  if parent.core_density_grid is not None or len(parent.k_frac)!=1:raise ValueError('Non-NLCC Gamma only')
  tables=parent.beta_tables;bk=float(parent.beta_kmax);gk=float(parent.grid_kmax)
  layout=GeometryLayout(16,rp['theta0'],include_coordinates=True)
  wanted_cell,wanted_pos=map(np.asarray,layout.deform(parent.crystal.cell_vectors,parent.crystal.positions,jnp.asarray(delta)))
  parent_cell=np.asarray(parent.crystal.cell_vectors);parent_positions=np.asarray(parent.crystal.positions)
  # _write_xyz uses modern Bohr; Crystal reads historical JR Bohr. Compensate
  # only at this I/O boundary; the response and all requested displacements use JR Bohr.
  factor=.529177210903*float(ANGSTROM2BOHR)
  _write_xyz(a.output/'input.xyz',wanted_cell/factor,wanted_pos/factor,sm['physical']['symbols'])
  del parent;jax.clear_caches();gc.collect()
  cfg.crystal_file_path_path=str(a.output/'input.xyz');cfg.epoch=a.epochs
  cfg.d3ft.optim.lr_schedule='cosine';cfg.d3ft.optim.lr_final=1e-6;cfg.optimizer_args.learning_rate=.01
  original_local,original_kmax,original_tables=pp.potential_local_reciprocal,pp.kmax_for_grid,pp.build_tables
  @contextmanager
  def fixed_program():
   def local(*args,**kw):kw['kmax']=gk;return original_local(*args,**kw)
   pp.potential_local_reciprocal=local;pp.kmax_for_grid=lambda *args,**kw:bk
   pp.build_tables=lambda pseudopot,kmax:tables
   try:yield
   finally:pp.potential_local_reciprocal=original_local;pp.kmax_for_grid=original_kmax;pp.build_tables=original_tables
  meta=dict(source=str(source),parent_hashes={n:sha(source/n) for n in ['metadata.json','state.pkl','mask.npy','density.npy','point.json','input.xyz']},
   inherited_original_code_sha256=sm['code_sha256'],new_code_sha256={str(p.relative_to(P)):sha(p) for p in (P/'d3ft').rglob('*.py')},
   parent_adoption='Owned E2-audited checkpoint; new physical GammaResponse primal and full-mode gate required; old SCF not relabeled',
   direction=a.direction,direction_vector=d.tolist(),h=a.h,sign=a.sign,delta=delta.tolist(),
   fixed_parent_miller_basis=True,fixed_parent_beta_tables=True,fixed_parent_local_radial_nodes=True,
   beta_kmax=bk,local_grid_kmax=gk,coordinate_unit='bohr',angstrom_to_bohr=float(ANGSTROM2BOHR),
   parent_cell_bohr=parent_cell.tolist(),parent_positions_bohr=parent_positions.tolist(),
   requested_cell_bohr=wanted_cell.tolist(),requested_positions_bohr=wanted_pos.tolist(),config=cfg.to_dict(),
   jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),new_response_result=True,
   scope='Independent electronic reoptimization then low-spectrum readout; no ionic relaxation or earlier solver study result')
  dump(a.output/'metadata.json',meta)
  with fixed_program():
   stage('electronic_reoptimization',epochs=a.epochs)
   out=calc(cfg,init_params=state,freq_mask=mask);jax.block_until_ready(out.params_pw);out.grid_kmax=gk
   if np.max(abs(np.asarray(out.crystal.positions)-wanted_pos))>2e-11 or np.max(abs(np.asarray(out.crystal.cell_vectors)-wanted_cell))>2e-11:raise ValueError('Geometry unit/serialization mismatch')
   np.save(a.output/'mask.npy',np.asarray(out.freq_mask));np.save(a.output/'density.npy',np.asarray(out.density_grid))
   np.save(a.output/'energy_history.npy',np.asarray(out.energy_history))
   with (a.output/'state.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5)
   meta['state_sha256']=sha(a.output/'state.pkl');dump(a.output/'metadata.json',meta)
   jax.clear_caches();gc.collect();stage('stationarity_forces_and_stress')
   with validated_radial_splines():
    report=fused_report(out);forces=force_report(out)
   residual=float(report['variational_residual']);asr=float(forces['acoustic_sum_rule'])
   charge=float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))
   electrons=float(sum(out.pseudopot.valence_charges))
   point=dict(energy_Ha=float(out.total_energy),residual=residual,acoustic_sum_rule_Ha_Bohr=asr,
    forces_Ha_Bohr=np.asarray(forces['forces_Ha_Bohr']).tolist(),stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),
    pressure_GPa=float(report['pressure_GPa']),positions_bohr=np.asarray(out.crystal.positions).tolist(),
    cell_bohr=np.asarray(out.crystal.cell_vectors).tolist(),stationarity_pass=residual<=1e-8,epochs=len(out.energy_history),
    electrons=electrons,integrated_valence_density=charge,coordinate_unit='bohr',structural_relaxation_performed=False)
   dump(a.output/'point.json',point)
   if residual>1e-8 or asr>5e-8 or abs(charge-electrons)>1e-8*electrons:raise ValueError(f'FD electronic state failed {residual}, ASR {asr}, charge {charge}')
   occ=np.asarray(out.get_occupation(out.params_occ,out.final_eps))
   if np.max(abs(occ-2.))>1e-12:raise ValueError('Not fixed integer occupation')
   jax.clear_caches();gc.collect();stage('independent_low_spectrum')
   with validated_radial_splines():factory=operator_factory(out)
   hop,ng=factory(0);action=jax.jit(hop);nocc=int(round(point['electrons']/2));nb=nocc+4;block=4
   @jax.jit
   def stream(x):
    pad=(-x.shape[1])%block;padded=jnp.pad(x,((0,0),(0,pad)))
    blocks=jnp.swapaxes(padded.reshape(ng,-1,block),0,1)
    h=jax.lax.map(jax.vmap(hop,in_axes=1,out_axes=1),blocks)
    return jnp.swapaxes(h,0,1).reshape(ng,-1)[:,:x.shape[1]]
   mat=LinearOperator((ng,ng),matvec=lambda x:np.asarray(action(jnp.asarray(x.reshape(-1)))),
    matmat=lambda x:np.asarray(stream(jnp.asarray(x))),dtype=np.complex128)
   diag=1/(1+np.asarray(hop.kinetic_diagonal))
   pre=LinearOperator((ng,ng),matvec=lambda x:diag*x.reshape(-1),matmat=lambda x:diag[:,None]*x,dtype=np.complex128)
   if not out.config.d3ft.optim.enabled:raise ValueError('Owned Stiefel checkpoint required')
   occupied=np.asarray(out.params_pw)[0,0]
   if occupied.shape!=(ng,nocc):raise ValueError('Unexpected occupied Stiefel block shape')
   parent_vectors=np.load(orb/'orbitals.npy');rng=np.random.default_rng(20261015)
   initial=np.concatenate([occupied,parent_vectors[:,nocc:],rng.normal(size=(ng,2))+1j*rng.normal(size=(ng,2))],axis=1)
   initial=np.linalg.qr(initial[:,:nb])[0]
   with warnings.catch_warnings(record=True) as recorded:
    vals,vectors=lobpcg(mat,initial,M=pre,largest=False,tol=1e-10,maxiter=400)
   # Independent final QR/Ritz and true action, never accept a LOBPCG warning as convergence.
   vectors=np.linalg.qr(vectors)[0];hv=mat.matmat(vectors)
   h=vectors.conj().T@hv;vals,rotate=np.linalg.eigh((h+h.conj().T)/2);vectors=vectors@rotate;hv=hv@rotate
   residuals=np.linalg.norm(hv-vectors*vals[None,:],axis=0);orth=float(np.max(abs(vectors.conj().T@vectors-np.eye(nb))))
   sepv=float(vals[nocc-1]-vals[nocc-2]);sepc=float(vals[nocc+1]-vals[nocc])
   summary=dict(eigenvalues_Ha=vals.tolist(),gap_eV=float((vals[nocc]-vals[nocc-1])*HA_EV),
    residuals_Ha=residuals.tolist(),maximum_ritz_residual_Ha=float(max(residuals)),orthogonality_max=orth,
    vbm_adjacent_separation_Ha=sepv,cbm_adjacent_separation_Ha=sepc,n_occupied=nocc,
    warnings=[str(w.message) for w in recorded],seconds=time.monotonic()-start)
   dump(a.output/'spectrum_summary.json',summary)
   np.save(a.output/'orbitals.npy',vectors)
   if not np.isfinite(vals).all() or max(residuals)>1e-9 or orth>1e-10 or min(sepv,sepc)<1e-5:raise ValueError('FD low-spectrum residual/branch gate failed')
   stage('completed',gap_eV=summary['gap_eV'],residual_Ha=residual,seconds=time.monotonic()-start)
except Exception as exc:stage('failed',error=str(exc));raise
finally:probe.finish()
