"""Fresh 64-k SCF checkpoint on one compact-stream device or legacy eight devices, with first-update charge gate."""
import argparse,datetime,gc,json,os,pickle,sys,time
from contextlib import nullcontext
from compact_k_stream import compact_k_stream_patch
from memory_probe import MemoryProbe
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import dump,sha,validated_radial_splines
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
from streamed_kpool import streamed_kpool_patch
from mesh_basis_restart import project_miller_coefficients
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from residual_report import fused_report
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from jrystal._src.const import ANGSTROM2BOHR

ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True)
ap.add_argument('--nu',type=float,required=True);ap.add_argument('--epochs',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--single-stream',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--checkpoint',type=Path);ap.add_argument('--read-only',action='store_true');a=ap.parse_args()
if len(jax.devices())!=(1 if a.single_stream else 8) or a.nu not in (0.,-.001,.001) or not 1<=a.epochs<=4000:raise ValueError('Registered one/eight-device k4 FD protocol')
a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output).start()
def stage(name,**kw):
 row=dict(time=datetime.datetime.now().isoformat(),stage=name,jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),nu=a.nu,**kw)
 dump(a.output/'status.json',row);print('E6_K4',json.dumps(row),flush=True)
try:
 m=json.loads((a.reference/'metadata.json').read_text());mask=np.load(a.reference/'mask.npy')
 for symbol,v in m['physical']['pp'].items():
  if sha(v['path'])!=v['sha256']:raise ValueError('Changed PP '+symbol)
 factor=float(ANGSTROM2BOHR);scale=np.exp(a.nu/3)
 cell=np.asarray(m['physical']['cell_angstrom'])*factor*scale;positions=np.asarray(m['physical']['positions_angstrom'])*factor*scale
 _write_xyz(a.output/'input.xyz',cell/(.529177210903*factor),positions/(.529177210903*factor),m['physical']['symbols'])
 seed=a.checkpoint or a.parent
 sm=json.loads((seed/'metadata.json').read_text())
 if sha(seed/'state.pkl')!=sm['state_sha256']:raise ValueError('Warm state changed')
 if a.checkpoint:
  if sm.get('nu_log_volume')!=a.nu:raise ValueError('Checkpoint geometry differs')
 with (seed/'state.pkl').open('rb') as f:state=pickle.load(f)
 pw=np.asarray(state['pw'])
 if pw.shape[1]!=64:raise ValueError('k4 warm orbitals required')
 oldmask=np.load(seed/'mask.npy')
 if not np.array_equal(oldmask,mask):
  pw,projection=project_miller_coefficients(pw,oldmask,mask)
  dump(a.output/'initialization_projection.json',dict(projection,source=str(seed),initialization_only=True))
  pw=np.asarray(jnp.linalg.qr(jnp.asarray(pw))[0]);jax.clear_caches();gc.collect()
 c=ConfigDict(m['config']);c.k_grid_sizes=4;c.d3ft.parallel.k_pool=not a.single_stream;c.crystal_file_path_path=str(a.output/'input.xyz')
 c.epoch=a.epochs;c.optimizer_args.learning_rate=0. if a.read_only else .001;c.d3ft.optim.lr_schedule='constant' if a.read_only else 'cosine';c.d3ft.optim.lr_final=1e-6
 c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
 with (streamed_kpool_patch() if not a.single_stream else nullcontext()),plain_fft_patch(k_pool=not a.single_stream),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,a.reference) as info,validated_radial_splines(),(compact_k_stream_patch(mask) if a.single_stream else nullcontext()):
  stage('fresh_k4_checkpoint',epochs=a.epochs,uniform_occupation='2/64; fresh API initialization, old occ not reused')
  start=time.monotonic();out=calc(c,init_params={'pw':pw},freq_mask=mask);jax.block_until_ready(out.params_pw);out.grid_kmax=info['grid_kmax']
  solve_seconds=time.monotonic()-start
  if len(out.k_frac)!=64 or max(np.max(abs(np.asarray(out.crystal.cell_vectors)-cell)),np.max(abs(np.asarray(out.crystal.positions)-positions)))>2e-11:raise ValueError('Wrong mesh or geometry')
  stage('native_force_stress_residual')
  report=fused_report(out);forces=force_report(out)
  charge=float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell)))
  res=float(report['variational_residual']);asr=float(forces['acoustic_sum_rule'])
  passed=res<=1e-8 and asr<=5e-8 and abs(charge-96)<=96e-8
  np.save(a.output/'mask.npy',mask);np.save(a.output/'density.npy',np.asarray(out.density_grid));np.save(a.output/'energy_history.npy',np.asarray(out.energy_history))
  with (a.output/'state.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5)
  (a.output/'beta_tables.pkl').write_bytes((a.reference/info['tables_file']).read_bytes())
  point=dict(energy_Ha=float(out.total_energy),residual=res,acoustic_sum_rule_Ha_Bohr=asr,charge=charge,electrons=96.,
   forces_Ha_Bohr=np.asarray(forces['forces_Ha_Bohr']).tolist(),stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),
   cell_bohr=cell.tolist(),positions_bohr=positions.tolist(),k_fractional=np.asarray(out.k_frac).tolist(),
   stationarity_pass=res<=1e-8,physical_passed=passed,nu_log_volume=a.nu,solve_seconds=solve_seconds,epochs=a.epochs,
   ionic_condition='fixed fractional positions',structural_relaxation_performed=False)
  dump(a.output/'point.json',point)
  physical=dict(m['physical'],cell_angstrom=(cell/factor).tolist(),positions_angstrom=(positions/factor).tolist(),k=[4])
  saved=c.to_dict();saved['d3ft']['charge_gate'].pop('report_path',None)
  dump(a.output/'metadata.json',dict(physical=physical,config=saved,state_sha256=sha(a.output/'state.pkl'),source=str(a.reference),
   initialization_only=str(seed),nu_log_volume=a.nu,new_response_state=True,checkpoint_accepted=passed,
   frozen_radial_program=dict(info,tables_file='beta_tables.pkl',tables_sha256=sha(a.output/'beta_tables.pkl')),
   code_sha256={str(f.relative_to(P)):sha(f) for f in (P/'d3ft').rglob('*.py')},jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID')))
  if abs(charge-96)>96e-8:raise ValueError('Charge not conserved')
  stage('accepted' if passed else 'unconverged_checkpoint_not_evidence',passed=passed,residual_Ha=res,ASR_Ha_Bohr=asr)
except Exception as exc:stage('failed',error=str(exc));raise
finally:probe.finish()
