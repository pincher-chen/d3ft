"""Fresh clamped-fractional k2 log-volume point on two explicit GPUs."""
import argparse,datetime,gc,json,os,pickle,sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from ml_collections import ConfigDict
from gamma_response import dump,sha,validated_radial_splines
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
from memory_probe import MemoryProbe
from streamed_kpool import streamed_kpool_patch
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from residual_report import fused_report
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from jrystal._src.const import ANGSTROM2BOHR
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--nu',type=float);ap.add_argument('--theta-json',type=Path);ap.add_argument('--parent-theta-json',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=4000);ap.add_argument('--devices',type=int,default=2,choices=[1,2]);ap.add_argument('--stream-k',action='store_true');a=ap.parse_args()
if len(jax.devices())!=a.devices:raise ValueError('E6 protocol device count mismatch')
if a.devices==1 and not a.stream_k:raise ValueError('Single-device E6 requires bounded k/band streaming')
if a.theta_json is None and (a.nu is None or abs(a.nu) not in (.002,.001)):raise ValueError('Unregistered log-volume step')
if a.theta_json is not None and (a.nu is not None or a.parent_theta_json is None):raise ValueError('Joint geometry requires parent and trial theta, no nu')
a.output.mkdir(exist_ok=False);probe=MemoryProbe(a.output).start()
def stage(s,**kw):
 d=dict(time=datetime.datetime.now().isoformat(),stage=s,jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),**kw);dump(a.output/'status.json',d);print('E6_K2',json.dumps(d),flush=True)
try:
 m=json.loads((a.source/'metadata.json').read_text());point=json.loads((a.source/'point.json').read_text())
 if sha(a.source/'state.pkl')!=m['state_sha256'] or not point['stationarity_pass']:raise ValueError('Parent k2 physical/hash gate failed')
 for symbol,item in m['physical']['pp'].items():
  if sha(item['path'])!=item['sha256']:raise ValueError('PP changed '+symbol)
 with (a.source/'state.pkl').open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.d3ft.parallel.k_pool=True;c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(a.source/'input.xyz')
 c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/'charge_gate.jsonl'))
 with (streamed_kpool_patch() if a.stream_k else nullcontext()),plain_fft_patch(k_pool=True),local_simpson_patch(),memory_bounded_xc(),source_radial_program(m,a.source) as info,validated_radial_splines():
  stage('reconstruct_new_k2_parent')
  parent=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'))
  if len(parent.k_frac)!=8 or parent.core_density_grid is not None:raise ValueError('Non-NLCC eight-k parent required')
  if a.theta_json is None:
   scale=np.exp(a.nu/3);cell=np.asarray(parent.crystal.cell_vectors)*scale;positions=np.asarray(parent.crystal.positions)*scale;theta=None
  else:
   from geometry_layout import GeometryLayout
   theta=np.asarray(json.loads(a.theta_json.read_text()),float);origin=np.asarray(json.loads(a.parent_theta_json.read_text()),float)
   layout=GeometryLayout(len(parent.crystal.positions),origin,include_coordinates=True)
   cell,positions=layout.deform(parent.crystal.cell_vectors,parent.crystal.positions,__import__('jax.numpy',fromlist=['asarray']).asarray(theta-origin));cell=np.asarray(cell);positions=np.asarray(positions)
  factor=.529177210903*float(ANGSTROM2BOHR)
  _write_xyz(a.output/'input.xyz',cell/factor,positions/factor,m['physical']['symbols'])
  del parent;jax.clear_caches();gc.collect()
  c.crystal_file_path_path=str(a.output/'input.xyz');c.epoch=a.epochs;c.optimizer_args.learning_rate=.01;c.d3ft.optim.lr_schedule='cosine';c.d3ft.optim.lr_final=1e-6
  stage('fresh_hydro_reoptimization',nu=a.nu,epochs=a.epochs)
  out=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'));jax.block_until_ready(out.params_pw);out.grid_kmax=info['grid_kmax']
  if np.max(abs(np.asarray(out.crystal.cell_vectors)-cell))>2e-11 or np.max(abs(np.asarray(out.crystal.positions)-positions))>2e-11:raise ValueError('Homothetic geometry I/O mismatch')
  np.save(a.output/'mask.npy',np.asarray(out.freq_mask));np.save(a.output/'density.npy',np.asarray(out.density_grid));np.save(a.output/'energy_history.npy',np.asarray(out.energy_history))
  with (a.output/'state.pkl').open('xb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ)),f,protocol=5)
  table=a.output/'beta_tables.pkl';table.write_bytes((a.source/info['tables_file']).read_bytes())
  jax.clear_caches();gc.collect();stage('forces_stress_and_stationarity')
  report=fused_report(out);forces=force_report(out);res=float(report['variational_residual']);asr=float(forces['acoustic_sum_rule']);charge=float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell)))
  point=dict(energy_Ha=float(out.total_energy),residual=res,acoustic_sum_rule_Ha_Bohr=asr,
   forces_Ha_Bohr=np.asarray(forces['forces_Ha_Bohr']).tolist(),stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),
   cell_bohr=cell.tolist(),positions_bohr=positions.tolist(),charge=charge,electrons=96.,k_fractional=np.asarray(out.k_frac).tolist(),
   nu_log_volume=a.nu,theta=None if theta is None else theta.tolist(),ionic_condition='fixed fractional coordinates' if theta is None else 'bounded joint strain/Cartesian trial',stationarity_pass=res<=1e-8,structural_relaxation_performed=False)
  dump(a.output/'point.json',point)
  physical={**m['physical'],'cell_angstrom':(cell/float(ANGSTROM2BOHR)).tolist(),
            'positions_angstrom':(positions/float(ANGSTROM2BOHR)).tolist()}
  saved_config=c.to_dict();saved_config['d3ft']['charge_gate'].pop('report_path',None)
  dump(a.output/'metadata.json',dict(config=saved_config,physical=physical,state_sha256=sha(a.output/'state.pkl'),source=str(a.source),
   source_hashes={n:sha(a.source/n) for n in ['metadata.json','state.pkl','density.npy','mask.npy','input.xyz']},
   frozen_radial_program={**info,'tables_file':'beta_tables.pkl','tables_sha256':sha(table)},
   new_response_state=True,nu_log_volume=a.nu,ionic_condition='fixed fractional coordinates',
   theta=None if theta is None else theta.tolist(),coordinate_unit='bohr',jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
   code_sha256={str(q.relative_to(P)):sha(q) for q in (P/'d3ft').rglob('*.py')}))
  if res>1e-8 or asr>5e-8 or abs(charge-96)>96e-8:raise ValueError(f'Hydro physical gate failed: residual={res}, ASR={asr}, charge={charge}')
  stage('completed',passed=True,residual_Ha=res)
except Exception as exc:stage('failed',error=str(exc));raise
finally:probe.finish()
