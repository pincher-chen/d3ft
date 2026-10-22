"""CPU-only factual endpoint metrics and qualified same-target comparison."""
import json,datetime as dt
from pathlib import Path
import numpy as np
from joint_design import deform,matrix,nearest_distance
BOHR_A=.529177210903
HA_EV=27.211386245988
FORCE_EV_A=51.42208619083232

def metrics(readout,theta,theta0,cell,pos,initial_cell,initial_pos,reference):
 f=np.asarray(readout['forces_Ha_Bohr']);du=(np.asarray(theta)-theta0)[6:].reshape(-1,3)
 return dict(gap_eV=readout['gap_eV'],energy_Ha=readout['energy_Ha'],energy_change_eV=(readout['energy_Ha']-reference['energy_Ha'])*HA_EV,
 energy_change_meV_atom=(readout['energy_Ha']-reference['energy_Ha'])*HA_EV*1000/len(f),
 max_single_atom_force_Ha_Bohr=float(np.linalg.norm(f,axis=1).max()),max_single_atom_force_eV_A=float(np.linalg.norm(f,axis=1).max()*FORCE_EV_A),
 max_force_component_Ha_Bohr=float(abs(f).max()),max_force_component_eV_A=float(abs(f).max()*FORCE_EV_A),
 nearest_distance_A=nearest_distance(cell,pos)*BOHR_A,nearest_distance_ratio=nearest_distance(cell,pos)/nearest_distance(initial_cell,initial_pos),
 max_internal_displacement_A=float(np.linalg.norm(du,axis=1).max()*BOHR_A),RMS_internal_displacement_A=float(np.sqrt(np.mean(np.sum(du*du,axis=1)))*BOHR_A),
 max_total_cartesian_displacement_A=float(np.linalg.norm(pos-initial_pos,axis=1).max()*BOHR_A),
 RMS_total_cartesian_displacement_A=float(np.sqrt(np.mean(np.sum((pos-initial_pos)**2,axis=1)))*BOHR_A),
 total_strain_norm=float(np.linalg.norm(np.asarray(theta[:6])-theta0[:6])),internal_translation_residual_Bohr=float(np.max(abs(du.sum(axis=0)))),
 residual_Ha=readout.get('residual_Ha',readout.get('residual')),ASR_Ha_Bohr=readout.get('ASR_Ha_Bohr',readout.get('acoustic_sum_rule_Ha_Bohr')),
 structural_equilibrium_claim=False)

def summarize(root):
 P=Path(__file__).resolve().parents[3];a=json.loads((root/'protocol.json').read_text());parent=Path(a['E4_root']);p=json.loads((parent/'protocol.json').read_text());trajectory=json.loads((parent/'design/trajectory.json').read_text());m=json.loads((Path(p['source'])/'metadata.json').read_text())
 factor=1.8897259886;origin=np.asarray(p['theta0']);inverse=np.linalg.inv(np.eye(3)+matrix(origin[:6])).T
 basecell=np.asarray(m['physical']['cell_angstrom'])*factor@inverse;basepos=np.asarray(m['physical']['positions_angstrom'])*factor@inverse
 initial_cell,initial_pos=deform(basecell,basepos,origin);reference=trajectory[0]['readout'];rows=[]
 for row in trajectory:
  cell,pos=deform(basecell,basepos,np.asarray(row['theta']));rows.append(dict(step=row['step'],**metrics(row['readout'],row['theta'],origin,cell,pos,initial_cell,initial_pos,reference)))
 end=json.loads((parent/'GPU_launch.json').read_text());initial_done=None
 for line in (parent/'design.log').read_text().splitlines():
  if line.startswith('E4_STAGE '):
   v=json.loads(line[len('E4_STAGE '):])
   if v['stage']=='readout_complete' and Path(v['folder']).name=='initial':initial_done=dt.datetime.fromisoformat(v['time']).replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
 design_hours=(dt.datetime.fromisoformat(end['end'])-initial_done).total_seconds()/3600 if initial_done else None
 six=P/'validation/results/response_design_20261006_six_strain_design_v2';sp=json.loads((six/'protocol.json').read_text());hist=json.loads((six/'reverse_history.json').read_text());sr=json.loads((six/'reverse_summary.json').read_text());point=json.loads((Path(hist[-1]['source'])/'point.json').read_text());point['gap_eV']=hist[-1]['gap_eV'];six_theta=np.r_[hist[-1]['theta'],np.zeros(48)]
 sm=metrics(point,six_theta,origin,np.asarray(point['cell_bohr']),np.asarray(point['positions_bohr']),initial_cell,initial_pos,reference)
 result=dict(time=dt.datetime.now().astimezone().isoformat(),E4_root=str(parent),source=p['source'],target_eV=p['target_eV'],
 trajectory=rows,endpoint=rows[-1],independent_endpoint_audit='pending',
 costs=dict(joint54_full_GPU_hours=end['GPU_hours'],joint54_after_initial_readout_GPU_hours=design_hours,joint54_completed_updates=len(rows)-1,
 six_strain_reverse_updates=len(hist)-1,six_strain_reverse_GPU_hours=sr['subprocess_seconds']/3600,six_strain_reverse_response_GPU_hours=sr['response_seconds']/3600,
 six_strain_reverse_ground_GPU_hours=sr['ground_seconds']/3600,six_strain_reverse_readout_GPU_hours=sr['readout_seconds']/3600),
 six_strain_endpoint=sm,comparison=dict(same_target=abs(sp['target_gamma_eV']-p['target_eV'])<1e-12,same_start_source=sp['base_source']==p['source'],
 common_1meV_target_gate_joint=abs(rows[-1]['gap_eV']-p['target_eV'])<=.001,common_1meV_target_gate_six=abs(hist[-1]['gap_eV']-p['target_eV'])<=.001,
 joint_strain_step=p['limits']['strain_step'],six_strain_step=sp['step_radius'],joint_electronic_epochs=p['epochs'],six_electronic_epochs=sp['design_epochs'],
 matched_optimizer_and_timing_protocol=False,formal_speedup_claim=False,
 caveat='54-variable steps cap strain at .003 vs .01 for old6; 2000 vs3000 epochs; initial reoptimization included only for54, different nodes/CPU settings. Descriptive costs, not a controlled speedup/iteration comparison.'),
 scope='Constrained gap design; electronic stationarity does not establish ionic equilibrium; intrinsic displacement excludes affine strain')
 (root/'preliminary_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
