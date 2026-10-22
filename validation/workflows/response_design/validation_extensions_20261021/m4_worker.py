"""48-coordinate ablation of frozen E4 backend; six strain inputs fixed."""
import argparse,json,sys,copy,datetime
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.protocol.read_text());R=Path(p['runtime']['gamma'])
sys.path[:0]=[str(R),str(R/'validation/workflows/response_design'),str(R/'validation/workflows/response_design/reverse_response'),str(R/'validation/workflows/phase3')]
from joint_design_dft import GammaDFTBackend
from joint_design import run_design,dump
class FixedCellBackend(GammaDFTBackend):
 def geometry(self,theta):
  if not np.array_equal(np.asarray(theta)[:6],self.theta0[:6]):raise ValueError('Strain must remain bit-identical to E4 initial')
  return super().geometry(theta)
 def gradient(self,theta,current,folder):
  full=np.asarray(super().gradient(theta,current,folder));g=full.copy();g[:6]=0
  dump(Path(folder)/'active48_gradient.json',{'full54_gradient_eV':full.tolist(),'active48_masked_gradient_eV':g.tolist(),'strain_frozen':True});return g
backend=FixedCellBackend(p['backend_protocol']);result=run_design(a.output,p['theta0'],p['target_eV'],p['limits'],backend,max_updates=3,max_backtracks=p['max_backtracks'])
trajectory=json.loads((a.output/'trajectory.json').read_text()) if (a.output/'trajectory.json').exists() else [{'step':0,'theta':p['theta0'],'readout':json.loads((a.output/'initial/readout.json').read_text())}]
rows=[]
for x in trajectory:
 r=x['readout'];force=np.asarray(r['forces_Ha_Bohr']);assert np.array_equal(np.asarray(x['theta'])[:6],np.asarray(p['theta0'])[:6]);rows.append({'step':x['step'],'gap_eV':r['gap_eV'],'target_error_eV':r['gap_eV']-p['target_eV'],'energy_Ha':r['energy_Ha'],'energy_change_meV_atom':(r['energy_Ha']-trajectory[0]['readout']['energy_Ha'])*27.211386245988*1000/16,'max_force_norm_Ha_Bohr':float(np.max(np.linalg.norm(force,axis=1))),'max_force_component_Ha_Bohr':float(np.max(abs(force)))})
result.update(passed=bool(result['target_reached']),completed=True,physical_checks_inherited=True,active_coordinates=48,strain_frozen=True,trajectory_summary=rows)
dump(a.output/'acceptance.json',result);print('M4-A_COMPLETE '+json.dumps(result),flush=True)
