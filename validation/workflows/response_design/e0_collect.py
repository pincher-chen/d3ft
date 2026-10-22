"""Collect E0 gates without modifying inputs, accepting partial or failed states."""
import json,datetime,hashlib
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261014_e0_migration'
def load(p):return json.loads(p.read_text()) if p.exists() else None
def main():
 now=datetime.datetime.now().isoformat();r=dict(time=now,passed=False,completed=False,gates={},jobid=250285)
 si=load(E/'checks/si_final_tail_parity.json');r['gates']['si']=si
 qpath=E/'alp_qe/scratch/AlP_e0.save/data-file-schema.xml';alp=[]
 if qpath.exists():
  out=ET.parse(qpath).getroot().find('output');en=float(out.findtext('total_energy/etot'));f=np.fromstring(out.findtext('forces'),sep=' ').reshape(2,3);s=np.fromstring(out.findtext('stress'),sep=' ').reshape(3,3);protocol=load(E/'inputs/AlP_NLCC/protocol.json');limits=protocol['acceptance']
  for label in ['alp_new_retry1','alp_new_qe_beta_retry1','alp_new_qe_local','alp_new_qe_tail']:
   point=load(E/label/'point.json')
   if not point:continue
   errors=dict(energy_Ha_cell=abs(point['energy_Ha']-en),forces_Ha_Bohr=float(np.max(abs(np.asarray(point['forces_Ha_Bohr'])-f))),stress_Ha_Bohr3=float(np.max(abs(np.asarray(point['stress_Ha_Bohr3'])+s))),stationarity=point['residual']);tests={k:v<=limits[k] for k,v in errors.items()}
   meta=load(E/label/'metadata.json');same_pp=all(meta['physical']['pp'][symbol]['sha256']==h for symbol,h in protocol['pp'].items());cell=np.array([np.fromstring(out.findtext(f'atomic_structure/cell/a{i}'),sep=' ') for i in [1,2,3]]);same_cell=float(np.max(abs(cell-np.asarray(point['cell_bohr']))))<1e-12;qe_converged=out.findtext('convergence_info/scf_conv/convergence_achieved')=='true';tests.update(same_pp=same_pp,same_cell=same_cell,QE_converged=qe_converged)
   alp.append(dict(run=label,errors=errors,tests=tests,passed=all(tests.values()),default_local_and_beta=label in ['alp_new_retry1','alp_new_qe_tail'],energy_shift_fitted=False))
 r['gates']['alp']=dict(comparisons=alp,passed=any(x['passed'] for x in alp),note='Default and opt-in reference quadrature results separately reported; no claim that a common old/new error was caused by NLCC.')
 fw=load(E/'mgsin2_forward/result.json');rv=load(E/'mgsin2_reverse_retry2/result.json')
 if fw and rv:
  f=fw['results']['0'];v=rv['results'];delta=abs(f['gap_derivative_eV']-v['gap_gradient_eV'][0]);tests=dict(mode_difference=delta<1e-6,reverse_converged=v['response_record']['converged'],forward_converged=f['response_record']['converged'],reverse_primal=rv['primal_residual_Ha']<1e-7,forward_primal=fw['primal_residual_Ha']<1e-7,reverse_symmetry=rv['kernel_symmetry_relative_error']<1e-8,forward_symmetry=fw['kernel_symmetry_relative_error']<1e-8)
  r['gates']['mgsin2']=dict(difference_eV=delta,reverse_gradient_eV=v['gap_gradient_eV'],forward_axis=0,forward_derivative_eV=f['gap_derivative_eV'],tests=tests,passed=all(tests.values()),scope='Fresh one-component mode agreement on owned Gamma state; six-component prior evidence not relabeled as new')
 else:r['gates']['mgsin2']=dict(passed=False,status='pending',reverse_complete=bool(rv),forward_complete=bool(fw))
 protected=load(E/'protected_artifacts.json');changed=[str(p) for p,h in protected.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];r['protected_artifacts_unchanged']=not changed;r['protected_artifact_changes']=changed
 r['passed']=bool(si and si['passed'] and r['gates']['alp']['passed'] and r['gates']['mgsin2']['passed'] and not changed);r['completed']=r['passed']
 path=E/'checks/e0_physical_acceptance.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(path);print(json.dumps(dict(passed=r['passed'],si=bool(si and si['passed']),alp=r['gates']['alp']['passed'],mgsin2=r['gates']['mgsin2']['passed'],protected=not changed)))
 return r
if __name__=='__main__':main()
