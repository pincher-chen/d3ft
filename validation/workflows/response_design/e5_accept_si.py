"""Complete prototype acceptance with independently reoptimized h/h2 targets."""
from pathlib import Path
import json,sys,datetime
import numpy as np
E=Path(sys.argv[1]);refined='--refined' in sys.argv[2:];R=Path(json.loads((E/'protocol.json').read_text())['runtime']);sys.path.insert(0,str(R/'validation/workflows/response_design/reverse_response'))
from mesh_target import select_mesh_target
cfg=json.loads((E/'protocol.json').read_text());read=lambda p:json.loads(p.read_text())
gamma=read(E/'si_gamma_equivalence/result.json');response=read(E/'si_k2_response/result.json');assert gamma['passed'] and response['passed']
directions=read(E/('fd_directions_refined.json' if refined else 'fd_directions.json'));index=read(E/('fd_index_refined.json' if refined else 'fd_index.json'));g=np.array(response['gradient_eV']);rows=[]
assert len(index)==8
for i,d in enumerate(directions['directions']):
 for h in (directions['steps_per_direction'][i] if refined else directions['steps']):
  pair=[r for r in index if r['direction']==i and r['h']==h];assert len(pair)==2;targets={}
  for r in pair:
   source=Path(r['point']);m=read(source/'metadata.json');p=read(source/'point.json');spec=read(Path(r['spectrum'])/'summary.json')
   assert m['restart']['fixed_parent_miller_mask'] and m['frozen_radial_program']
   assert p['residual']<=1e-8 and p['acoustic_sum_rule_Ha_Bohr']<=5e-8 and spec['maximum_ritz_residual_Ha']<=1e-9
   assert np.array_equal(np.load(source/'mask.npy'),np.load(E/'si_k2_base/mask.npy'))
   _,target=select_mesh_target(spec['eigenvalues_Ha'],spec['n_occupied'],cfg['objective'],cfg.get('soft_tau_Ha'))
   if cfg['objective']=='hard':
    assert target['vbm_k_index']==response['target']['vbm_k_index'] and target['cbm_k_index']==response['target']['cbm_k_index']
   targets[r['sign']]=target['target_Ha']*27.211386245988
  assert set(targets)=={-1,1};fd=(targets[1]-targets[-1])/(2*h);prediction=float(g@np.array(d));error=abs(fd-prediction);limit=3e-5+.002*abs(prediction)
  rows.append(dict(direction=i,h=h,adjoint=prediction,reoptimized_FD=fd,error=error,limit=limit,passed=error<=limit))
report=dict(time=datetime.datetime.now().isoformat(),passed=all(r['passed'] for r in rows),FD=rows,
 Gamma_equivalence=gamma,k2=response,scope='Si non-NLCC single-GPU uniform k2 prototype only; not MgSiN2 design, distributed k2, k4 production or a convergence certificate')
(E/('si_prototype_acceptance_refined.json' if refined else 'si_prototype_acceptance.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
if not report['passed']:raise RuntimeError('E5 finite difference gate failed; no production')
