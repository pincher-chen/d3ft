"""Merge independent physical response shards, reject missing axes or memory."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();E=a.root
read=lambda p:json.loads(p.read_text())
r=read(E/'mg_reverse/result.json');g=np.asarray(r['results']['gap_gradient_eV'])
assert r['n_variables']==54 and r['results']['response_record']['converged'] is True
assert g.shape==(54,) and np.isfinite(g).all()
forward={};memory=[]
for tag in ['mg_reverse','mg_forward_0','mg_forward_1','mg_forward_2']:
 result=read(E/tag/'result.json');assert result['primal_residual_Ha']<=1e-7
 probe=read(E/tag/'memory_peak.json');memory.append(dict(tag=tag,**probe))
 if tag!='mg_reverse':
  for axis,row in result['results'].items():
   assert axis not in forward and row['response_record']['converged'] is True
   forward[axis]=row['gap_derivative_eV']
assert set(forward)==set(map(str,range(54)))
t=np.asarray([forward[str(i)] for i in range(54)]);diff=abs(g-t)
old=read(E.parent/'response_design_20261014_e0_migration/mgsin2_reverse_retry2/result.json')['results']['gap_gradient_eV']
translation=g[6:].reshape(16,3).sum(axis=0);translation_t=t[6:].reshape(16,3).sum(axis=0)
checks=dict(full_54_mode_agreement=bool(np.all(diff<=5e-6+1e-6*np.maximum(abs(g),abs(t)))),
 translation_reverse=bool(max(abs(translation))<=1e-6),translation_forward=bool(max(abs(translation_t))<=1e-6),
 original_six_strain_regression=bool(max(abs(g[:6]-np.asarray(old)))<=5e-6),
 nontrivial_coordinate_gradient=bool(np.linalg.norm(g[6:])>1e-5),
 sampled_memory_and_UUID_recorded=all(m['samples']>0 and m['gpu_UUIDs'] and m['peak_used_MiB'] is not None for m in memory))
result=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,
 reverse_gradient_eV=g.tolist(),forward_gradient_eV=t.tolist(),max_mode_error=float(max(diff)),
 reverse_uniform_translation_eV_Bohr=translation.tolist(),forward_uniform_translation_eV_Bohr=translation_t.tolist(),
 six_strain_regression_max=float(max(abs(g[:6]-np.asarray(old)))),memory=memory,
 scope='Full component/memory substep only; mixed h/h2 reoptimized FD and formal timing NOT complete')
(E/'mg_mode_acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result),flush=True)
if not result['passed']:raise RuntimeError('Mg mode/translation/memory gate failed')
