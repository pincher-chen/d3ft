"""Three predetermined mixed directions, h/h2 independent reoptimized gaps."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();E=a.root
read=lambda p:json.loads(p.read_text());mode=read(E/'mg_mode_acceptance.json');assert mode['passed']
dirs=read(E/'mg_fd_directions.json');index=read(E/'mg_fd_index.json');g=np.asarray(mode['reverse_gradient_eV']);rows=[]
assert len(index)==12
for k,d in enumerate(dirs['directions']):
 for h in dirs['steps']:
  pair=[r for r in index if r['direction']==k and r['h']==h];assert len(pair)==2
  gaps={}
  for r in pair:
   root=Path(r['output']);m=read(root/'metadata.json');point=read(root/'point.json');s=read(root/'spectrum_summary.json')
   assert read(root/'status.json')['stage']=='completed'
   assert m['fixed_parent_miller_basis'] and m['fixed_parent_beta_tables'] and m['fixed_parent_local_radial_nodes']
   assert point['residual']<=1e-8 and point['acoustic_sum_rule_Ha_Bohr']<=5e-8
   assert s['maximum_ritz_residual_Ha']<=1e-9 and s['orthogonality_max']<=1e-10
   assert min(s['vbm_adjacent_separation_Ha'],s['cbm_adjacent_separation_Ha'])>=1e-5
   assert np.max(abs(np.asarray(point['positions_bohr'])-np.asarray(m['requested_positions_bohr'])))<2e-11
   assert np.max(abs(np.asarray(point['cell_bohr'])-np.asarray(m['requested_cell_bohr'])))<2e-11
   gaps[r['sign']]=s['gap_eV']
  assert set(gaps)=={-1,1}
  prediction=float(g@np.asarray(d));fd=(gaps[1]-gaps[-1])/(2*h);error=abs(fd-prediction)
  tolerance=3e-5+.002*abs(prediction)
  rows.append(dict(direction=k,h=h,reverse_directional_eV=prediction,reoptimized_FD_eV=fd,
   absolute_error=error,tolerance=tolerance,passed=error<=tolerance))
result=dict(time=datetime.datetime.now().isoformat(),passed=all(r['passed'] for r in rows),FD=rows,
 definition='eV per scalar mixed step t: 6 engineering strains and 48 Cartesian Bohr components',
 scope='All E3 development derivative gates complete if PASS; hn48 exclusive formal timings remain unsubmitted')
(E/'mg_fd_acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
if not result['passed']:raise RuntimeError('Mixed h/h2 FD gate failed; do not submit formal timings')
