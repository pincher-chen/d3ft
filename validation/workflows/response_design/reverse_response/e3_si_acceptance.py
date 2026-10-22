"""Fail-closed Si coordinate response gates; never alters electronic results."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--stage',choices=['mode','full'],required=True)
ap.add_argument('--refined',action='store_true');a=ap.parse_args();E=a.root
def read(p):return json.loads(p.read_text())
S=read(E/'development_protocol.json')['Si'];T=S['thresholds']
outputs=read(E/'si_response_outputs.json')
rev=read(Path(outputs['reverse'])/'result.json');fwd=read(Path(outputs['forward'])/'result.json')
assert rev['n_variables']==fwd['n_variables']==12
assert rev['results']['response_record']['converged'] is True
g=np.asarray(rev['results']['gap_gradient_eV']);t=[]
for i in range(12):
 r=fwd['results'][str(i)];assert r['response_record']['converged'] is True
 t.append(r['gap_derivative_eV'])
t=np.asarray(t);assert np.isfinite(g).all() and np.isfinite(t).all()
assert max(rev['primal_residual_Ha'],fwd['primal_residual_Ha'])<=T['primal_residual_Ha']
diff=np.abs(g-t);tol=T['mode_absolute_eV']+T['mode_relative']*np.maximum(abs(g),abs(t))
translation_reverse=g[6:].reshape(2,3).sum(axis=0)
translation_forward=t[6:].reshape(2,3).sum(axis=0)
checks=dict(full_12_mode_comparison=bool(np.all(diff<=tol)),
 nontrivial_coordinate_gradient=bool(np.linalg.norm(g[6:])>=T['minimum_atomic_gradient_norm_eV_Bohr']),
 overall_translation_reverse=bool(max(abs(translation_reverse))<=T['translation_eV_Bohr']),
 overall_translation_forward=bool(max(abs(translation_forward))<=T['translation_eV_Bohr']))
result=dict(stage=a.stage,time=datetime.datetime.now().isoformat(),checks=checks,
 reverse_gradient=g.tolist(),forward_gradient=t.tolist(),max_mode_absolute_error=float(max(diff)),
 reverse_uniform_translation_eV_Bohr=translation_reverse.tolist(),
 forward_uniform_translation_eV_Bohr=translation_forward.tolist(),
 coordinate_gradient_norm_eV_Bohr=float(np.linalg.norm(g[6:])),
 definition='du added after affine strain, strain followed by atom-major xyz, Bohr coordinates; clamped-ion electronic reoptimization')
if a.stage=='full':
 prefix='si_refined' if a.refined else 'si'
 dirs=read(E/(prefix+'_fd_directions.json'));index=read(E/(prefix+'_fd_index.json'));rows=[]
 assert len(index)==8
 basepoint=read(E/'si_base/point.json');basemask=np.load(E/'si_base/mask.npy')
 for k,d in enumerate(dirs['directions']):
  d=np.asarray(d);prediction=float(g@d)
  for h in dirs['steps_Bohr']:
   entries=[r for r in index if r['direction']==k and r['h']==h]
   assert len(entries)==2 and {r['sign'] for r in entries}=={-1,1}
   gaps={}
   for r in entries:
    point=read(Path(r['point'])/'point.json');meta=read(Path(r['point'])/'metadata.json')
    assert point['residual']<=1e-8 and point['acoustic_sum_rule_Ha_Bohr']<=5e-8
    assert meta['restart']['fixed_parent_miller_mask'] is True
    assert np.array_equal(basemask,np.load(Path(r['point'])/'mask.npy'))
    # Assert actual displaced geometry, not merely requested CLI values.
    actual=np.asarray(point['positions_bohr'])-np.asarray(basepoint['positions_bohr'])
    assert np.max(abs(actual-r['sign']*h*d[6:].reshape(2,3)))<1e-10
    band=read(Path(r['bands'])/'summary.json');assert band['maximum_ritz_residual_Ha']<=1e-9
    assert min(band['vbm_adjacent_separation_Ha'],band['cbm_adjacent_separation_Ha'])>=1e-5
    gaps[r['sign']]=band['gap_eV']
   fd=(gaps[1]-gaps[-1])/(2*h);error=abs(fd-prediction)
   limit=T['FD_absolute_eV_Bohr']+T['FD_relative']*abs(prediction)
   rows.append(dict(direction=k,h_Bohr=h,adjoint_directional_eV_Bohr=prediction,
    reoptimized_FD_eV_Bohr=fd,absolute_error=error,limit=limit,passed=error<=limit))
 checks['two_atomic_directions_h_and_half_reoptimized_FD']=all(r['passed'] for r in rows)
 result['FD']=rows
result['passed']=all(checks.values());dest=E/('si_mode_acceptance.json' if a.stage=='mode' else ('si_acceptance_refined.json' if a.refined else 'si_acceptance.json'))
tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
print(json.dumps(result),flush=True)
if not result['passed']:raise RuntimeError('Si derivative acceptance failed; retain outputs, no downstream submission')
