"""Pure NumPy E9 controller; true energy/forces/spectrum are injected.

Protocol limits are read verbatim, never tuned after a result. SCF timeout
recovery belongs to the evaluation store; accepted ionic states are durable.
"""
from pathlib import Path
import time
import numpy as np
from common import read,dump
from joint_design import translation_free,atom_ball,proposal,HA_EV

def max_force(r):return float(np.max(np.linalg.norm(np.asarray(r['forces_Ha_Bohr']),axis=1)))
def atom_norm(theta):return float(np.max(np.linalg.norm(np.asarray(theta)[6:].reshape(-1,3),axis=1)))
def numerical_checks(r,p,spectrum):
 f=np.asarray(r['forces_Ha_Bohr'])
 okay=bool(np.isfinite(np.r_[r['energy_Ha'],f.reshape(-1)]).all()
  and r['residual_Ha']<=p['electronic_residual_Ha'] and r['ASR_Ha_Bohr']<=p['ASR_Ha_Bohr']
  and abs(r['charge']-r['electrons'])<=p['charge_error'] and r['cell_determinant_bohr3']>0)
 if spectrum:okay=okay and r['ritz_max_Ha']<=p['ritz_max_Ha'] and r['branch_margin_Ha']>p['branch_separation_Ha']
 return bool(okay)

def relax(root,theta,parent,backend,p,origin,reference=None):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);theta=np.array(theta,float);strain=theta[:6].copy()
 state=root/'ionic_state.json';n=theta.size-6
 if state.exists():
  s=read(state);theta=np.array(s['theta']);current=s['readout'];H=np.array(s['inverse_H']);accepted=s['accepted_steps'];evaluations=s['force_evaluations'];start=np.array(s['start_theta'])
 else:
  current=backend.evaluate(theta,root/'force_initial',parent,spectrum=False)
  if not numerical_checks(current,p,False):raise ValueError('Initial electronic force state failed')
  H=np.eye(n);accepted=0;evaluations=1;start=theta.copy()
  dump(state,dict(theta=theta.tolist(),readout=current,inverse_H=H.tolist(),accepted_steps=accepted,force_evaluations=evaluations,start_theta=start.tolist()))
 while max_force(current)>=p['relaxation_force_max_norm_Ha_Bohr']:
  if accepted>=p['max_ionic_accepted_steps_per_relaxation'] or evaluations>=p['max_energy_force_evaluations_per_relaxation']-1:
   raise RuntimeError('Registered ionic step/evaluation limit reached')
  g=-np.asarray(current['forces_Ha_Bohr']).reshape(-1);g=(g.reshape(-1,3)-g.reshape(-1,3).mean(axis=0)).reshape(-1)
  direction=(-H@g).reshape(-1,3)
  if float(g@direction.reshape(-1))>=0:direction=-g.reshape(-1,3);H=np.eye(n)
  direction=atom_ball(direction,p['ionic_step_radius_Bohr']).reshape(-1);found=False
  for backtrack in range(p['max_ionic_backtracks']):
   if evaluations>=p['max_energy_force_evaluations_per_relaxation']-1:break
   trial=theta.copy();trial[6:]+=direction*(.5**backtrack);trial=translation_free(trial)
   if atom_norm(trial-start)>p['ionic_total_radius_Bohr']+1e-12 or atom_norm(trial-origin)>p['max_total_internal_displacement_from_original_start_Bohr']+1e-12:
    continue
   cell,pos=backend.geometry(trial)
   from joint_design import nearest_distance
   if np.linalg.det(cell)<=0 or (reference and nearest_distance(cell,pos)<p['distance_fraction']*reference['nearest_distance_bohr']):continue
   candidate=backend.evaluate(trial,root/f'ionic_{accepted}_backtrack_{backtrack}',current,spectrum=False);evaluations+=1
   valid=numerical_checks(candidate,p,False) and candidate['energy_Ha']<current['energy_Ha']
   if reference:valid=valid and (candidate['energy_Ha']-reference['energy_Ha'])*HA_EV/len(candidate['forces_Ha_Bohr'])<=p['energy_rise_eV_atom']
   dump(root/f'ionic_{accepted}_backtrack_{backtrack}_decision.json',dict(accepted=bool(valid),theta=trial.tolist(),readout=candidate))
   if not valid:continue
   svec=(trial-theta)[6:];newg=-np.asarray(candidate['forces_Ha_Bohr']).reshape(-1);newg=(newg.reshape(-1,3)-newg.reshape(-1,3).mean(axis=0)).reshape(-1);y=newg-g;ys=float(y@svec)
   if ys>1e-12*np.linalg.norm(y)*np.linalg.norm(svec):
    rho=1/ys;v=np.eye(n)-rho*np.outer(svec,y);H=v@H@v.T+rho*np.outer(svec,svec)
   theta=trial;current=candidate;accepted+=1;found=True
   dump(state,dict(theta=theta.tolist(),readout=current,inverse_H=H.tolist(),accepted_steps=accepted,force_evaluations=evaluations,start_theta=start.tolist()))
   break
  if not found:raise RuntimeError('Registered ionic backtracking exhausted')
  if not np.array_equal(theta[:6],strain):raise ValueError('Internal relaxation changed fixed cell')
 result=backend.evaluate(theta,root/'relaxed_spectrum',current,spectrum=True)
 if not numerical_checks(result,p,True) or max_force(result)>=p['relaxation_force_max_norm_Ha_Bohr']:
  raise ValueError('Relaxed spectral readout violates registered force/numerical criteria')
 result.update(ionic_accepted_steps=accepted,ionic_force_evaluations=evaluations+1,max_force_norm_Ha_Bohr=max_force(result))
 dump(root/'relaxation_result.json',dict(theta=theta.tolist(),readout=result))
 return theta,result

def run(root,theta0,backend,p):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);theta0=np.array(theta0,float)
 theta,initial=relax(root/'initial_relaxation',theta0,None,backend,p,theta0)
 frozen=root/'target.json';target=initial['gap_eV']+p['target_offset_eV']
 if frozen.exists():
  prior=read(frozen)
  if abs(prior['target_eV']-target)>1e-12:raise ValueError('Relaxed-start target changed on restart')
 else:dump(frozen,dict(target_eV=target,initial_relaxed_gap_eV=initial['gap_eV'],defined_once_before_outer_updates=True))
 tracefile=root/'trajectory.json';trajectory=read(tracefile) if tracefile.exists() else [dict(step=0,theta=theta.tolist(),readout=initial)]
 theta=np.array(trajectory[-1]['theta']);current=trajectory[-1]['readout'];limits={k:p[k] for k in ['strain_step','strain_total','atomic_step_bohr','atomic_total_bohr']}
 accepted_origin=np.array(trajectory[0]['theta']);reference=initial
 for iteration in range(len(trajectory)-1,p['max_outer_accepted_updates']):
  if abs(current['gap_eV']-target)<=p['target_tolerance_eV']:break
  before=backend.spent();step_start=time.monotonic();gradient=np.asarray(backend.gradient(theta,current,root/f'gradient_{iteration}'))
  if not np.isfinite(gradient).all() or max(abs(gradient[6:].reshape(-1,3).sum(axis=0)))>1e-6:raise ValueError('Nonfinite/translation-violating adjoint')
  accepted=False
  for backtrack in range(p['max_outer_backtracks']):
   folder=root/f'outer_{iteration}_backtrack_{backtrack}';folder.mkdir(exist_ok=True)
   trial=proposal(theta,accepted_origin,gradient,current['gap_eV'],target,limits,backtrack=.5**backtrack)
   dump(folder/'proposal.json',dict(theta=trial.tolist(),parent_theta=theta.tolist(),target_eV=target))
   try:
    relaxed,result=relax(folder/'relaxation',trial,current,backend,p,theta0,reference)
    valid=bool(abs(result['gap_eV']-target)<abs(current['gap_eV']-target)
       and np.linalg.norm(relaxed[:6]-accepted_origin[:6])<=p['strain_total']+1e-12
       and atom_norm(relaxed-accepted_origin)<=p['atomic_total_bohr']+1e-12
       and atom_norm(relaxed-theta0)<=p['max_total_internal_displacement_from_original_start_Bohr']+1e-12
       and result['nearest_distance_bohr']>=p['distance_fraction']*reference['nearest_distance_bohr']
       and (result['energy_Ha']-reference['energy_Ha'])*HA_EV/len(result['forces_Ha_Bohr'])<=p['energy_rise_eV_atom'])
    decision=dict(accepted=valid,theta=relaxed.tolist(),readout=result)
   except Exception as exc:
    valid=False;decision=dict(accepted=False,error=repr(exc),failure_retained=True)
   dump(folder/'decision.json',decision)
   if valid:
    theta=relaxed;current=result;trajectory.append(dict(step=iteration+1,theta=theta.tolist(),readout=current));accepted=True;dump(tracefile,trajectory)
    if iteration==0:
     stats=[read(f) for f in root.glob('outer_0_backtrack_*/relaxation/ionic_state.json')]
     backend.first_step_report(dict(accepted=True,initial_relaxation_ionic_steps=initial['ionic_accepted_steps'],initial_force_evaluations=initial['ionic_force_evaluations'],design_step_ionic_steps=sum(x['accepted_steps'] for x in stats),design_force_evaluations=sum(x['force_evaluations'] for x in stats)+len(list(root.glob('outer_0_backtrack_*/relaxation/relaxation_result.json'))),accepted_candidate_ionic_steps=result['ionic_accepted_steps'],step_GPU_hours=backend.spent()-before,total_GPU_hours=backend.spent(),active_attempt_seconds=time.monotonic()-step_start,remaining_max_updates=p['max_outer_accepted_updates']-1,target_error_eV=result['gap_eV']-target))
    break
  if not accepted:
   if iteration==0:
    stats=[read(f) for f in root.glob('outer_0_backtrack_*/relaxation/ionic_state.json')]
    backend.first_step_report(dict(accepted=False,initial_relaxation_ionic_steps=initial['ionic_accepted_steps'],initial_force_evaluations=initial['ionic_force_evaluations'],design_step_ionic_steps=sum(x['accepted_steps'] for x in stats),design_force_evaluations=sum(x['force_evaluations'] for x in stats),step_GPU_hours=backend.spent()-before,total_GPU_hours=backend.spent(),active_attempt_seconds=time.monotonic()-step_start,remaining_max_updates=0,target_error_eV=current['gap_eV']-target))
   break
 audit=backend.evaluate(theta,root/'independent_endpoint',current,spectrum=True)
 audit_passed=numerical_checks(audit,p,True) and max_force(audit)<p['relaxation_force_max_norm_Ha_Bohr'] and abs(audit['gap_eV']-current['gap_eV'])<=p['target_tolerance_eV']
 target_passed=abs(audit['gap_eV']-target)<=p['target_tolerance_eV']
 result=dict(completed=True,passed=bool(audit_passed and target_passed),target_reached=bool(target_passed),audit_passed=bool(audit_passed),target_eV=target,final_gap_eV=audit['gap_eV'],max_force_norm_Ha_Bohr=max_force(audit),accepted_updates=len(trajectory)-1,theta=theta.tolist(),endpoint=audit,initial=initial,force_bound_Ha_Bohr=p['relaxation_force_max_norm_Ha_Bohr'],relaxed_ionic_design=True,relaxed_ion_implicit_derivative_claim=False)
 dump(root/'result.json',result);return result
