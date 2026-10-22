"""Bounded strain + Cartesian scalar-gap design with genuine backend readouts.

Pure NumPy controller; electronic backend is an injected callable. A successful
geometry proposal or surrogate dry run is never an accepted DFT endpoint.
Canonical order: six engineering strains, then Cartesian increments in Bohr.
"""
import itertools,json
from pathlib import Path
import numpy as np

HA_EV=27.211386245988

def matrix(v):
 return np.array([[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]])

def dump(path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def translation_free(x):
 x=np.array(x,float,copy=True);u=x[6:].reshape(-1,3);u-=u.mean(axis=0);return x

def atom_ball(u,radius):
 u=u-u.mean(axis=0)
 return u*min(1.,radius/max(float(np.max(np.linalg.norm(u,axis=1))),1e-30))

def deform(base_cell,base_positions,theta):
 theta=translation_free(theta);f=np.eye(3)+matrix(theta[:6])
 return np.asarray(base_cell)@f.T,np.asarray(base_positions)@f.T+theta[6:].reshape(-1,3)

def nearest_distance(cell,positions):
 cell=np.asarray(cell);positions=np.asarray(positions)
 # Wrap first; explicit image enumeration is sound for these mildly deformed
 # orthorhombic/fcc cells. Refuse strongly ill-conditioned cells.
 if np.linalg.cond(cell)>5:raise ValueError('Cell outside nearest-image guard domain')
 frac=positions@np.linalg.inv(cell);frac-=np.floor(frac)
 delta=frac[:,None,:]-frac[None,:,:]
 best=np.inf
 for image in itertools.product(range(-2,3),repeat=3):
  distances=np.linalg.norm((delta+image)@cell,axis=-1)
  if image==(0,0,0):np.fill_diagonal(distances,np.inf)
  best=min(best,float(np.min(distances)))
 return best

def proposal(theta,origin,gradient,gap,target,limits,backtrack=1.):
 theta,origin,gradient=map(lambda x:np.asarray(x,float),(theta,origin,gradient))
 if theta.shape!=origin.shape or gradient.shape!=theta.shape or len(theta)<9 or (len(theta)-6)%3:raise ValueError('Canonical geometry shape')
 if not np.isfinite(np.r_[theta,origin,gradient,gap,target]).all():raise ValueError('Nonfinite design input')
 gradient=translation_free(gradient)
 scale=np.r_[np.full(6,limits['strain_step']),np.full(len(theta)-6,limits['atomic_step_bohr'])]
 scaled=gradient*scale;squared=float(scaled@scaled)
 if squared<1e-20:raise RuntimeError('Zero scaled gap gradient')
 dz=(target-gap)*scaled/squared;dz*=min(1.,backtrack/max(np.linalg.norm(dz),1e-30))
 step=translation_free(dz*scale)
 step[:6]*=min(1.,limits['strain_step']/max(np.linalg.norm(step[:6]),1e-30))
 step[6:]=atom_ball(step[6:].reshape(-1,3),limits['atomic_step_bohr']).reshape(-1)
 total=translation_free(theta+step-origin)
 total[:6]*=min(1.,limits['strain_total']/max(np.linalg.norm(total[:6]),1e-30))
 total[6:]=atom_ball(total[6:].reshape(-1,3),limits['atomic_total_bohr']).reshape(-1)
 trial=origin+total
 # The prior accepted state is feasible; scale towards it if projections at
 # intersecting total/step bounds increased the actual Cartesian change.
 delta=trial-theta
 factor=min(1.,limits['strain_step']/max(np.linalg.norm(delta[:6]),1e-30),
            limits['atomic_step_bohr']/max(np.max(np.linalg.norm(delta[6:].reshape(-1,3),axis=1)),1e-30))
 trial=translation_free(theta+factor*delta)
 if np.linalg.norm(trial-theta)<1e-12:raise RuntimeError('Bounded design stalled')
 return trial

def physical_checks(readout,reference,limits):
 required=['energy_Ha','gap_eV','residual_Ha','forces_Ha_Bohr','ASR_Ha_Bohr','charge','electrons','ritz_max_Ha','nearest_distance_bohr','cell_determinant_bohr3','branch_margin_Ha','electronic_reoptimized']
 if any(k not in readout for k in required):raise ValueError('Missing true electronic/spectral readout')
 force=np.asarray(readout['forces_Ha_Bohr'])
 return dict(reoptimized=readout['electronic_reoptimized'] is True,
  finite=bool(np.isfinite(np.r_[readout['energy_Ha'],readout['gap_eV'],force.reshape(-1)]).all()),
  stationary=readout['residual_Ha']<=1e-8,ASR=readout['ASR_Ha_Bohr']<=5e-8,
  charge=abs(readout['charge']-readout['electrons'])<=1e-8*readout['electrons'],
  eigenpairs=readout['ritz_max_Ha']<=1e-9,branch=readout['branch_margin_Ha']>1e-5,
  energy=(readout['energy_Ha']-reference['energy_Ha'])*HA_EV/len(force)<=limits['energy_rise_eV_atom'],
  distances=readout['nearest_distance_bohr']>=limits['distance_fraction']*reference['nearest_distance_bohr'],
  volume=readout['cell_determinant_bohr3']>0,
  force_cap=float(abs(force).max())<=limits['force_cap_Ha_Bohr'])

def run_design(root,theta0,target,limits,backend,max_updates=3,max_backtracks=3):
 root=Path(root);root.mkdir(exist_ok=False);theta=translation_free(theta0)
 current=backend.evaluate(theta,root/'initial',None);reference=current
 target=float(target(current) if callable(target) else target)
 dump(root/'target.json',dict(target_eV=target,defined_once_before_updates=True))
 checks=physical_checks(current,reference,limits)
 dump(root/'initial_checks.json',checks)
 if not all(checks.values()):raise RuntimeError('Initial physical state fails gates')
 trajectory=[dict(step=0,theta=theta.tolist(),readout=current)]
 for iteration in range(max_updates):
  if abs(current['gap_eV']-target)<=limits['target_tolerance_eV']:break
  gradient=np.asarray(backend.gradient(theta,current,root/f'gradient_{iteration}'))
  dump(root/f'gradient_{iteration}.json',dict(theta=theta.tolist(),gradient_eV=gradient.tolist(),unit='strain then Bohr',translation_sum=gradient[6:].reshape(-1,3).sum(axis=0).tolist()))
  if np.max(abs(gradient[6:].reshape(-1,3).sum(axis=0)))>1e-6:raise RuntimeError('Response translation zero mode failed')
  accepted=False
  for attempt in range(max_backtracks):
   folder=root/f'trial_{iteration}_{attempt}';folder.mkdir()
   trial=proposal(theta,theta0,gradient,current['gap_eV'],target,limits,backtrack=.5**attempt)
   dump(folder/'proposal.json',dict(theta=trial.tolist(),parent_theta=theta.tolist(),target_eV=target))
   cell,pos=backend.geometry(trial);distance=nearest_distance(cell,pos)
   if np.linalg.det(cell)<=0 or distance<limits['distance_fraction']*reference['nearest_distance_bohr']:
    dump(folder/'decision.json',dict(accepted=False,reason='pre-SCF geometry guard',nearest_distance_bohr=distance));continue
   try:
    result=backend.evaluate(trial,folder/'electronic_state',current)
    gates=physical_checks(result,reference,limits)
    gates['actual_loss_decrease']=abs(result['gap_eV']-target)<abs(current['gap_eV']-target)
   except Exception as exc:
    dump(folder/'decision.json',dict(accepted=False,reason='electronic/backend failure',error=str(exc)));continue
   accepted=all(gates.values());dump(folder/'decision.json',dict(accepted=accepted,checks=gates,readout=result))
   if accepted:
    theta=trial;current=result;trajectory.append(dict(step=iteration+1,theta=theta.tolist(),readout=current));break
  dump(root/'trajectory.json',trajectory)
  if not accepted:break
 result=dict(completed_updates=len(trajectory)-1,target_eV=target,final_gap_eV=current['gap_eV'],
             target_reached=abs(current['gap_eV']-target)<=limits['target_tolerance_eV'],
             theta=theta.tolist(),structural_relaxation_performed=False,
             scope='Constrained joint design; nonzero force states are not relaxed equilibrium',
             backend_kind=backend.kind,physical_DFT_result=backend.kind=='dft',
             CPU_dry_run=bool(getattr(backend,'p',{}).get('CPU_dry_run',False)),
             paper_quality_result=False)
 dump(root/'result.json',result);return result
