"""Convert fixed-volume elastic derivatives to equilibrium derivatives.

At P=0, d ln(V)/dp = (partial P/partial p)/B. For an energy
curvature Q=(E+ + E- - 2E0)/(n V h²), isotropic scaling gives
partial Q/partial ln(V) = -(J+ P+ + J- P- - 2P0)/(n h²) - Q.
J is the strain determinant. Relaxed shear points must also be force stationary.
This uses the stress already computed at each point; no extra volume scan.
The formula is for the finite-strain estimator itself, before h extrapolation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from force_acceptance import check_force_sum

def volume_derivative(value, plus, minus, center, denominator, h):
    weighted=0.
    for r in (plus,minus):
        jac=float(np.linalg.det(np.eye(3)+np.asarray(r['strain'])))
        weighted+=jac*r['pressure_GPa']
    return -(weighted-2*center['pressure_GPa'])/(denominator*h*h)-value

def derive(work):
    result=json.loads((work/'summary.json').read_text())
    points=json.loads((work/'points.json').read_text())
    for point in points:
        for key in ['residual','energy_Ha','pressure_GPa','volume_bohr3','dE_dp']:
            if not np.isfinite(np.asarray(point[key],float)).all():
                raise ValueError(f'Nonfinite {key} in {point["label"]}')
        if point['volume_bohr3']<=0:
            raise ValueError('Nonpositive cell volume')
    if not np.isfinite(result['B_GPa']) or result['B_GPa']<=0:
        raise ValueError('Nonpositive or nonfinite bulk modulus')
    stress_bulk=result['B_stress_GPa']
    if not np.isfinite(stress_bulk) or abs(result['B_GPa']-stress_bulk)>.1:
        raise ValueError('Energy/pressure bulk estimators disagree beyond the 0.1 GPa numerical budget; refine strain or electronic convergence')
    center=[p for p in points if p['label'].startswith('eq_')][-1]
    if max(p['residual'] for p in points)>1e-6: raise ValueError('Nonstationary input')
    if abs(center['pressure_GPa'])/(3*result['B_GPa'])>=1e-5: raise ValueError('Non-equilibrium input')
    arguments=json.loads((work/'metadata.json').read_text())['arguments']
    h=arguments.get('strain_step',.01)
    iso={round(p['strain'][0][0],6):p for p in points if p['label'].startswith('iso_')}
    # Fourth-order derivative of dE/dp along isotropic strain, with one fixed mask.
    g=lambda d: np.asarray(iso[round(d,6)]['dE_dp'])
    gradP=-(8*(g(h)-g(-h))-(g(2*h)-g(-2*h)))*29421.02648438959/(36*h*center['volume_bohr3'])
    dlogV=gradP/result['B_GPa']
    derived=dict(source=str(work),method_version=2,dP_fixed_volume_dp=gradP.tolist(),dlogV_equilibrium_dp=dlogV.tolist())
    # Apply the volume response to the actual Richardson energy estimator.
    # The +2P/3 term has zero TOTAL derivative along the equilibrium manifold.
    # This avoids approximating its volume response through a fitted B'.
    q=[]; qv=[]; qg=[]
    for step in [h,2*h]:
        plus,minus=iso[round(step,6)],iso[round(-step,6)]
        factor=29421.02648438959/(9*center['volume_bohr3']*step**2)
        value=(plus['energy_Ha']+minus['energy_Ha']-2*center['energy_Ha'])*factor
        q.append(value)
        qv.append(volume_derivative(value,plus,minus,center,9,step))
        qg.append((np.array(plus['dE_dp'])+np.array(minus['dE_dp'])-2*np.array(center['dE_dp']))*factor)
    derived['B_curvature_dlnV']=(4*qv[0]-qv[1])/3
    derived['dB_equilibrium_dp']=((4*qg[0]-qg[1])/3+derived['B_curvature_dlnV']*dlogV).tolist()
    for branch,key,denom,old,new in [
        ('tet_','C11_minus_C12_GPa',6,'dD_fixed_volume_dp','dD_equilibrium_dp'),
        ('shear_','C44_GPa',4,'dC44_fixed_volume_dp','dC44_equilibrium_dp')]:
        if key not in result: continue
        # Last shear iteration for each sign is the actually relaxed state.
        branchpoints={round(p['strain'][0][0] if branch=='tet_' else p['strain'][1][2],6):p
                      for p in points if p['label'].startswith(branch)}
        plus,minus=branchpoints[round(h,6)],branchpoints[round(-h,6)]
        if branch=='shear_':
            for point in (plus,minus):
                force=np.asarray(point['forces']); largest=np.abs(force).max()
                if largest>=1e-6: raise ValueError('Nonstationary internal coordinates')
                check_force_sum(force,floor=arguments.get('asr_force_floor',0.))
        qv=volume_derivative(result[key],plus,minus,center,denom,h)
        derived[key+'_dlnV']=qv
        derived[new]=(np.asarray(result[old])+qv*dlogV).tolist()
    return derived

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('work',type=Path); ap.add_argument('--out',type=Path); args=ap.parse_args()
    dest=args.out or args.work/'equilibrium_jacobian.json'
    with dest.open('x') as f: json.dump(derive(args.work),f,indent=2)
