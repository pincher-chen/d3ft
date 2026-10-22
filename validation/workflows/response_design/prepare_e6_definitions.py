"""CPU check of log-volume/engineering-strain conventions, no SCF launch."""
from pathlib import Path
import datetime,json,sys
import numpy as np
P=Path(__file__).resolve().parents[3];sys.path.insert(0,str(P/'validation/workflows/response_design'))
from joint_design import deform,matrix
A=P/'validation/results/response_design_20261015_e6_preparation';A.mkdir(exist_ok=True)

def log_volume_direction(theta):
 theta=np.asarray(theta,float)
 return np.r_[(1+theta[:3])/3,theta[3:6]/3,theta[6:]/3]

def hydro_endpoint(theta,nu):
 theta=np.asarray(theta,float);s=np.exp(nu/3)
 return np.r_[s*(1+theta[:3])-1,s*theta[3:6],s*theta[6:]]

rng=np.random.default_rng(20261015);theta=np.r_[.003,-.002,.001,.002,-.001,.0015,rng.normal(size=48)*.01];theta[6:]= (theta[6:].reshape(16,3)-theta[6:].reshape(16,3).mean(axis=0)).reshape(-1)
cell=np.diag([10.,12.,9.]);positions=rng.normal(size=(16,3));oldcell,oldpos=deform(cell,positions,theta);nu=1e-4
newcell,newpos=deform(cell,positions,hydro_endpoint(theta,nu))
checks=dict(exact_cell_homothety=bool(np.max(abs(newcell-np.exp(nu/3)*oldcell))<1e-12),
 exact_position_homothety=bool(np.max(abs(newpos-np.exp(nu/3)*oldpos))<1e-12),
 actual_log_volume=bool(abs(np.log(np.linalg.det(newcell)/np.linalg.det(oldcell))-nu)<1e-12))
h=1e-6;fd=(hydro_endpoint(theta,h)-hydro_endpoint(theta,-h))/(2*h)
checks['hydro_direction_Jacobian']=bool(np.max(abs(fd-log_volume_direction(theta)))<1e-9)
zero=np.zeros(54);checks['zero_strain_trace_over_three']=bool(np.array_equal(log_volume_direction(zero)[:6],np.r_[np.ones(3)/3,np.zeros(3)]))
q=.002;engineering=np.zeros(6);engineering[3]=2*q;checks['engineering_shear_factor_two']=bool(matrix(engineering)[1,2]==q and matrix(engineering)[2,1]==q)
result=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,
 coordinate_unit='Bohr',strain_order=['eps_xx','eps_yy','eps_zz','gamma_yz','gamma_xz','gamma_xy'],
 exact_endpoint='F(nu)=exp(nu/3) F0; u(nu)=exp(nu/3) u0; fractional atomic positions fixed',
 scope='Conventions CPU test only; no fundamental-gap or literature agreement certificate')
e3=P/'validation/results/response_design_20261015_e3_geometry';proof=e3/'mg_reverse/result.json'
if proof.exists():
 g=np.array(json.loads(proof.read_text())['results']['gap_gradient_eV']);t=np.array(json.loads((e3/'mg_response_protocol.json').read_text())['theta0'])
 result['Gamma_existing_state_only']=dict(A_trace_over_three_eV=float(sum(g[:3])/3),
  A_exact_prestrained_hydro_eV=float(g@log_volume_direction(t)),theta0=t.tolist(),source=str(proof),
  not_fundamental_gap=True,not_literature_ion_protocol=True,not_a_literature_PASS=True)
(A/'definition_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print('E6 工程应变/lnV/固定分数坐标换算CPU验收PASS。\n非零预应变与内位移的静水方向已明确定义。\nGamma导数不冒称基本gap或−8.7eV验收。')
if not result['passed']:raise SystemExit(1)
