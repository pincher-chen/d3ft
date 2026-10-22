"""M7 CPU-only contract regression. Synthetic responses are NOT DFT evidence."""
import copy,importlib.util,itertools,json,sys,tempfile
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;P=W.parents[3]
R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
D=Path(json.loads((R/'M7-A/current.json').read_text())['root']);p=json.loads((D/'protocol.json').read_text())
sys.path.insert(0,str(Path(p['runtime']['gamma'])/'validation/workflows/response_design'))
import joint_design as jd
spec=importlib.util.spec_from_file_location('m7worker',W/'m7_strain_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.dump=jd.dump
e4=json.loads(Path(p['baseline_E4_protocol']).read_text())
assert p['theta0']==e4['theta0'] and p['limits']==e4['limits'] and p['target_eV']==4.510987370941646
assert p['max_updates']==5 and p['max_backtracks']==e4['max_backtracks']==3
assert p['limits']['strain_step']==.003 and p['limits']['strain_total']==.03
assert p['backend_protocol']['grid']==e4['grid']==144 and p['backend_protocol']['epochs']==e4['epochs']==2000
assert p['backend_protocol']['cutoff_Ha']==e4['cutoff_Ha']==100

class FakeBackend:
    kind='fake_CPU_contract'
    def __init__(self,protocol):
        self.p=protocol;self.theta0=np.asarray(protocol['theta0']);self.basecell=np.eye(3)*10
        self.basepos=(np.array(list(itertools.product(range(4),range(2),range(2))))+.2)*[2,4,4]
    def geometry(self,theta):return jd.deform(self.basecell,self.basepos,theta)
    def gradient(self,theta,current,folder):
        Path(folder).mkdir();u=np.arange(48).reshape(16,3)/100;u-=u.mean(axis=0)
        return np.r_[7.4,np.zeros(5),u.reshape(-1)]
    def evaluate(self,theta,folder,parent):
        Path(folder).mkdir();cell,pos=self.geometry(theta)
        return dict(energy_Ha=-10.,gap_eV=float(p['target_eV']-.1+7.4*(theta[0]-self.theta0[0])),
             residual_Ha=1e-10,forces_Ha_Bohr=np.zeros((16,3)).tolist(),ASR_Ha_Bohr=1e-10,
             charge=96.,electrons=96.,ritz_max_Ha=1e-11,nearest_distance_bohr=jd.nearest_distance(cell,pos),
             cell_determinant_bohr3=float(np.linalg.det(cell)),branch_margin_Ha=.001,
             electronic_reoptimized=True,state_directory=str(folder))

with tempfile.TemporaryDirectory(prefix='M7_CPU_contract_') as tmp:
    tmp=Path(tmp);bp=copy.deepcopy(p['backend_protocol']);bp['event_file']=str(tmp/'events.jsonl')
    b=m.backend_class(FakeBackend)(bp);theta=np.array(p['theta0']);cell,pos=b.geometry(theta)
    bad=theta.copy();bad[6]=1e-5
    try:b.geometry(bad)
    except ValueError:pass
    else:raise AssertionError('Moved coordinate accepted')
    grad=b.gradient(theta,None,tmp/'gradient');assert np.array_equal(grad[6:],np.zeros(48)) and grad[0]==7.4
    result=jd.run_design(tmp/'design',theta,p['target_eV'],p['limits'],b,max_updates=5,max_backtracks=3)
    assert result['completed_updates']==5 and result['target_reached']
    tr=json.loads((tmp/'design/trajectory.json').read_text());rows=m.report_trajectory(tr,b,p)
    assert len(rows)==6 and abs(rows[3]['target_error_meV'])>2 and abs(rows[5]['target_error_meV'])<=2
    assert all(r['strain_step_L2']<=.003+2e-14 and r['strain_total_L2']<=.03+2e-14 and r['fractional_max_error']<2e-12 for r in rows)
    good=tr[0]['readout'];checks=jd.physical_checks(good,good,p['limits']);assert all(checks.values())
    cases=[('residual_Ha',1.1e-8,'stationary'),('ASR_Ha_Bohr',6e-8,'ASR'),('charge',96.001,'charge'),
           ('ritz_max_Ha',2e-9,'eigenpairs'),('branch_margin_Ha',1e-5,'branch'),('energy_Ha',0.,'energy'),
           ('nearest_distance_bohr',.5*good['nearest_distance_bohr'],'distances'),('cell_determinant_bohr3',-1.,'volume')]
    for name,value,key in cases:
        bad=copy.deepcopy(good);bad[name]=value;assert not jd.physical_checks(bad,good,p['limits'])[key]
    bad=copy.deepcopy(good);bad['forces_Ha_Bohr'][0][0]=.05001;assert not jd.physical_checks(bad,good,p['limits'])['force_cap']
    # Distinguish the original component cap from the norm used only in reporting.
    bad=copy.deepcopy(good);bad['forces_Ha_Bohr'][0]=[.04,.04,.04];assert jd.physical_checks(bad,good,p['limits'])['force_cap']
print(json.dumps({'passed':True,'CPU_only':True,'synthetic_not_DFT':True,'checks':['exactE4 settings','frozen48/affine fractionals','masked6 gradient','continues after3 to5','original physical gates reject violations','original component force cap retained','step/total norms','per-step timing rows']}))
