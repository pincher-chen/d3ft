"""Pure CPU surrogate tests; never claim these are physical MgSiN2 data."""
import ast,copy,json,sys,tempfile,unittest,types
from pathlib import Path
import numpy as np
from unittest.mock import patch
import e9_queue
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relaxed_design import run,relax,max_force
from common import read,dump

class FakeBackend:
 def __init__(self):self.cost=0.;self.report=None
 def geometry(self,t):return np.eye(3)*8,np.array([[1.,1.,1.],[4.,4.,4.]])+np.asarray(t)[6:].reshape(-1,3)
 def evaluate(self,t,folder,parent,spectrum=True):
  self.cost+=.1;t=np.asarray(t);u=t[6:].reshape(-1,3);force=-u
  return dict(energy_Ha=float(.5*(u*u).sum()),gap_eV=float(1+100*t[0]),residual_Ha=1e-12,forces_Ha_Bohr=force.tolist(),ASR_Ha_Bohr=0.,charge=96.,electrons=96.,nearest_distance_bohr=4.,cell_determinant_bohr3=512.,ritz_max_Ha=1e-12,branch_margin_Ha=.1,electronic_reoptimized=True,state_directory=str(folder))
 def gradient(self,t,r,folder):self.cost+=.05;g=np.zeros_like(t);g[0]=100.;return g
 def spent(self):return self.cost
 def first_step_report(self,m):self.report=m

class Checks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  original=Path(__file__).resolve().parents[4]/'reproducibility/response_design/records/response_design_20261020_e9_relaxed_design/registered_protocol.json'
  self.p=read(original)['E9'];self.t=np.zeros(12);self.t[6]=.02;self.t[9]=-.02
 def tearDown(self):self.tmp.cleanup()
 def test_read_only_adjoint_removes_ConfigDict_checkpoints(self):
  # Compile the actual method, with a CPU-only base; no JAX import/device init.
  from ml_collections import ConfigDict
  tree=ast.parse(Path(__file__).with_name('e9_backend.py').read_text())
  backend=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='E9Backend')
  method=next(n for n in backend.body if isinstance(n,ast.FunctionDef) and n.name=='gradient')
  module=types.ModuleType('joint_design_dft');calls=[]
  def original(config,**kwargs):
   self.assertNotIn('checkpoint',config.d3ft)
   self.assertNotIn('checkpoint_resume',config.d3ft)
   self.assertEqual(config.d3ft.optim.lr_schedule,'constant')
   self.assertEqual(kwargs,{'sentinel':17})
   calls.append(config.d3ft.is_locked)
   return 'cpu-read-only-ok'
  module.calc=original
  class CPUBase:
   def gradient(inner,theta,current,folder):return module.calc(inner.config,sentinel=17)
  cls=ast.ClassDef(name='E9Backend',bases=[ast.Name(id='CPUBase',ctx=ast.Load())],
                  keywords=[],body=[method],decorator_list=[])
  code=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));ns={'CPUBase':CPUBase}
  exec(compile(code,str(Path(__file__).with_name('e9_backend.py')),'exec'),ns)
  for locked in [False,True]:
   for present in [False,True]:
    with self.subTest(locked=locked,checkpoint_fields_present=present):
     b=ns['E9Backend']();b.config=ConfigDict({'d3ft':{'optim':{'lr_schedule':'constant'}}})
     if present:
      b.config.d3ft.checkpoint={'root':'must-not-write'}
      b.config.d3ft.checkpoint_resume={'completed_steps':500}
     if locked:b.config.lock()
     with patch.dict(sys.modules,{'joint_design_dft':module}):
      self.assertEqual(b.gradient(None,None,None),'cpu-read-only-ok')
     self.assertIs(module.calc,original)
     self.assertEqual(calls[-1],locked)
 def test_no_E9_probe_or_submission_before_AlP_pair_finishes(self):
  root=self.root/'queue';root.mkdir();B=self.root/'B'
  dump(root/'registered_protocol.json',dict(project=str(self.root)))
  dump(root/'resume_authorization.json',dict(B_root=str(B)))
  dump(root/'backend_protocol.json',{})
  dump(root/'GPU_budget.json',dict(attempts=[]))
  dump(B/'budget.json',dict(logical_results={'AlP_fitted':dict(passed=True)},attempts=[]))
  with patch('e9_queue.subprocess.run') as probe,patch('e9_queue.subprocess.Popen') as submit,patch('e9_queue.time.sleep',side_effect=InterruptedError):
   with self.assertRaises(InterruptedError):e9_queue.main(root)
   probe.assert_not_called();submit.assert_not_called()
  self.assertEqual(read(root/'status.json')['stage'],'waiting_AlP_pair_complete_and_accepted_no_E9_GPU_submitted')
 def test_preregistered_limits_unchanged(self):
  self.assertEqual(self.p['relaxation_force_max_norm_Ha_Bohr'],.001)
  self.assertEqual(self.p['max_ionic_accepted_steps_per_relaxation'],40)
  self.assertEqual(self.p['max_energy_force_evaluations_per_relaxation'],80)
  self.assertEqual(self.p['target_tolerance_eV'],.001)
  self.assertEqual(self.p['max_outer_accepted_updates'],5)
 def test_relaxation_force_bound_and_fixed_cell(self):
  b=FakeBackend();t,r=relax(self.root/'relax',self.t,None,b,self.p,self.t)
  self.assertLess(max_force(r),.001);np.testing.assert_array_equal(t[:6],self.t[:6]);np.testing.assert_allclose(t[6:].reshape(-1,3).sum(axis=0),0,atol=1e-12)
  self.assertGreater(r['ionic_accepted_steps'],0)
 def test_full_target_then_independent_audit(self):
  b=FakeBackend();r=run(self.root/'design',self.t,b,self.p)
  self.assertTrue(r['passed']);self.assertLessEqual(abs(r['final_gap_eV']-r['target_eV']),.001)
  self.assertEqual(b.report['initial_relaxation_ionic_steps'],1);self.assertGreater(b.report['step_GPU_hours'],0)
 def test_ionic_limit_fails_without_relaxing_gate(self):
  p=copy.deepcopy(self.p);p['max_ionic_accepted_steps_per_relaxation']=0
  with self.assertRaises(RuntimeError):relax(self.root/'limit',self.t,None,FakeBackend(),p,self.t)
 def test_force_uses_atom_vector_norm(self):
  r=dict(forces_Ha_Bohr=[[.0007,.0007,.0007]])
  self.assertGreater(max_force(r),.001)
 def test_frozen_target_cannot_change_on_restart(self):
  root=self.root/'restart';run(root,self.t,FakeBackend(),self.p)
  q=read(root/'target.json');q['target_eV']+=.02;dump(root/'target.json',q)
  with self.assertRaises(ValueError):run(root,self.t,FakeBackend(),self.p)
 def test_resume_reuses_accepted_ionic_state(self):
  folder=self.root/'resume';t,r=relax(folder,self.t,None,FakeBackend(),self.p,self.t)
  old=read(folder/'ionic_state.json');t2,r2=relax(folder,self.t,None,FakeBackend(),self.p,self.t)
  np.testing.assert_allclose(t,t2);self.assertEqual(read(folder/'ionic_state.json'),old)

if __name__=='__main__':unittest.main(verbosity=2)
