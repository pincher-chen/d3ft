import unittest,numpy as np
from e7_three_material_fit import dataset,proposal,accept,AXES,BOUNDS
class FitTests(unittest.TestCase):
 def test_external_holdout_blocked(self):
  with self.assertRaises(ValueError):dataset(['SiC'],{}, {})
 def test_fold_does_not_touch_validation_rows(self):
  class Forbidden(dict):
   def __getitem__(self,k):raise AssertionError('Validation accessed')
  g=[0.,1,0,0,0,2,3];states={'Si':dict(summary={'a0_A':5.},jacobian={'a0_A':g}),'BN':dict(summary={'B_GPa':400.},jacobian={'B_GPa':g}),'diamond':Forbidden()};refs={'Si':{'a0_A':5.1},'BN':{'B_GPa':381.1},'diamond':Forbidden()}
  r,J,rows=dataset(['Si','BN'],states,refs);self.assertEqual(J.shape,(2,3));self.assertEqual([r[0] for r in rows],['Si','BN'])
 def test_rank_deficient_regularized_step_and_selected_axes(self):
  r=np.array([2.,-1.]);J=np.array([[1.,2.,3.],[0.,1.,2.]])
  v,record=proposal(np.zeros(7),r,J,1.,.2);self.assertTrue(np.isfinite(v).all());self.assertEqual(record['rank'],2);self.assertFalse(record['unique_three_parameter_identifiability']);self.assertTrue(np.all(abs(v[AXES])<=BOUNDS));self.assertTrue(np.array_equal(v[[0,2,3,4]],np.zeros(4)))
 def test_actual_worse_loss_rejected(self):
  self.assertFalse(accept(10.,11.,1.)['accepted']);self.assertTrue(accept(10.,3.,2.)['accepted'])
if __name__=='__main__':unittest.main()
