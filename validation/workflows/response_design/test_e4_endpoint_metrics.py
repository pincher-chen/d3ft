import unittest,numpy as np
from e4_endpoint_metrics import metrics,BOHR_A,FORCE_EV_A
class MetricTests(unittest.TestCase):
 def test_atomic_vector_force_distinct_from_component(self):
  r=dict(forces_Ha_Bohr=[[.03,.04,0],[0,0,0]],energy_Ha=-2.,gap_eV=1.,residual_Ha=0,ASR_Ha_Bohr=0)
  cell=np.eye(3)*10;pos=np.array([[0,0,0],[2,0,0]]);theta=np.zeros(12)
  m=metrics(r,theta,theta,cell,pos,cell,pos,r)
  self.assertAlmostEqual(m['max_single_atom_force_Ha_Bohr'],.05);self.assertAlmostEqual(m['max_force_component_Ha_Bohr'],.04)
  self.assertAlmostEqual(m['max_single_atom_force_eV_A'],.05*FORCE_EV_A)
 def test_affine_displacement_not_internal_coordinate_motion(self):
  r=dict(forces_Ha_Bohr=np.zeros((2,3)),energy_Ha=-2.,gap_eV=1.,residual_Ha=0,ASR_Ha_Bohr=0)
  cell=np.eye(3)*10;pos=np.array([[0,0,0],[2,0,0]]);theta=np.zeros(12);theta[0]=.01
  f=np.diag([1.01,1,1]);m=metrics(r,theta,np.zeros(12),cell@f,pos@f,cell,pos,r)
  self.assertEqual(m['max_internal_displacement_A'],0);self.assertAlmostEqual(m['max_total_cartesian_displacement_A'],.02*BOHR_A)
if __name__=='__main__':unittest.main()
