"""CPU checks of the only permitted old-k4 -> current-basis reuse."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'reverse_response'))
from mesh_basis_restart import project_miller_coefficients

class ProjectionGuards(unittest.TestCase):
    def test_maps_by_G_identity_not_compact_index(self):
        old=np.array([1,0,1,1,0],bool);new=np.array([0,1,1,1,1],bool)
        pw=np.array([[[[10+1j],[20+2j],[30+3j]]]])
        mapped,info=project_miller_coefficients(pw,old,new)
        np.testing.assert_array_equal(mapped.reshape(-1),[0,20+2j,30+3j,0])
        self.assertEqual(info['common_G'],2);self.assertFalse(info['density_reused'])
        self.assertTrue(info['orthonormalization_required'])

    def test_same_basis_is_exact(self):
        mask=np.array([1,0,1,1],bool);pw=np.arange(24).reshape(1,2,3,4).astype(complex)
        result,info=project_miller_coefficients(pw,mask,mask)
        np.testing.assert_array_equal(result,pw)

    def test_rejects_different_FFT_grid_or_wrong_G_count(self):
        with self.assertRaises(ValueError):project_miller_coefficients(np.zeros((1,2,3,4)),np.ones(3,bool),np.ones(4,bool))
        with self.assertRaises(ValueError):project_miller_coefficients(np.zeros((1,2,2,4)),np.ones(3,bool),np.ones(3,bool))

if __name__=='__main__':unittest.main()
