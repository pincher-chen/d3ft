import unittest
from bounded_trial import decide


class BoundedTrialTests(unittest.TestCase):
    def test_exact_loss_and_ratio(self):
        d=decide([2.],[1.],[[1.]],[-1.],[0.],[0.],[[0.]])
        self.assertEqual(d['rho_interval'],[1.,1.])
        self.assertEqual(d['conditional_action'],'accept')
        self.assertFalse(d['production_accepted'])

    def test_error_information_changes_same_nominal_step(self):
        loose=decide([2.],[1.],[[1.]],[-1.],[.8],[.8],[[.2]])
        tight=decide([2.],[1.],[[1.]],[-1.],[.01],[.01],[[.01]])
        self.assertEqual(loose['conditional_action'],'refine')
        self.assertEqual(tight['conditional_action'],'accept')

    def test_increase_rejected_and_negative_bounds_invalid(self):
        d=decide([2.],[3.],[[1.]],[-1.],[0.],[0.],[[0.]])
        self.assertEqual(d['conditional_action'],'reject')
        with self.assertRaises(ValueError):
            decide([2.],[1.],[[1.]],[-1.],[0.],[0.],[[-.1]])

    def test_intervals_enclose_perturbed_model_and_actual(self):
        # The baseline perturbation is shared by the model and actual loss.
        import itertools
        d=decide([2.],[1.],[[1.]],[-1.],[.1],[.1],[[.1]])
        for e0,e1,ej in itertools.product((-.1,0.,.1),repeat=3):
            r=2+e0;c=1+e1;j=1+ej
            a=r*r-c*c;p=r*r-(r-j)**2
            for value,key in [(a,'reduction_interval'),(p,'predicted_reduction_interval')]:
                lo,hi=d[key];self.assertLessEqual(lo-1e-12,value);self.assertLessEqual(value,hi+1e-12)
            lo,hi=d['rho_interval']
            self.assertLessEqual(lo-1e-12,a/p);self.assertLessEqual(a/p,hi+1e-12)


if __name__=='__main__':unittest.main()
