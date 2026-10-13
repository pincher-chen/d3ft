import itertools
import unittest
from loss_intervals import loss_interval, reduction_interval


class LossIntervalTests(unittest.TestCase):
    def test_encloses_samples_and_zero_crossing(self):
        r,e=[.2,-2.],[.5,.1]
        lo,hi=loss_interval(r,e)
        self.assertAlmostEqual(lo,1.9**2)
        for a,b in itertools.product([-.5,-.2,0.,.5],[-.1,0.,.1]):
            v=(r[0]+a)**2+(r[1]+b)**2
            self.assertGreaterEqual(v+1e-12,lo)
            self.assertLessEqual(v,hi+1e-12)

    def test_positive_nominal_improvement_can_be_unresolved(self):
        self.assertEqual(reduction_interval([1.],[.99],[.1],[.1])['conditional_decision'],'unresolved')
        self.assertEqual(reduction_interval([2.],[1.],[.01],[.01])['conditional_decision'],'resolved_decrease')
        self.assertEqual(reduction_interval([1.],[2.],[.01],[.01])['conditional_decision'],'resolved_increase')

    def test_invalid_and_exact(self):
        self.assertEqual(loss_interval([2.,-3.],[0.,0.]),(13.,13.))
        for r,e in [([],[]),([1.],[]),([1.],[-1.]),([float('nan')],[0.])]:
            with self.assertRaises(ValueError): loss_interval(r,e)


if __name__=='__main__': unittest.main()
