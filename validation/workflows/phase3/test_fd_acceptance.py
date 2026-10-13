import unittest
from fd_acceptance import component, aggregate


class FDAcceptanceTests(unittest.TestCase):
    def test_accurate_but_unresolved_is_not_accepted(self):
        rows = [component(k, .0001, .0001, 5., .02) for k in
                ('a0_A', 'B_GPa', 'C11_minus_C12_GPa', 'C44_GPa')]
        self.assertTrue(aggregate(rows)['agreement_passed'])
        self.assertFalse(aggregate(rows)['joint_gate_passed'])

    def test_joint_gate_needs_four_distinct_resolved_observables(self):
        rows = [component(k, 1., 1., 5., .02) for k in
                ('a0_A', 'B_GPa', 'C11_minus_C12_GPa', 'C44_GPa')]
        self.assertTrue(aggregate(rows)['joint_gate_passed'])
        self.assertFalse(aggregate(rows[:3])['joint_gate_passed'])
        self.assertFalse(aggregate([rows[0]] * 4)['joint_gate_passed'])
        self.assertFalse(aggregate([])['agreement_passed'])

    def test_relative_lattice_budget_and_invalid_data(self):
        row = component('a0_A', .0006, .0006, 5., .02)
        self.assertTrue(row['signal_resolved'])
        self.assertAlmostEqual(row['signal_to_budget'], 1.2)
        self.assertFalse(component('B_GPa', .1, .1, 5., .02)['signal_resolved'])
        with self.assertRaises(ValueError):
            component('B_GPa', float('nan'), 1., 5., .02)


if __name__ == '__main__':
    unittest.main()
