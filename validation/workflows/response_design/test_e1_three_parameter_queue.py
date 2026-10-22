"""CPU guards for the reduced FD scope and concurrent resource budget."""
import datetime as dt
import unittest

from e1_three_parameter_queue import AXES, OMITTED_AXES, allowed_seconds, matching_vector, observed_completion_code


class ReducedFDGuards(unittest.TestCase):
    def setUp(self):
        self.now = dt.datetime(2026, 10, 16, 1, 4, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        self.deadline = self.now + dt.timedelta(hours=24)
        self.steps = [.01, .01, .05039265872238211, .01, .01, .014316199851767995, .01]

    def test_six_legacy_theta_s_subpoints_are_not_other_parameters(self):
        saved = {'parameter_vector': [-.01, 0, 0, 0, 0, 0, 0]}
        for axis in AXES:
            for sign in (-1, 1):
                self.assertFalse(matching_vector(saved, axis, sign, self.steps))
        self.assertTrue(matching_vector(saved, 0, -1, self.steps))

    def test_wrong_sign_or_extra_component_cannot_be_reused(self):
        v = [0.] * 7
        v[5] = self.steps[5]
        self.assertTrue(matching_vector({'parameter_vector': v}, 5, 1, self.steps))
        self.assertFalse(matching_vector({'parameter_vector': v}, 5, -1, self.steps))
        v[0] = 1e-15
        self.assertFalse(matching_vector({'parameter_vector': v}, 5, 1, self.steps))

    def test_deadline_limits_child_without_signalling_gpu(self):
        self.assertEqual(allowed_seconds(self.deadline, 68.93, 0, 250, self.now), 43200)
        near = self.deadline - dt.timedelta(seconds=3600)
        self.assertEqual(allowed_seconds(self.deadline, 68.93, 0, 250, near), 3540)
        self.assertEqual(allowed_seconds(self.deadline, 68.93, 0, 250, self.deadline), 0)

    def test_other_active_children_are_reserved_in_budget(self):
        # Two outstanding eight-hour, two-card children reserve 32 GPU h.
        seconds = allowed_seconds(self.deadline, 210, 32, 250, self.now)
        self.assertEqual(seconds, 14340)
        self.assertLessEqual(210 + 32 + 2*seconds/3600, 250)
        self.assertEqual(allowed_seconds(self.deadline, 220, 32, 250, self.now), 0)

    def test_scopes_partition_seven_columns(self):
        self.assertEqual(AXES, (5, 6, 1))
        self.assertEqual(set(AXES) | set(OMITTED_AXES), set(range(7)))
        self.assertFalse(set(AXES) & set(OMITTED_AXES))

    def test_observer_does_not_invent_exit_code_or_timeout(self):
        self.assertEqual(observed_completion_code('point', True, 'DONE point\n'), 0)
        self.assertEqual(observed_completion_code('point', False, 'DONE point\n'), 1)
        self.assertEqual(observed_completion_code('point', True, 'still running'), 1)
        self.assertEqual(observed_completion_code('point', False, 'srun: task 0: Exited with exit code 124'), 124)

    def test_deadline_still_truncates_extended_12h_ceiling(self):
        now = self.deadline-dt.timedelta(hours=2)
        self.assertEqual(allowed_seconds(self.deadline, 100, 0, 250, now), 7140)


if __name__ == '__main__':
    unittest.main()
