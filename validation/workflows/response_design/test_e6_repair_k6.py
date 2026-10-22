"""Do not reuse a failed spectrum or cross-wire a hydro-state/k index."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from e6_repair_k6 import validate_local, sha


class SpectrumReuseGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        np.save(self.path/'orbitals.npy', np.eye(52))
        self.row = dict(k_index=6, source_state_sha256='state-A',
            saved_orbitals_sha256=sha(self.path/'orbitals.npy'),
            eigenvalues_Ha=list(np.linspace(-1, 1, 52)),
            residuals_Ha=[1e-10]*52, orthogonality_max=1e-14)

    def write(self):
        (self.path/'summary.json').write_text(json.dumps(self.row))

    def test_passed_spectrum_accepted(self):
        self.write()
        self.assertEqual(validate_local(self.path, 'state-A', 6)['k_index'], 6)

    def test_old_failed_empty_band_not_reused_as_pass(self):
        self.row['residuals_Ha'][-1] = 1.53e-8
        self.write()
        with self.assertRaises(ValueError): validate_local(self.path, 'state-A', 6)

    def test_wrong_source_or_k_rejected(self):
        self.write()
        with self.assertRaises(ValueError): validate_local(self.path, 'state-B', 6)
        with self.assertRaises(ValueError): validate_local(self.path, 'state-A', 5)

    def test_modified_ritz_vectors_rejected(self):
        self.write()
        np.save(self.path/'orbitals.npy', np.ones((52,52)))
        with self.assertRaises(ValueError): validate_local(self.path, 'state-A', 6)

    def test_nonfinite_and_nonorthogonal_rejected(self):
        for residual, orth in [(float('nan'), 1e-14), (1e-10, 2e-10)]:
            self.row['residuals_Ha'][-1] = residual
            self.row['orthogonality_max'] = orth
            self.write()
            with self.assertRaises(ValueError): validate_local(self.path, 'state-A', 6)


if __name__ == '__main__': unittest.main()
