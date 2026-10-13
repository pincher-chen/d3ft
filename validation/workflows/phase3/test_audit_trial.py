"""Reject stale-density and incomplete-mesh spectrum certificates."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_trial import audit_spectrum, sha


class SpectrumAuditTests(unittest.TestCase):
    def test_provenance_and_missing_k_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; spectrum = root/'spectrum'
            source.mkdir(); spectrum.mkdir()
            def put(path, value):
                path.write_text(json.dumps(value))
            (source/'density.npy').write_bytes(b'fixture density')
            put(source/'metadata.json', {'config': {'k_grid_sizes': 1}})
            put(source/'points.json', [{'label': 'eq_0', 'density_file': 'density.npy',
                                       'density_sha256': sha(source/'density.npy')}])
            put(spectrum/'metadata.json', {
                'source': str(source), 'source_metadata_sha256': sha(source/'metadata.json'),
                'density_sha256': sha(source/'density.npy'), 'mesh': [[0.,0.,0.]],
                'indices': [0], 'solver': 'block', 'occupied_bands': 1,
                'electrons': 2., 'density_electrons': 2., 'charge_budget': 2e-5})
            put(spectrum/'summary.json', {'solver':'block', 'full_mesh':True,
                'mesh_fixed_occupations_passed':True, 'gap_eV':27.211386245988,
                'max_eigen_residual_Ha':1e-9})
            put(spectrum/'k_0000.json', {'index':0,'k_frac':[0.,0.,0.],
                'eigenvalues_Ha':[0.,1.], 'residuals_Ha':[1e-9,1e-9],
                'vbm_Ha':0., 'cbm_Ha':1.})
            self.assertGreater(audit_spectrum(source,spectrum)['gap_eV'], 0)
            (source/'density.npy').write_bytes(b'different density')
            with self.assertRaisesRegex(ValueError, 'density mismatch'):
                audit_spectrum(source,spectrum)
            (source/'density.npy').write_bytes(b'fixture density')
            (spectrum/'k_0000.json').unlink()
            with self.assertRaises(FileNotFoundError):
                audit_spectrum(source,spectrum)


if __name__ == '__main__':
    unittest.main()
