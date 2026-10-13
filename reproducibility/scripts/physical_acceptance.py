"""Unchanged thresholds with the electron count read from each physical input."""
import json

def check(folder, audit):
    paths = (folder/'summary.json', folder/'endpoint.json', audit/'summary.json')
    if not all(p.exists() for p in paths):
        return dict(complete=False, passed=False, tests={})
    s, e, a = (json.loads(p.read_text()) for p in paths)
    ne = s['expected_electrons']
    tests = dict(charge=abs(s['charge']-ne)<=1e-10,
        orthogonality=e['orbital_orthonormality_max']<=1e-10,
        external=e['orbital_outside_max_Ha']<=1e-6,
        commutator=e['orbital_commutator_max_Ha']<=1e-6,
        occupation=e['fermi_dirac_occupation_max_error']<=1e-6,
        spectrum=a['maximum_ritz_residual_Ha']<=1e-8,
        density=a['spectral_update_density_L1_electrons']/ne<=1e-6,
        free_energy=abs(a['spectral_update_free_energy_change_Ha'])<=1e-8)
    return dict(complete=True, passed=all(tests.values()), tests=tests)
