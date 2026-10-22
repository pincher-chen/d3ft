"""Real owned Si k2 regression of Gamma occupation broadcast and charge gate."""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from ml_collections import ConfigDict
from gamma_response import validated_radial_splines
from bounded_xc import memory_bounded_xc
from kmesh_restart import gamma_orbital_warm_start
P = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(P/'validation/workflows/response_design'))
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
a.output.mkdir(exist_ok=False)
if jax.default_backend() != 'cpu' or len(jax.devices()) != 2:
    raise ValueError('Two virtual CPU devices required')
m = json.loads((a.source/'metadata.json').read_text())
with (a.source/'state.pkl').open('rb') as f: saved = pickle.load(f)
# Reproduce the Gamma -> eight-k transition using owned Gamma-block orbitals.
gamma = dict(pw=saved['pw'][:, :1], occ=saved['occ'][:, :1]*8)
warm = gamma_orbital_warm_start(gamma, 8)
assert set(warm) == {'pw'}
bad = dict(pw=warm['pw'], occ=np.repeat(gamma['occ'], 8, axis=1))
good = dict(pw=warm['pw'], occ=np.repeat(gamma['occ']/8, 8, axis=1))
results = {}
for tag, initial, pool in [('bad_serial', bad, False), ('fixed_serial', warm, False),
                            ('explicit_serial', good, False), ('bad_pool', bad, True),
                            ('fixed_pool', warm, True)]:
    c = ConfigDict(m['config']); c.epoch = 1
    c.d3ft.optim.lr_schedule = 'constant'; c.optimizer_args.learning_rate = 0.
    c.d3ft.parallel.k_pool = pool
    c.d3ft.charge_gate = dict(enabled=True, absolute_tolerance=1e-6,
                            report_path=str(a.output/(tag+'_charge.jsonl')))
    c.crystal_file_path_path = str(a.source/'input.xyz')
    try:
        with plain_fft_patch(k_pool=pool), local_simpson_patch(), memory_bounded_xc(), validated_radial_splines():
            out = calc(c, init_params=initial, freq_mask=np.load(a.source/'mask.npy'))
        if tag.startswith('bad'): raise AssertionError('Bad warm start escaped gate')
        rho = np.asarray(out.density_grid)
        results[tag] = dict(energy_Ha=float(out.total_energy), charge=float(rho.sum()*float(out.crystal.vol)/np.prod(rho.shape[-3:])))
        np.save(a.output/(tag+'_density.npy'), rho)
    except ValueError as exc:
        if not tag.startswith('bad') or 'Physical charge gate failed at step 0' not in str(exc): raise
        results[tag] = dict(rejected_before_update=True, error=str(exc))
    (a.output/'partial.json').write_text(json.dumps(results, indent=2)+'\n')
errors = {}
for tag in ['explicit_serial', 'fixed_pool']:
    errors[tag+'_energy_Ha'] = abs(results[tag]['energy_Ha']-results['fixed_serial']['energy_Ha'])
    errors[tag+'_density'] = float(np.max(abs(np.load(a.output/(tag+'_density.npy'))-np.load(a.output/'fixed_serial_density.npy'))))
passed = all(e <= 1e-11 for e in errors.values()) and all(abs(r['charge']-8) <= 1e-6 for r in results.values() if 'charge' in r)
(a.output/'acceptance.json').write_text(json.dumps(dict(passed=passed, results=results, errors=errors,
    scope='Real owned Si k2 functional, initial and step-one physical density; serial vs two CPU shards; malformed Gamma occupations rejected before update'), indent=2)+'\n')
if not passed: raise ValueError('Restart equivalence failed')
print('E5 warm-start regression PASS: bad 64-electron Si rejected; corrected density eight electrons; serial/pool energy and density match.', flush=True)
