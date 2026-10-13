"""Nonzero parameter centers and the actual relaxation guard call chain."""
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from d3ft.calc import screening
from d3ft.calc.forces import relax_internal_coordinates

out=SimpleNamespace(crystal=SimpleNamespace(positions=np.zeros((2,3)), cell_vectors=np.eye(3)*10), config=SimpleNamespace(ewald_args={"ewald_cutoff": 1.0}))
kw=dict(params_pw=None, params_occ=None, theta_nonlocal=0.7,
        theta_nonlocal_l=[0.2,-0.1,0.3], xc_params=None)
def toy_energy(strain, pw, occ, theta_nonlocal_l, xc_params, **kwargs):
    t=theta_nonlocal_l
    mu=xc_params['gga_x_pbe']['mu']
    return jnp.sum(t*t)+mu*t[0]+mu*mu
with patch.object(screening, '_kwargs_from', return_value=kw), \
     patch.object(screening, 'free_energy_at_strain', side_effect=toy_energy):
    got=screening.sensitivity(out, {'gga_x_pbe':{'mu':0.4}}, 3)
    np.testing.assert_allclose(got,[0.8,-0.2,0.6,1.0],atol=1e-14)

crystal=SimpleNamespace(cell_vectors=np.eye(3)*10, positions=np.array([[0.,0.,0.],[2.,2.,2.]]))
cfg=SimpleNamespace(crystal=None, crystal_file_path_path='unused')
fake=SimpleNamespace(freq_mask=np.ones((2,2,2),bool), params_pw=None, params_occ=None)
with tempfile.TemporaryDirectory() as tmp, \
     patch('jrystal.calc.opt_utils.create_crystal', return_value=crystal), \
     patch('d3ft.calc.ground_state.calc', return_value=fake), \
     patch('d3ft.calc.forces.forces_from_ground_state', return_value=np.array([[1e-3,0,0],[-5e-4,0,0]])):
    try:
        relax_internal_coordinates(cfg,['Ga','As'],workdir=tmp,verbose=False)
    except RuntimeError as e:
        assert 'acoustic sum rule' in str(e)
    else:
        raise AssertionError('mean subtraction hid invalid raw forces')
with tempfile.TemporaryDirectory() as tmp, \
     patch('jrystal.calc.opt_utils.create_crystal', return_value=crystal), \
     patch('d3ft.calc.ground_state.calc', return_value=fake), \
     patch('d3ft.calc.forces.forces_from_ground_state', return_value=np.array([[1e-3,0,0],[-1e-3,0,0]])):
    result=relax_internal_coordinates(cfg,['Si','Si'],max_steps=1,workdir=tmp,verbose=False)
    assert not result['converged']
    np.testing.assert_allclose(result['positions'],crystal.positions)
print('test_27: current parameter center and unprojected-force guards pass')
