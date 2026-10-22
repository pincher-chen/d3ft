"""Independent geometry-map/units/phase checks before physical response."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from geometry_layout import GeometryLayout,strain_matrix
from jrystal._src.const import ANGSTROM2BOHR
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
cell=jnp.array([[4.,.2,.1],[.1,5.,.3],[.2,.1,6.]])
base_positions=jnp.array([[.2,.4,.7],[1.2,2.1,3.4]])
theta=jnp.array([.01,-.006,.004,.003,-.005,.002]);u=jnp.array([[.03,-.02,.01],[-.015,.022,-.017]])
F=jnp.eye(3)+strain_matrix(theta);current_cell=cell@F.T;current_positions=base_positions@F.T+u
layout=GeometryLayout(2,jnp.r_[theta,u.reshape(-1)],include_coordinates=True)
delta=jnp.arange(12)*.0001-.0003
new_cell,new_positions=layout.deform(current_cell,current_positions,delta)
expected_cell=cell@(jnp.eye(3)+strain_matrix(theta+delta[:6])).T
expected_positions=base_positions@(jnp.eye(3)+strain_matrix(theta+delta[:6])).T+u+delta[6:].reshape(2,3)
err_cell=float(jnp.max(abs(new_cell-expected_cell)));err_positions=float(jnp.max(abs(new_positions-expected_positions)))
assert max(err_cell,err_positions)<1e-12
legacy=GeometryLayout(2,theta);d6=delta[:6]
old_cell=current_cell@(jnp.eye(3)+strain_matrix(d6)@jnp.linalg.inv(F)).T
old_positions=current_positions@(jnp.eye(3)+strain_matrix(d6)@jnp.linalg.inv(F)).T
c,p=legacy.deform(current_cell,current_positions,d6)
assert float(jnp.max(abs(c-old_cell)))<1e-14 and float(jnp.max(abs(p-old_positions)))<1e-14
jac=jax.jacfwd(lambda d:layout.deform(current_cell,current_positions,d)[1])(jnp.zeros(12))
assert np.max(abs(np.asarray(jac)[:,:,6:].reshape(6,6)-np.eye(6)))<1e-14
celljac=jax.jacfwd(lambda d:layout.deform(current_cell,current_positions,d)[0])(jnp.zeros(12))
assert float(jnp.max(abs(celljac[:,:,6:])))==0.
a_layout=GeometryLayout(2,jnp.r_[theta,u.reshape(-1)/ANGSTROM2BOHR],include_coordinates=True,displacement_unit='angstrom')
deltaA=delta.at[6:].set(delta[6:]/ANGSTROM2BOHR)
cA,pA=a_layout.deform(current_cell,current_positions,deltaA)
assert float(jnp.max(abs(pA-new_positions)))<1e-12
vectors=jnp.array([[.3,.7,-.2],[1.1,-.4,.5]]);direction=jnp.array([.2,-.3,.4]);point=current_positions[0]
phase=lambda r:jnp.exp(-1j*(vectors@r))
primal,tangent=jax.jvp(phase,(point,),(direction,))
expected=-1j*(vectors@direction)*primal;phase_error=float(jnp.max(abs(tangent-expected)))
assert phase_error<1e-14
result=dict(passed=True,geometry_cell_error=err_cell,geometry_position_error=err_positions,
 legacy_six_input_equivalence=True,coordinates_added_after_affine_strain=True,
 Cartesian_coordinate_jacobian_identity=True,atomic_displacements_do_not_change_cell=True,
 angstrom_bohr_mapping=True,phase_direction_error=phase_error,
 scope='Interface checks only; electronic response, FD and translation gap zero pending')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
