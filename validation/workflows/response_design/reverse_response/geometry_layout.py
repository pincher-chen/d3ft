"""Geometry input ordering and units for Gamma strain/Cartesian response.

Coordinates are absolute Cartesian increments added AFTER affine deformation.
The current geometry may already contain u0: that part must not be strained
again when differentiating strain at fixed Cartesian displacement.
"""
import numpy as np
import jax.numpy as jnp
from jrystal._src.const import ANGSTROM2BOHR

def strain_matrix(v):
 return jnp.array([[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]])

class GeometryLayout:
 def __init__(self,natoms,theta0,*,include_coordinates=False,displacement_unit='bohr',active_axes=None):
  if natoms<1:raise ValueError('Positive atom count required')
  if displacement_unit not in ['bohr','angstrom']:raise ValueError('Cartesian displacement unit must be bohr or angstrom')
  self.natoms=int(natoms);self.include_coordinates=bool(include_coordinates)
  self.displacement_unit=displacement_unit
  self.coordinate_to_bohr=1. if displacement_unit=='bohr' else float(ANGSTROM2BOHR)
  self.ncanonical=6+3*self.natoms if include_coordinates else 6
  axes=np.arange(self.ncanonical) if active_axes is None else np.asarray(active_axes)
  if axes.ndim!=1 or not np.issubdtype(axes.dtype,np.integer) or len(axes)<1 or len(set(axes.tolist()))!=len(axes) or np.min(axes)<0 or np.max(axes)>=self.ncanonical:raise ValueError('Invalid unique active geometry axes')
  self.active_axes=axes.astype(int);self.nvariables=len(axes)
  theta0=np.asarray(theta0,dtype=float)
  if theta0.shape==(6,) and include_coordinates:theta0=np.r_[theta0,np.zeros(3*self.natoms)]
  if theta0.shape!=(self.ncanonical,) or not np.isfinite(theta0).all():raise ValueError('Reference geometry parameter shape/values invalid')
  self.theta0=jnp.asarray(theta0);self.inverse0=jnp.linalg.inv(jnp.eye(3)+strain_matrix(self.theta0[:6]))
  self.current_displacement=(self.theta0[6:].reshape(self.natoms,3)*self.coordinate_to_bohr if include_coordinates else jnp.zeros((self.natoms,3)))
  names=['eps_xx','eps_yy','eps_zz','gamma_yz','gamma_xz','gamma_xy']+([f'u{i}_{axis}' for i in range(self.natoms) for axis in 'xyz'] if include_coordinates else [])
  units=['eV / unit engineering strain']*6+([f'eV / {displacement_unit}']*(3*self.natoms) if include_coordinates else [])
  self.names=[names[i] for i in self.active_axes];self.gradient_units=[units[i] for i in self.active_axes]
 def deform(self,cell,positions,delta):
  if delta.shape!=(self.nvariables,):raise ValueError('Geometry direction size mismatch')
  delta=jnp.zeros(self.ncanonical,dtype=delta.dtype).at[self.active_axes].set(delta)
  relative=strain_matrix(delta[:6])@self.inverse0;transform=jnp.eye(3)+relative
  deformed_cell=cell@transform.T
  deformed_positions=(positions-self.current_displacement)@transform.T+self.current_displacement
  if self.include_coordinates:deformed_positions+=delta[6:].reshape(self.natoms,3)*self.coordinate_to_bohr
  return deformed_cell,deformed_positions
