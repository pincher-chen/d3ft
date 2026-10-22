"""QE-compatible local radial quadrature for the opt-in PP_RAB path.

The auxiliary 10-Bohr odd radial mesh follows QE read_pseudo. Fourier tables
use dq=.01, Gaussian Coulomb subtraction, and four-point interpolation; G=0
retains the independently integrated bare alpha-Z. No fitted constants.
"""
import numpy as np
import jax.numpy as jnp
from scipy.special import spherical_jn,erf
from .geometry import safe_norm
from .quadrature import restored_mesh,qe_simpson_weights,qe_radial_size

def qe_local_reciprocal(positions,g,r_grid,v_grid,charges,volume,*,r_ab,kmax=None):
 radius=safe_norm(g,axis=-1)
 if kmax is None:kmax=float(np.max(np.asarray(radius)))
 if len(r_grid)!=len(r_ab):raise ValueError('Local PP_RAB metadata length mismatch')
 qs=np.arange(0.,kmax+.040001,.01);forms=[]
 for r,v,z,rab in zip(r_grid,v_grid,charges,r_ab):
  r,v,rab=restored_mesh(r,np.asarray(v)[None,...],rab);m=qe_radial_size(r);r,v,rab=r[:m],v[0,:m],rab[:m];weights=qe_simpson_weights(rab)
  alpha=4*np.pi*np.sum(r*(r*v+z)*weights)
  integrand=r*(r*v+z*erf(r))*weights;table=np.empty(len(qs))
  for start in range(0,len(qs),256):table[start:start+256]=4*np.pi*(spherical_jn(0,qs[start:start+256,None]*r)@integrand)
  table=jnp.asarray(table);t=radius/.01;i=jnp.floor(t).astype(jnp.int32);p=t-i;u=1-p;vv=2-p;w=3-p
  f=table[i]*u*vv*w/6+table[i+1]*p*vv*w/2-table[i+2]*p*u*w/2+table[i+3]*p*u*vv/6
  safe=radius.at[0,0,0].set(1.)
  f=f-4*jnp.pi*z*jnp.exp(-safe**2/4)/safe**2
  forms.append(f.at[0,0,0].set(alpha))
 structure=jnp.exp(-1j*g@positions.T)
 return jnp.sum(jnp.stack(forms)*jnp.moveaxis(structure,-1,0),axis=0)*np.prod(g.shape[:-1])/volume
