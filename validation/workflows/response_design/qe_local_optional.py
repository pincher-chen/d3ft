"""Opt-in same-UPF QE-style local radial table for numerical diagnostics.

PP_RAB Simpson, exact q=0, dq=.01 and four-node interpolation. Default D3FT
local potential remains unchanged. No fitted energy/charge offset.
"""
from contextlib import contextmanager
import numpy as np
from scipy.special import spherical_jn
import jax.numpy as jnp
from d3ft.pseudopotential.quadrature import qe_simpson_weights,restored_mesh
from d3ft.pseudopotential.geometry import safe_norm
import xml.etree.ElementTree as ET
from jrystal.pseudopotential.load import find_upf

@contextmanager
def qe_local_patch(directory):
 from d3ft import pseudopotential as pp
 from d3ft.pseudopotential import local
 original=pp.potential_local_reciprocal
 radial={}
 for symbol in ['Al','P']:
  root=ET.parse(find_upf(directory,symbol)).getroot();r=np.fromstring(root.findtext('PP_MESH/PP_R'),sep=' ');rab=np.fromstring(root.findtext('PP_MESH/PP_RAB'),sep=' ');radial[tuple(r[r>0])]=rab[r>0]
 def reference(positions,g,r_grid,v_grid,charges,vol,*,kmax=None,**kwargs):
  if kwargs:raise ValueError('Unsupported QE local diagnostic option')
  radius=safe_norm(g,axis=-1)
  if kmax is None:kmax=float(np.max(np.asarray(radius)))
  qq=np.arange(0.,kmax+.040001,.01);forms=[]
  for r,v,z in zip(r_grid,v_grid,charges):
   rn=np.asarray(r);vp=np.asarray(v)+z/rn
   re,ve,rab=restored_mesh(rn,vp[None],radial[tuple(rn)]);integrand=ve[0]*re*re*qe_simpson_weights(rab);table=np.empty(len(qq))
   for start in range(0,len(qq),256):table[start:start+256]=4*np.pi*(spherical_jn(0,qq[start:start+256,None]*re)@integrand)
   table=jnp.asarray(table);t=radius/.01;i=jnp.floor(t).astype(jnp.int32);p=t-i;u=1-p;vv=2-p;w=3-p
   f=table[i]*u*vv*w/6+table[i+1]*p*vv*w/2-table[i+2]*p*u*w/2+table[i+3]*p*u*vv/6
   safe=radius.at[0,0,0].set(1.);f=f-4*jnp.pi*z/safe**2;f=f.at[0,0,0].set(table[0]);forms.append(f)
  sf=jnp.exp(-1j*g@positions.T);return jnp.sum(jnp.stack(forms)*jnp.moveaxis(sf,-1,0),axis=0)*np.prod(g.shape[:-1])/vol
 pp.potential_local_reciprocal=reference
 try:yield
 finally:pp.potential_local_reciprocal=original
