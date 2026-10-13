"""Independent NLCC Hamiltonian candidate matching QE7.5 rhoc_mod/set_rhoc.

PP_RAB Simpson radial integration; dq=.01 bohr^-1 four-node interpolation;
explicit density reciprocal sphere. No charge renormalization or fitted offset.
The legacy no-RAB path is retained for PP_RHOATOM initial-density callers.
Adapted from jrystal pseudopotential NLCC (Apache-2.0); see project NOTICE.
"""
import numpy as np
import jax.numpy as jnp
from scipy.special import spherical_jn
from interpax import CubicSpline
from jrystal.sbt import sbt_numerical
from .geometry import safe_norm
from .quadrature import qe_simpson_weights, restored_mesh

def core_charge_reciprocal(positions,g_vector_grid,r_grid,core_charge_grid,vol,
                           kmax=None,*,r_ab=None,density_cutoff_Ha=None):
    radius=safe_norm(g_vector_grid,axis=-1)
    if kmax is None:kmax=float(np.max(np.asarray(radius)))
    ngrid=np.prod(g_vector_grid.shape[:-1]);rho_g=[]
    if r_ab is not None:
        if density_cutoff_Ha is None or density_cutoff_Ha<=0:raise ValueError('QE NLCC requires positive explicit density cutoff in Ha')
        if len(r_grid)!=len(r_ab) or len(r_grid)!=len(core_charge_grid):raise ValueError('NLCC radial metadata lengths differ')
    for idx,(r,rc) in enumerate(zip(r_grid,core_charge_grid)):
        if r_ab is None:
            kk,fk=sbt_numerical(r,jnp.expand_dims(rc,0),l=0,kmax=kmax)
            radial=CubicSpline(kk,fk[0],axis=1)(radius)
        else:
            r,rc_,rab=restored_mesh(r,np.asarray(rc)[None],r_ab[idx]);rc=rc_[0]
            q=np.arange(0.,kmax+.040001,.01);integrand=rc*r*r*qe_simpson_weights(rab);tab=np.empty(len(q))
            for start in range(0,len(q),256):tab[start:start+256]=spherical_jn(0,q[start:start+256,None]*r)@integrand
            tab=jnp.asarray(tab);t=radius/.01;i=jnp.floor(t).astype(jnp.int32);p=t-i;u=1-p;v=2-p;w=3-p
            radial=tab[i]*u*v*w/6+tab[i+1]*p*v*w/2-tab[i+2]*p*u*w/2+tab[i+3]*p*u*v/6
        rho_g.append(radial*4*jnp.pi)
    forms=jnp.stack(rho_g)
    sf=jnp.exp(-1j*jnp.matmul(g_vector_grid,positions.T))
    out=jnp.sum(forms*jnp.moveaxis(sf,-1,0),axis=0)*ngrid/vol
    if r_ab is not None:out=jnp.where(jnp.sum(g_vector_grid**2,axis=-1)<=2*density_cutoff_Ha+1e-10,out,0.)
    return out

def core_charge_real(positions,g_vector_grid,r_grid,core_charge_grid,vol,kmax=None,
                     *,r_ab=None,density_cutoff_Ha=None):
    rec=core_charge_reciprocal(positions,g_vector_grid,r_grid,core_charge_grid,vol,kmax,
                             r_ab=r_ab,density_cutoff_Ha=density_cutoff_Ha)
    return jnp.real(jnp.fft.ifftn(rec,axes=range(-g_vector_grid.shape[-1],0)))
