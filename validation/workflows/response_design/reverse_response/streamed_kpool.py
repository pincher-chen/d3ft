"""Stream local occupied FFTs/kinetic bands, retaining the original k weights.

Only a local evaluation order changes. One common density and one psum remain;
no independent-k SCF approximation or extra Nk normalization is introduced.
Restricted spin is deliberate. Used by the GPU7 E6 memory-bounded pilot only.
"""
from contextlib import contextmanager
import numpy as np
import jax
import jax.numpy as jnp

def streamed_density(coeff,vol,occupation,block=4):
    if coeff.shape[0]!=1 or coeff.shape[2]%block:
        raise ValueError('Restricted-spin whole band blocks required')
    nk,nb=coeff.shape[1:3];shape=coeff.shape[-3:];n=int(np.prod(shape))
    chunks=coeff[0].reshape(nk*(nb//block),block,*shape)
    weights=occupation[0].reshape(-1,block)
    def one(rho,pair):
        c,w=pair
        wave=jnp.fft.ifftn(c,axes=(-3,-2,-1))*n/jnp.sqrt(vol)
        density=jnp.einsum('bxyz,b->xyz',jnp.real(wave*jnp.conj(wave)),w)
        return rho+density,None
    density,_=jax.lax.scan(jax.checkpoint(one),jnp.zeros(shape,dtype=coeff.real.dtype),(chunks,weights))
    return jax.lax.psum(density[None,...],'k')

def streamed_kinetic(g,kpts,coeff,occupation,block=4):
    if coeff.shape[0]!=1 or coeff.shape[2]%block:
        raise ValueError('Restricted-spin whole band blocks required')
    nk,nb=coeff.shape[1:3];shape=coeff.shape[-3:]
    chunks=coeff[0].reshape(nk*(nb//block),block,*shape)
    weights=occupation[0].reshape(-1,block);ks=jnp.repeat(kpts,nb//block,axis=0)
    def one(total,pair):
        c,w,k=pair;gk2=jnp.sum((g+k)**2,axis=-1)
        value=jnp.sum(jnp.real(c*jnp.conj(c))*w[:,None,None,None]*gk2[None,...])/2
        return total+value,None
    total,_=jax.lax.scan(jax.checkpoint(one),jnp.asarray(0.,dtype=coeff.real.dtype),(chunks,weights,ks))
    return jax.lax.psum(total,'k')

@contextmanager
def streamed_kpool_patch():
    from d3ft.parallel import kpool
    old_d,old_k=kpool._local_occupied_density,kpool._local_kinetic
    kpool._local_occupied_density=streamed_density;kpool._local_kinetic=streamed_kinetic
    try:yield
    finally:kpool._local_occupied_density=old_d;kpool._local_kinetic=old_k
