"""Bound pointwise XC derivative tapes without changing quadrature or PBE.

FFT interpolation/gradient and the integral remain the original discrete program.
Only independent local XC evaluations are mapped in rematerialized chunks.
"""
from contextlib import contextmanager
import jax
import jax.numpy as jnp

@contextmanager
def memory_bounded_xc(chunk_size=65536):
    from d3ft.calc import xc_tunable as xc
    original=xc._exc_density
    def chunked(kind,rho,sigma,params):
        if rho.shape[0]!=1:raise ValueError('Restricted-spin XC response only')
        length=rho.size
        # Small tests use the original pointwise path; large quadrature maps chunks.
        size=min(chunk_size,length)
        padding=(-length)%size
        rf=jnp.pad(rho.reshape(-1),(0,padding),constant_values=.1).reshape(-1,size)
        sf=jnp.pad(sigma.reshape(-1),(0,padding),constant_values=.01).reshape(-1,size)
        def one(pair):
            r,s=pair
            return original(kind,r[None,:],s[None,:],params)
        value=jax.lax.map(jax.checkpoint(one),(rf,sf))
        return value.reshape(-1)[:length].reshape(rho.shape[1:])
    xc._exc_density=chunked
    try:yield
    finally:xc._exc_density=original
