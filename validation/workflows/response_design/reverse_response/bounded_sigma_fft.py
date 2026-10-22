"""Equivalent restricted-spin sigma, one Cartesian FFT at a time.

Keep complex gradient magnitudes exactly as in jrystal, including Nyquist
terms. This reduces the cuFFT work batch from three to one; it does not change
the FFT grid, Fourier interpolation or discrete functional.
"""
from contextlib import contextmanager
import jax
import jax.numpy as jnp

def scalar_sigma(density, g):
    if density.shape[0]!=1:raise ValueError('Restricted-spin numerical verification only')
    reciprocal=jnp.fft.fftn(density,axes=(-3,-2,-1))
    def component(total, axis):
        vector=jax.lax.dynamic_index_in_dim(g,axis,axis=-1,keepdims=False)
        gradient=jnp.fft.ifftn(reciprocal*1j*vector,axes=(-3,-2,-1))
        return total+jnp.real(jnp.conj(gradient)*gradient),None
    result,_=jax.lax.scan(jax.checkpoint(component),jnp.zeros_like(density),jnp.arange(3))
    return result

@contextmanager
def bounded_sigma_fft():
    from jrystal._src import xc as reference
    from d3ft.calc import xc_tunable as tunable
    old_reference,old_tunable=reference.sigma_r_fn,tunable.sigma_r_fn
    reference.sigma_r_fn=tunable.sigma_r_fn=scalar_sigma
    try:yield
    finally:reference.sigma_r_fn,tunable.sigma_r_fn=old_reference,old_tunable

def parity_check():
    import numpy as np
    from jrystal._src.xc import sigma_r_fn
    from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
    from jrystal._src.grid import g_vectors
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(29);rho=jnp.asarray(.1+.02*rng.random((1,6,8,10)))
    g=g_vectors(jnp.asarray([[7.,.1,.2],[.0,8.,.1],[.2,.1,9.]]),jnp.asarray([6,8,10]))
    old=sigma_r_fn(rho,g);new=scalar_sigma(rho,g)
    err=float(jnp.max(jnp.abs(old-new)))
    params=default_xc_params('gga_x_pbe+gga_c_pbe')
    def energy(r):return xc_energy_tunable(r,g,jnp.array(503.),'gga_x_pbe+gga_c_pbe',xc_params=params,grid_factor=2)
    tangent=jnp.asarray(rng.normal(size=rho.shape));gradient_old=jax.grad(energy)(rho)
    hv_old=jax.jvp(jax.grad(energy),(rho,),(tangent,))[1];energy_old=float(energy(rho))
    with bounded_sigma_fft():
        energy_new=float(energy(rho));gradient_new=jax.grad(energy)(rho)
        hv_new=jax.jvp(jax.grad(energy),(rho,),(tangent,))[1]
    errors=dict(sigma_max_error=err,energy_error=abs(energy_new-energy_old),
        XC_gradient_max_error=float(jnp.max(jnp.abs(gradient_new-gradient_old))),
        XC_Hessian_vector_max_error=float(jnp.max(jnp.abs(hv_new-hv_old))))
    if max(errors.values())>1e-10:raise AssertionError(errors)
    return dict(passed=True,**errors,scope='Same discrete restricted PBE energy, first derivative and Hessian-vector; complex/Nyquist terms retained; no resolution change.')

if __name__=='__main__':
    import json
    print(json.dumps(parity_check(),indent=2))
