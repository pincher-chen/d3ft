"""Exact refined-grid reciprocal vectors without a dense Cartesian tensor.

Represent G as three one-dimensional frequency arrays and a 3x3 reciprocal
basis. Construct one Cartesian component inside the existing scalar FFT scan.
Fourier interpolation, Nyquist splitting, PBE and quadrature are unchanged.
"""
from contextlib import contextmanager
import inspect
import jax
import jax.numpy as jnp


@contextmanager
def compact_reciprocal_grid():
    from d3ft.calc import xc_tunable as xc
    old_refine,old_sigma=xc._refine_density_grid,xc.sigma_r_fn
    source=inspect.getsource(old_refine)
    anchor='''    fine_g = (frequencies[0][:, None, None, None] * reciprocal_steps[0]
              + frequencies[1][None, :, None, None] * reciprocal_steps[1]
              + frequencies[2][None, None, :, None] * reciprocal_steps[2])
    return refined, fine_g'''
    if source.count(anchor)!=1:raise RuntimeError('Refined reciprocal-grid source anchor changed')
    source=source.replace(anchor,'    return refined, (tuple(frequencies), jnp.stack(reciprocal_steps))',1)
    namespace=dict(jax=jax,jnp=jnp)
    exec(compile(source,inspect.getsourcefile(old_refine),'exec'),namespace)

    def sigma(density,g):
        if not isinstance(g,tuple):return old_sigma(density,g)
        if density.shape[0]!=1:raise ValueError('Restricted-spin compact reciprocal grid only')
        frequencies,basis=g
        reciprocal=jnp.fft.fftn(density,axes=(-3,-2,-1))
        def component(total,axis):
            weights=jax.lax.dynamic_index_in_dim(basis,axis,axis=1,keepdims=False)
            vector=(frequencies[0][:,None,None]*weights[0]
                    +frequencies[1][None,:,None]*weights[1]
                    +frequencies[2][None,None,:]*weights[2])
            gradient=jnp.fft.ifftn(reciprocal*1j*vector,axes=(-3,-2,-1))
            return total+jnp.real(jnp.conj(gradient)*gradient),None
        result,_=jax.lax.scan(jax.checkpoint(component),jnp.zeros_like(density),jnp.arange(3))
        return result

    xc._refine_density_grid=namespace['_refine_density_grid'];xc.sigma_r_fn=sigma
    try:yield
    finally:xc._refine_density_grid,xc.sigma_r_fn=old_refine,old_sigma


def parity_check():
    import numpy as np
    from jrystal._src.grid import g_vectors
    from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
    from bounded_sigma_fft import bounded_sigma_fft
    from bounded_xc import memory_bounded_xc
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(63);rho=jnp.asarray(.1+.02*rng.random((1,6,8,10)))
    dr=jnp.asarray(rng.normal(size=rho.shape))
    cell0=jnp.asarray([[7.,.1,.2],[.0,8.,.1],[.2,.1,9.]])
    d=jnp.array([.003,-.002,.001,.002,-.001,.0015]);dt=jnp.asarray(rng.normal(size=(6,)))
    params=default_xc_params('gga_x_pbe+gga_c_pbe')
    def energy(r,t):
        strain=jnp.array([[t[0],t[5]/2,t[4]/2],[t[5]/2,t[1],t[3]/2],[t[4]/2,t[3]/2,t[2]]])
        cell=cell0@(jnp.eye(3)+strain).T;volume=jnp.abs(jnp.linalg.det(cell))
        g=g_vectors(cell,np.asarray(r.shape[-3:]))
        return xc_energy_tunable(r,g,volume,'gga_x_pbe+gga_c_pbe',xc_params=params,grid_factor=2)
    def evaluate():
        value=energy(rho,d)
        grad_r,grad_t=jax.grad(energy,argnums=(0,1))(rho,d)
        hess_r,hess_t=jax.jvp(jax.grad(energy,argnums=(0,1)),(rho,d),(dr,dt))[1]
        def scalar(t):return jax.jvp(lambda r:energy(r,t),(rho,),(dr,))[1]
        mixed=jax.grad(scalar)(d)
        return [np.asarray(x) for x in [value,grad_r,grad_t,hess_r,hess_t,mixed]]
    with bounded_sigma_fft(),memory_bounded_xc(chunk_size=128):
        old=evaluate()
        with compact_reciprocal_grid():new=evaluate()
    labels=['energy','density_gradient','geometry_gradient','density_Hessian_vector','geometry_Hessian_vector','mixed_density_geometry_gradient']
    errors={name:float(np.max(np.abs(a-b))) for name,a,b in zip(labels,old,new)}
    if max(errors.values())>1e-10:raise AssertionError(errors)
    return dict(passed=True,errors=errors,scope='Same refined PBE quadrature, complex/Nyquist density gradients, density/geometry first derivatives and joint Hessian-vector/mixed derivatives; compressed reciprocal basis only.')


if __name__=='__main__':
    import json
    print(json.dumps(parity_check(),indent=2))
