"""Exact scalar geometry gradient with bounded sequential AD contractions.

This changes only the local six-coordinate differentiation order. The
self-consistent non-energy response still uses one full-basis adjoint solve.
No finite differences or extra electronic-response solves are introduced.
"""
import jax
import jax.numpy as jnp


def sequential_scalar_gradient(function, coordinates=6):
    directional=jax.jit(lambda x,a,b,c,t:
        jax.jvp(lambda z:function(z,a,b,c),(x,),(t,))[1])

    def gradient(x,a,b,c):
        if x.shape!=(coordinates,):raise ValueError('Unexpected geometry coordinate shape')
        values=[]
        # A Python loop with synchronization avoids retaining six large FFT
        # intermediates in one executable. Every entry is an exact AD JVP.
        for axis in range(coordinates):
            value=directional(x,a,b,c,jnp.eye(coordinates,dtype=x.dtype)[axis])
            jax.block_until_ready(value)
            values.append(value)
        return jnp.stack(values)

    return gradient


def parity_check():
    """Mixed PBE density/geometry derivatives, including a complex term."""
    import numpy as np
    from jrystal._src.grid import g_vectors
    from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
    from bounded_sigma_fft import bounded_sigma_fft
    from bounded_xc import memory_bounded_xc
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(51)
    rho=jnp.asarray(.15+.02*rng.random((6,8,10)))
    cell0=jnp.asarray([[7.,.1,.2],[.0,8.,.1],[.2,.1,9.]])
    v0=jnp.abs(jnp.linalg.det(cell0))
    params=default_xc_params('gga_x_pbe+gga_c_pbe')
    projector=jnp.asarray(rng.normal(size=(3,5))+1j*rng.normal(size=(3,5)))
    occupied=jnp.asarray(rng.normal(size=(5,4))+1j*rng.normal(size=(5,4)))
    weight_g=jnp.asarray(rng.normal(size=(5,)))
    weight_r=jnp.asarray(.01*rng.normal(size=rho.shape))
    def scalar_geometry(d,wg,wr,lam):
        strain=jnp.array([[d[0],d[5]/2,d[4]/2],[d[5]/2,d[1],d[3]/2],[d[4]/2,d[3]/2,d[2]]])
        cell=cell0@(jnp.eye(3)+strain).T
        volume=jnp.abs(jnp.linalg.det(cell));g=g_vectors(cell,np.asarray(rho.shape))
        density=rho*v0/volume
        rec=jnp.fft.fftn(density);g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
        vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*rec/g2).at[0,0,0].set(0.)))
        def energy(r):return xc_energy_tunable(r[None,...],g,volume,
            'gga_x_pbe+gga_c_pbe',xc_params=params,grid_factor=2)
        xc_direction=jax.jvp(energy,(density,),(wr/volume,))[1]
        beta=projector*jnp.exp(1j*(d[0]+2*d[3]))
        nonlocal_term=jnp.real(jnp.vdot(beta@lam,beta@occupied))/volume
        kinetic=jnp.dot(wg,jnp.sum(g.reshape(-1,3)[:5]**2,axis=-1)/2)
        return kinetic+jnp.sum(wr*vh)/rho.size+nonlocal_term+xc_direction
    records=[]
    with bounded_sigma_fft(),memory_bounded_xc(chunk_size=128):
        exact=jax.jit(jax.grad(scalar_geometry,argnums=0))
        bounded=sequential_scalar_gradient(scalar_geometry)
        for d,lam in [(jnp.zeros(6),jnp.zeros_like(occupied)),
                      (jnp.array([.003,-.002,.001,.002,-.001,.0015]),occupied*.23)]:
            old=np.asarray(exact(d,weight_g,weight_r,lam));new=np.asarray(bounded(d,weight_g,weight_r,lam))
            error=float(np.max(np.abs(old-new)))
            if error>1e-10:raise AssertionError((old,new,error))
            records.append(dict(max_absolute_error=error,passed=True))
    return dict(passed=True,cases=records,
        scope='Same scalar mixed PBE/geometry/Hartree/complex-projector derivative: reverse-over-forward versus six sequential forward-over-forward AD contractions; no finite differences.')


if __name__=='__main__':
    import json
    print(json.dumps(parity_check(),indent=2))
