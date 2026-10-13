"""XC quadrature refinement agrees with an independently sampled fine grid."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from d3ft.calc.xc_tunable import _refine_density_grid,xc_energy_tunable,default_xc_params


def grid(n,a):
    f=jnp.fft.fftfreq(n)*n
    return jnp.stack(jnp.meshgrid(f,f,f,indexing='ij'),axis=-1)*2*jnp.pi/a


def density(n,mode,amplitude=.1):
    x,y,z=jnp.meshgrid(*(jnp.arange(n)/n,)*3,indexing='ij')
    return (1+amplitude*jnp.cos(2*jnp.pi*x)+.03*jnp.cos(2*jnp.pi*z)
            +.02*jnp.cos(2*jnp.pi*mode*x)*jnp.cos(2*jnp.pi*mode*y))[None]


for n in [7,8]:
    mode=n//2
    fine,g=_refine_density_grid(density(n,mode),grid(n,5.),4)
    np.testing.assert_allclose(fine,density(n*4,mode),atol=3e-15,rtol=0)
    np.testing.assert_allclose(g,grid(n*4,5.),atol=1e-14,rtol=0)
    np.testing.assert_allclose(fine.mean(),density(n,mode).mean(),atol=1e-14,rtol=0)


def energy(parameters,n,factor):
    amp,a,beta=parameters
    rho=density(n,2,amp)/a**3
    params=default_xc_params('gga_x_pbe+gga_c_pbe')
    params['gga_c_pbe']['beta']=beta
    return xc_energy_tunable(rho,grid(n,a),a**3,'gga_x_pbe+gga_c_pbe',xc_params=params,grid_factor=factor)


p=jnp.array([.2,5.,.066])
coarse=jax.jit(jax.value_and_grad(lambda x:energy(x,8,4)))(p)
fine=jax.jit(jax.value_and_grad(lambda x:energy(x,32,1)))(p)
for actual,expected in zip(coarse,fine):
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-11)
try:
    xc_energy_tunable(density(8,2),grid(8,5.),125.,'gga_x_pbe+gga_c_pbe',
                      core_density_grid=jnp.ones((8,8,8)),grid_factor=2)
except ValueError as error:
    assert 'NLCC' in str(error)
else:
    raise AssertionError('Unsupported sampled NLCC was silently interpolated')
print('test_30: even/odd Fourier interpolation, mean, XC energy and density/cell/parameter gradients passed')
