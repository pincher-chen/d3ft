"""Numerical equivalence of rematerialized XC values, gradients and kernels."""
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
from bounded_xc import memory_bounded_xc
shape=(4,4,4);volume=100.;f=[jnp.fft.fftfreq(s)*s for s in shape]
g=jnp.stack(jnp.meshgrid(*f,indexing='ij'),axis=-1)*2*jnp.pi/volume**(1/3)
x=jnp.arange(64.).reshape(shape);rho=.2+.015*jnp.cos(x);d=.01*jnp.sin(.3*x)
def energy(r):return xc_energy_tunable(r[None,...],g,volume,'gga_x_pbe+gga_c_pbe',
           xc_params=default_xc_params('gga_x_pbe+gga_c_pbe'),grid_factor=2)
oldE=float(energy(rho));oldG=jax.grad(energy)(rho);oldK=jax.jvp(jax.grad(energy),(rho,),(d,))[1]
with memory_bounded_xc(chunk_size=17):
 newE=float(energy(rho));newG=jax.grad(energy)(rho);newK=jax.jvp(jax.grad(energy),(rho,),(d,))[1]
errors={'value':abs(newE-oldE),'gradient':float(jnp.max(abs(newG-oldG))),'kernel':float(jnp.max(abs(newK-oldK)))}
assert max(errors.values())<1e-11,errors
print(json.dumps({'bounded_XC_equivalence_errors':errors,'nondivisible_chunk_padding_checked':True,
 'scope':'real refined discrete-PBE test; quadrature and physics unchanged'},indent=2))
