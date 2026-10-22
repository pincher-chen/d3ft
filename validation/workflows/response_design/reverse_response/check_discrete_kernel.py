"""Check the actual discrete PBE/Hartree kernel, including XC refinement."""
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
shape=(4,4,4);n=np.prod(shape);volume=100.
f=[jnp.fft.fftfreq(s)*s for s in shape]
g=jnp.stack(jnp.meshgrid(*f,indexing='ij'),axis=-1)*2*jnp.pi/volume**(1/3)
x=jnp.arange(64.).reshape(shape)
rho=.2+.015*jnp.cos(x);d=.01*jnp.sin(.3*x);other=.02*jnp.cos(.7*x)
results=[]
for factor in [1,2]:
 def hxc(r):
  rec=jnp.fft.fftn(r);g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
  vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*rec/g2).at[0,0,0].set(0.)))
  def energy(z):return xc_energy_tunable(z[None,...],g,volume,'gga_x_pbe+gga_c_pbe',
       xc_params=default_xc_params('gga_x_pbe+gga_c_pbe'),grid_factor=factor)
  return vh+jax.grad(energy)(r)*n/volume
 fn=jax.jit(hxc);kernel=jax.jit(lambda z:jax.jvp(hxc,(rho,),(z,))[1])
 analytic=kernel(d);step=1e-4;fd=(fn(rho+step*d)-fn(rho-step*d))/(2*step)
 relative=float(jnp.linalg.norm(analytic-fd)/jnp.linalg.norm(analytic))
 lhs=float(jnp.vdot(d,kernel(other)));rhs=float(jnp.vdot(kernel(d),other))
 symmetry=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
 assert relative<1e-6 and symmetry<1e-10,(factor,relative,symmetry)
 results.append(dict(xc_grid_factor=factor,kernel_FD_relative_error=relative,real_symmetry_error=symmetry))
print(json.dumps({'discrete_Hartree_PBE_kernel_checks':results,'scope':'Small real discrete-kernel check; not material-level gradient acceptance'},indent=2))
