"""Independent full-potential versus scalar XC pullback under cell strain."""
import json
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from d3ft.calc.xc_tunable import xc_energy_tunable, default_xc_params
from bounded_xc import memory_bounded_xc

shape=(4,4,4); n=64; v0=100.
freq=jnp.stack(jnp.meshgrid(*[jnp.fft.fftfreq(s)*s for s in shape],indexing='ij'),axis=-1)
x=jnp.arange(n,dtype=float).reshape(shape)
rho0=.2+.01*jnp.cos(x); weight=.03*jnp.sin(.3*x)
def terms(theta):
    cell=v0**(1/3)*jnp.array([[1+theta[0],theta[1],0.],[0.,1.,0.],[0.,0.,1.]])
    volume=jnp.linalg.det(cell);g=2*jnp.pi*freq@jnp.linalg.inv(cell)
    rho=rho0*v0/volume
    def energy(r):
        return xc_energy_tunable(r[None,...],g,volume,'gga_x_pbe+gga_c_pbe',
            xc_params=default_xc_params('gga_x_pbe+gga_c_pbe'),grid_factor=2)
    return rho,volume,energy
def original(theta):
    rho,volume,energy=terms(theta)
    return jnp.sum(weight*jax.grad(energy)(rho))/volume
def scalar(theta):
    rho,volume,energy=terms(theta)
    return jax.jvp(energy,(rho,),(weight/volume,))[1]
with memory_bounded_xc(chunk_size=17):
    theta=jnp.array([.003,-.002])
    errors=dict(value=float(abs(original(theta)-scalar(theta))),
        geometric_gradient=float(jnp.max(abs(jax.grad(original)(theta)-jax.grad(scalar)(theta)))))
assert max(errors.values())<1e-11,errors
print(json.dumps(dict(scalar_XC_pullback_equivalence=errors,
    scope='Actual refined PBE, volume and reciprocal-cell dependence; independent potential expression'),indent=2))

# Compare all scalar contraction weights with the actual FFT/KB Hamiltonian
# action on independent complex orbitals, not only the XC identity above.
import numpy as np
from gamma_response import GammaResponse
from projected_solver import real_inner
r=object.__new__(GammaResponse);r.shape=shape;r.n=n;r.block=1
r.idx=jnp.arange(16);r.volume=jnp.asarray(v0)
rng=np.random.default_rng(217)
def complex_random(s):return jnp.asarray(rng.normal(size=s)+1j*rng.normal(size=s))
basis=jnp.linalg.qr(complex_random((16,4)))[0];r.q=basis[:,:3];r.edges=basis[:,2:4]
lam=complex_random((16,3));lam=lam-r.q@(r.q.conj().T@lam)
r.nl=jnp.array([[.2,.04],[.04,-.1]])
beta=complex_random((2,3,16));pot=.02*jnp.cos(x)
eg=jnp.abs(r.edges[:,1])**2-jnp.abs(r.edges[:,0])**2
er=(r.density(r.edges[:,1:2],v0,1.)-r.density(r.edges[:,:1],v0,1.))*v0
wg=eg-4*jnp.real(jnp.sum(jnp.conj(lam)*r.q,axis=1))
wr=er-r.density_tangent(lam)*v0
def inputs(t):return jnp.arange(16.)*(1+t[0]),beta*(1+.03*t[1]),pot*(1+t[1]),v0*(1+t[0])
def hamiltonian_contract(t):
    args=inputs(t);hq=r.ham_action(r.q,*args);he=r.ham_action(r.edges,*args)
    return real_inner(r.edges[:,1],he[:,1])-real_inner(r.edges[:,0],he[:,0])-4*real_inner(lam,hq)
def scalar_contract(t):
    diagonal,projectors,potential,volume=inputs(t)
    edge=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),r.edges)
    occ=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),r.q)
    adj=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),lam)
    def coupling(a,b):return jnp.real(jnp.einsum('ami,ab,bmi->',jnp.conj(a),r.nl,b))
    nl=(coupling(edge[:,:,1:2],edge[:,:,1:2])-coupling(edge[:,:,:1],edge[:,:,:1])-4*coupling(adj,occ))/volume
    return jnp.sum(wg*diagonal)+jnp.sum(wr*potential)/n+nl
all_errors=dict(value=float(abs(hamiltonian_contract(theta)-scalar_contract(theta))),
    gradient=float(jnp.max(abs(jax.grad(hamiltonian_contract)(theta)-jax.grad(scalar_contract)(theta)))))
assert max(all_errors.values())<1e-10,all_errors
print(json.dumps(dict(full_complex_Hamiltonian_contraction=all_errors,
    scope='Independent FFT Hamiltonian action versus scalar kinetic/local/KB contraction'),indent=2))
