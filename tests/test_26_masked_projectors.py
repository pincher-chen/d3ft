"""Masking drops only zero coefficients, including geometry derivatives."""
import os
os.environ["PROJ_MASK"] = "1"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from d3ft.pseudopotential import gk_vectors, projectors_reciprocal, energy_nonlocal, hamiltonian_nonlocal

axis = np.arange(-2, 2)
g = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).astype(float)
mask = np.linalg.norm(g, axis=-1) < 1.8
idx = np.flatnonzero(mask.ravel())
k = jnp.asarray([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.3]])
rng = np.random.default_rng(4)
c = rng.normal(size=(1, 2, 2, 4, 4, 4)) + 1j*rng.normal(size=(1, 2, 2, 4, 4, 4))
c = jnp.asarray(c * mask)
occ = jnp.ones((1, 2, 2)) / 2
l = [np.asarray([0, 1, 2])]
d = jnp.diag(jnp.asarray([0.8, 1.2, -.3]))

def energy(v, masked):
    gv = jnp.asarray(g) * (1+v[0])
    kv = k * (1+v[0])
    gk = gk_vectors(gv, kv, mask if masked else None)
    radius2 = jnp.sum(gk*gk, axis=-1)
    beta = [jnp.stack([jnp.exp(-radius2), jnp.exp(-0.5*radius2),jnp.exp(-.25*radius2)], axis=1)]
    pos = jnp.asarray([[0.2, 0.3, 0.4]]) + v[1:4]
    p = projectors_reciprocal(pos, gk, l, beta)
    return energy_nonlocal(c*(1+v[4]), p, 20., occ, d*(1+v[5]), freq_mask=mask)

v = jnp.asarray([0.03, 0.01, -0.02, 0.04, 0.2, 0.1])
np.testing.assert_allclose(energy(v, True), energy(v, False), rtol=1e-13)
gm = np.asarray(jax.grad(lambda x: energy(x, True))(v))
gf = np.asarray(jax.grad(lambda x: energy(x, False))(v))
np.testing.assert_allclose(gm, gf, rtol=1e-12, atol=1e-11)
for i in range(len(v)):
    h=np.zeros(len(v)); h[i]=1e-5
    fd=float((energy(v+h, True)-energy(v-h, True))/(2e-5))
    np.testing.assert_allclose(gm[i], fd, rtol=2e-7, atol=1e-7)
print("test_26: masked/full energy and strain/position/coefficient/KB gradients agree")

hamiltonians=[]
for masked in [False,True]:
    gk=gk_vectors(jnp.asarray(g),k,mask if masked else None)
    r2=jnp.sum(gk*gk,axis=-1)
    beta=[jnp.stack([jnp.exp(-r2),jnp.exp(-.5*r2),jnp.exp(-.25*r2)],axis=1)]
    proj=projectors_reciprocal(jnp.asarray([[.2,.3,.4]]),gk,l,beta)
    hamiltonians.append(np.asarray(hamiltonian_nonlocal(c,proj,20.,d,freq_mask=mask)))
np.testing.assert_allclose(hamiltonians[0],hamiltonians[1],rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonians[0]),np.linalg.eigvalsh(hamiltonians[1]),rtol=1e-12,atol=1e-12)
print('test_26: full nonlocal Hamiltonian and spectrum agree')
