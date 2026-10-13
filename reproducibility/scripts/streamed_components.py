"""Exact k-additive density/kinetic/nonlocal reduction, one k point at a time."""
import jax
import jax.numpy as jnp
import numpy as np
from d3ft.parallel import plain_fft_patch
from d3ft import pseudopotential as pp_module
from jrystal._src import pw, energy


def components(vectors, occupations, *, mask, gvec, kvec, projectors, crystal, out, cfg, vloc):
    # Occupations already include the original full-grid capacity 2/K.
    # Never renormalize the weight of a single-k block.
    f = np.asarray(occupations)
    assert f.shape == (1, len(vectors), vectors[0].shape[1])
    indices = jnp.asarray(np.flatnonzero(np.asarray(mask).reshape(-1)))

    @jax.jit
    def block(u, occupation, k, p):
        c = jnp.zeros((1, 1, u.shape[1], int(np.prod(mask.shape))), dtype=jnp.complex128)
        c = c.at[:,:,:,indices].set(u.T[None,None,...])
        c = c.reshape((1, 1, u.shape[1], *mask.shape))
        return (pw.density_grid(c, crystal.vol, occupation),
                energy.kinetic(gvec, k, c, occupation),
                pp_module.energy_nonlocal(c, p, vol=crystal.vol, occupation=occupation,
                    d_matrix=out.nl_d_matrix, freq_mask=mask))

    rho = jnp.zeros_like(out.density_grid)
    kinetic, nonlocal_energy = 0., 0.
    with plain_fft_patch():
        for ik, u in enumerate(vectors):
            p = jax.tree.map(lambda x: x[ik:ik+1], projectors)
            density, ek, enl = block(jnp.asarray(u), jnp.asarray(f[:,ik:ik+1]), kvec[ik:ik+1], p)
            rho = (rho + density).block_until_ready()
            kinetic += float(ek)
            nonlocal_energy += float(enl)
        rec = jnp.fft.fftn(rho, axes=(-3,-2,-1))
        result = dict(kinetic_Ha=kinetic,
            hartree_Ha=float(energy.hartree(rec, gvec, crystal.vol)),
            local_Ha=float(pp_module.energy_local(rec, vloc, vol=crystal.vol)),
            nonlocal_Ha=nonlocal_energy,
            xc_Ha=float(energy.xc_energy(rho, gvec, crystal.vol, cfg.xc, kohn_sham=False,
                core_density_grid=out.core_density_grid)))
    return result, rho
