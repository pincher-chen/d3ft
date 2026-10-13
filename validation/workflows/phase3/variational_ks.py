"""Workflow KS operator matching the tunable, possibly refined SCF functional.

The existing matrix-free operator computes the stock XC potential internally.
An explicit local-potential correction replaces that term with the derivative
of the actual discrete SCF XC energy. The ground-state scaled KB matrix is
passed directly. No changes to the installed/reference code are needed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from d3ft.calc.xc_tunable import xc_energy_tunable
from d3ft.spectrum.ks_matvec import make_ks_hvp, _xc_potential_autodiff
from d3ft import pseudopotential as d3pp
from jrystal._src.grid import g_vectors


def operator_factory(out):
    if out.config.spin != 0 or out.density_grid.shape[0] != 1:
        raise ValueError('This workflow currently supports one restricted spin channel')
    cell=jnp.asarray(out.crystal.cell_vectors)
    positions=jnp.asarray(out.crystal.positions)
    volume=jnp.abs(jnp.linalg.det(cell))
    shape=np.asarray(out.freq_mask.shape)
    g=g_vectors(cell,shape)
    rho=jnp.real(out.density_grid)
    core=out.core_density_grid
    factor=out.config.d3ft.get('xc_grid_factor',1)
    params=out.config.d3ft.get('xc_params',None)
    def energy(density):
        return xc_energy_tunable(density,g,volume,out.config.xc,
                                 core_density_grid=core,xc_params=params,grid_factor=factor)
    actual=jax.grad(energy)(rho)[0]*float(np.prod(shape))/volume
    rho_stock=rho if core is None else rho+core[None,...]
    stock=_xc_potential_autodiff(rho_stock,g,out.config.xc,3)
    pp=out.pseudopot
    local=d3pp.potential_local_reciprocal(positions,g,pp.r_grid,pp.local_potential_grid,
                                        pp.local_potential_charge,volume,kmax=out.grid_kmax)
    local_corrected=jnp.fft.ifftn(local,axes=(-3,-2,-1))+actual-stock
    reciprocal=2*jnp.pi*jnp.linalg.inv(cell).T
    kcart=jnp.asarray(out.k_frac)@reciprocal

    def at_k(index):
        k=kcart[index:index+1]
        gk=d3pp.gk_vectors(g,k,out.freq_mask)
        beta=d3pp.beta_grid(pp.r_grid,pp.nonlocal_beta_grid,pp.nonlocal_angular_momentum,
                           gk,out.beta_kmax,tables=out.beta_tables)
        projectors=d3pp.projectors_reciprocal(positions,gk,pp.nonlocal_angular_momentum,beta)
        pk=jax.tree_util.tree_map(lambda x:x[0],projectors)
        op,dimension=make_ks_hvp(rho,local,pk,g,k[0],volume,out.freq_mask,xc=out.config.xc,
                           core_density_grid=core,nl_d_matrix=out.nl_d_matrix,
                           potential_loc_real=local_corrected)
        op.kinetic_diagonal=np.asarray(jnp.sum(gk[0]**2,axis=-1)/2)
        return op,dimension
    return at_k
