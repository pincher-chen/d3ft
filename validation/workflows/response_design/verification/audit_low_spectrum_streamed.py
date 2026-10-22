"""Audit independent low eigenspaces supplied as a k-block iterator.

The caller must independently solve eigenpairs and provide the exact physical
density and free-energy functionals. No stock-XC fallback or earlier solver study data path is
allowed. This helper does not itself solve eigenpairs or certify a full k mesh.
"""
import numpy as np
from .endpoint_residuals import ritz_residuals


def spectral_update(blocks,*,state_block,physical_free_energy,old_density,
                    old_free_energy,volume,electrons,expected_indices):
    seen=set();density=None;largest=0.;orth=0.
    reduction=dict(kinetic_Ha=0.,nonlocal_Ha=0.,entropy_dimensionless=0.)
    for index,vectors,actions,values,occupations in blocks:
        if index in seen:
            raise ValueError('Duplicate spectral k block')
        seen.add(index);check=ritz_residuals(vectors,actions,values)
        largest=max(largest,check['maximum_ritz_residual_Ha'])
        orth=max(orth,check['orthogonality_max'])
        contribution=state_block(index,vectors,occupations)
        part=np.asarray(contribution['density'])
        for name in reduction:
            reduction[name]+=float(contribution[name])
        density=part.copy() if density is None else density+part
    if seen!=set(expected_indices) or density is None:
        raise ValueError('Incomplete low spectrum')
    old=np.asarray(old_density)
    if density.shape!=old.shape:
        raise ValueError('Spectral update changed density voxels')
    # Kinetic, nonlocal and entropy contributions depend on the new orbitals/
    # occupations; the density alone is insufficient to reconstruct free energy.
    change=float(physical_free_energy(density,reduction))-float(old_free_energy)
    delta=float(np.sum(abs(density-old))*volume/old.size)
    return dict(full_mesh=True,maximum_ritz_residual_Ha=largest,
        spectrum_orthogonality_max=orth,spectral_update_density_L1_electrons=delta,
        spectral_update_density_L1_per_electron=delta/electrons,
        spectral_update_free_energy_change_Ha=change,
        spectral_update_free_energy_change_abs_Ha=abs(change),
        density=density,
        scope='One independently supplied spectral reoccupation, not a new SCF fixed point.')
