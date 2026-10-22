"""CPU-only geometry preflight for the archived two-atom cubic worker.

Reads the actual jrystal unit/FFT table; no GPU initialization or SCF solve.
Spherical masks are checked against archived raw masks before production use.
"""
import ast
from pathlib import Path
import numpy as np


REFERENCE = Path('${D3FT_ROOT}/third_party/jrystal_dependency/jrystal/_src')


def dependency_constants():
    values = {}
    tree = ast.parse((REFERENCE/'const.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ['ANGSTROM2BOHR','CUFFT_FACTORS']:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {'ANGSTROM2BOHR','CUFFT_FACTORS'}:
        raise ValueError('Dependency constants changed format')
    return values


def actual_fft_size(requested):
    choices = dependency_constants()['CUFFT_FACTORS']
    return next(n for n in choices if n >= requested)


def cubic_cell(a0):
    # Reproduce the archived extXYZ writer's 12-decimal Angstrom lattice.
    half = float(f'{a0/2:.12f}')
    return np.array([[0,half,half],[half,0,half],[half,half,0]]) * dependency_constants()['ANGSTROM2BOHR']


def sphere_from_cell(cell_bohr, grid, cutoff):
    if actual_fft_size(grid) != grid:
        raise ValueError('Preflight grid must be the actual FFT size')
    b = 2*np.pi*np.linalg.inv(cell_bohr).T
    axis = np.fft.fftfreq(grid, 1/grid)
    g = (axis[:,None,None,None]*b[0] + axis[None,:,None,None]*b[1] +
         axis[None,None,:,None]*b[2])
    return np.linalg.norm(g,axis=-1)**2 <= cutoff*2


def sphere(a0, grid, cutoff):
    return sphere_from_cell(cubic_cell(a0),grid,cutoff)


def density_support(mask):
    if mask.dtype != bool or mask.ndim != 3 or not mask.any():
        raise ValueError('Require a nonempty three-dimensional Boolean mask')
    indices = np.column_stack(np.nonzero(mask))
    millers = np.column_stack([np.rint(np.fft.fftfreq(n)*n)[indices[:,i]]
        for i,n in enumerate(mask.shape)]).astype(int)
    bound = np.abs(millers).max(axis=0)
    return dict(max_abs_miller=bound.tolist(), density_frequency_bound=(2*bound).tolist(),
        fft_shape=list(mask.shape), full_density_support=bool(np.all(4*bound<np.array(mask.shape))))
