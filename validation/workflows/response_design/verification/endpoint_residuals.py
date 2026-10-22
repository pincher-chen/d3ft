"""Portable orbital/occupation diagnostics adapted from imported source code.

The caller supplies the Hamiltonian of the exact SCF functional. This avoids
silently auditing tunable/refined XC with the stock XC potential. Equal integer
occupations are handled explicitly; no finite-temperature root is invented.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def orbital_residuals(vectors, actions, occupations):
    """One compact k block; occupations already contain k weights."""
    q,hq,f=map(np.asarray,[vectors,actions,occupations])
    if q.ndim!=2 or q.shape!=hq.shape or f.shape!=(q.shape[1],):
        raise ValueError('Compact vectors/actions/occupation dimensions disagree')
    if not all(np.isfinite(x).all() for x in [q,hq,f]):
        raise ValueError('Nonfinite orbital diagnostic input')
    h=q.conj().T@hq
    outside=hq-q@h
    commutator=h*f[None,:]-f[:,None]*h
    return dict(orbital_orthonormality_max=float(np.max(abs(q.conj().T@q-np.eye(q.shape[1])))),
        orbital_outside_weighted_norm_Ha=float(np.linalg.norm(outside*f[None,:])),
        orbital_outside_unweighted_norm_Ha=float(np.linalg.norm(outside)),
        orbital_outside_max_Ha=float(np.max(np.linalg.norm(outside,axis=0))),
        orbital_commutator_max_Ha=float(np.max(abs(commutator))),
        orbital_commutator_frobenius_Ha=float(np.linalg.norm(commutator)),
        orbital_hamiltonian_hermiticity_max_Ha=float(np.max(abs(h-h.conj().T))),
        rayleigh_energies_Ha=np.real(np.diag(h)).tolist(),
        scope='Subspace stationarity; does not prove this is the lowest occupied eigenspace.')


def occupation_residual(energies,occupations,electrons,capacity,temperature=None):
    e,f=map(np.asarray,[energies,occupations])
    if e.shape!=f.shape or not np.isfinite(e).all() or not np.isfinite(f).all():
        raise ValueError('Invalid occupation diagnostic arrays')
    if capacity<=0 or np.any(f < -1e-12) or np.any(f > capacity+1e-12):
        raise ValueError('Occupation capacity violated')
    result=dict(charge_error=abs(float(f.sum())-electrons))
    if temperature is None or temperature==0:
        error=float(np.max(np.minimum(abs(f),abs(f-capacity))))
        result.update(mode='fixed_integer',occupation_max_error=error,
            finite_T_FD_applicable=False,
            scope='Integer/capacity/charge test; band filling/gap requires independent low spectrum.')
        return result
    if temperature<0 or not 0<electrons<e.size*capacity:
        raise ValueError('Finite-T root needs positive T and non-boundary electron count')
    mu=brentq(lambda value:float(np.sum(capacity*expit((value-e)/temperature))-electrons),
        float(e.min()-100*temperature),float(e.max()+100*temperature),xtol=1e-14)
    target=capacity*expit((mu-e)/temperature)
    result.update(mode='finite_temperature',chemical_potential_Ha=mu,
        occupation_max_error=float(np.max(abs(f-target))),finite_T_FD_applicable=True)
    return result


def ritz_residuals(vectors,actions,eigenvalues):
    q,hq,e=map(np.asarray,[vectors,actions,eigenvalues])
    if q.shape!=hq.shape or e.shape!=(q.shape[1],):
        raise ValueError('Ritz dimensions disagree')
    values=np.linalg.norm(hq-q*e[None,:],axis=0)
    return dict(maximum_ritz_residual_Ha=float(np.max(values)),residuals_Ha=values.tolist(),
        orthogonality_max=float(np.max(abs(q.conj().T@q-np.eye(q.shape[1])))))
