"""Gamma orbitals can seed a mesh; Gamma occupation weights cannot."""
import numpy as np


def gamma_orbital_warm_start(parent, nk):
    """Reuse orbitals only; calc creates target-mesh occupations via its API.

    In particular uniform() gives 2/Nk, whereas repeating stored Gamma
    occupations gives 2 at *every* k. No posterior charge renormalisation.
    This entry deliberately supports the existing dense Stiefel array only.
    """
    pw = np.asarray(parent['pw'])
    if pw.ndim != 4 or pw.shape[1] != 1 or nk < 1:
        raise ValueError('Expected dense (spin,1,G,band) Gamma orbitals')
    return {'pw': np.repeat(pw, nk, axis=1)}
