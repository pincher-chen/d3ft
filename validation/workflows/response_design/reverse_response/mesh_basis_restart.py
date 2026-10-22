"""Map owned orbitals by ordered Miller indices; never reuse their density."""
import numpy as np


def project_miller_coefficients(pw, old_mask, new_mask):
    old_mask, new_mask = np.asarray(old_mask, bool), np.asarray(new_mask, bool)
    if old_mask.shape != new_mask.shape or pw.shape[-2] != int(old_mask.sum()):
        raise ValueError('Miller grid or compact coefficient dimensions mismatch')
    old = np.flatnonzero(old_mask.reshape(-1))
    new = np.flatnonzero(new_mask.reshape(-1))
    common, old_indices, new_indices = np.intersect1d(old, new, return_indices=True)
    out = np.zeros((*pw.shape[:-2], len(new), pw.shape[-1]), dtype=pw.dtype)
    out[..., new_indices, :] = pw[..., old_indices, :]
    return out, dict(common_G=len(common), old_G=len(old), new_G=len(new),
                     new_G_initialized_to_zero=len(new)-len(common),
                     density_reused=False, orthonormalization_required=True)
