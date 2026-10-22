"""Path-free matched-spectrum/density metrics; references explicitly supplied."""
import numpy as np
HA_EV=27.211386245988


def spectra(ours,reference,k_ours,k_reference,*,allow_reciprocal_folding=False):
    a,b,ka,kb=map(np.asarray,[ours,reference,k_ours,k_reference])
    if a.shape!=b.shape or a.shape[0]!=len(ka) or ka.shape!=kb.shape:
        raise ValueError('Spectrum/k-point shapes differ')
    offset=ka[:,None,:]-kb[None,:,:]
    if allow_reciprocal_folding:
        offset-=np.rint(offset)
    distances=np.max(abs(offset),axis=2);permutation=np.argmin(distances,axis=1)
    error=float(np.max(distances[np.arange(len(ka)),permutation]))
    if error>1e-8 or len(set(permutation.tolist()))!=len(ka):
        raise ValueError('No unique same-representative k mapping')
    delta=a-b[permutation];shift=float(delta.mean());aligned=delta-shift
    return dict(global_shift_Ha=shift,aligned_band_MAE_eV=float(abs(aligned).mean()*HA_EV),
        aligned_band_RMSE_eV=float(np.sqrt(np.mean(aligned**2))*HA_EV),k_match_error=error,
        reciprocal_folding_used=allow_reciprocal_folding,
        scope='Global spectral alignment is reported; it does not align total energies or absolute deformation potentials.')


def density_measure(ours,reference,cell,electrons):
    a,b,cell=map(np.asarray,[ours,reference,cell])
    if a.shape!=b.shape or a.ndim!=3 or cell.shape!=(3,3) or electrons<=0:
        raise ValueError('Density comparison requires matching fractional voxels and cell')
    dv=abs(np.linalg.det(cell))/a.size
    return dict(charge_ours=float(a.sum()*dv),charge_reference=float(b.sum()*dv),
        density_L1_per_electron=float(np.sum(abs(a-b))*dv/electrons),
        scope='No charge renormalization or fitted density scaling.')
