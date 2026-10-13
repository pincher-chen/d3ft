"""Compare full-k posterior spectra and electron-measure density on identical voxels."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from physical_acceptance import check
import os
ROOT = Path(os.environ.get("D3FT_RUN_ROOT", Path(__file__).resolve().parents[1]))


def compare(state, audit, reference):
    refs = json.loads((ROOT/'inputs/QE_references.json').read_text())
    if reference in refs:
        entry = refs[reference]
        folder = Path(entry['folder'])
        if not folder.is_absolute(): folder = ROOT / folder
        for name, digest in entry['files'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    else:
        folder = ROOT/'results'/reference
        if not all((folder/name).exists() for name in ['summary.json','reference_manifest.json']):
            return dict(status='pending_reference', reference=reference, passed=False)
        entry = json.loads((folder/'reference_manifest.json').read_text())
        for name, digest in entry.items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    s = json.loads((state/'summary.json').read_text())
    p = json.loads((state/'provenance.json').read_text())
    c = json.loads((state/'config.json').read_text())
    q = json.loads((folder/'summary.json').read_text())
    assert abs(s['temperature_Ha']-q['physical_temperature_Ha']) < 1e-14
    ne = p['input']['valence_electrons']
    k = c['k_grid_sizes']
    axes = np.fft.fftfreq(k)
    mesh = np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),-1).reshape(-1,3)
    with np.load(audit/'low_spectrum.npz') as data:
        ours = data['energies_Ha'].copy()
    with np.load(folder/'spectrum.npz') as data:
        ref, points, cell = [data[n].copy() for n in ['energies_Ha','k_fractional','cell_bohr']]
    offset = mesh[:,None,:]-points[None,:,:]
    offset -= np.rint(offset)
    distance = np.max(abs(offset),axis=2)
    perm = np.argmin(distance,axis=1)
    err = float(distance[np.arange(k**3),perm].max())
    assert err < 1e-8 and len(set(perm.tolist())) == k**3
    nb = ne//2+c['empty_bands']
    assert min(ours.shape[1],ref.shape[1]) >= nb
    delta = ours[:,:nb]-ref[perm,:nb]
    shift = float(delta.mean())
    mae = float(np.mean(abs(delta-shift))*27.211386245988)
    lines = (folder/'rho.cube').read_text().splitlines()
    nat = int(lines[2].split()[0])
    dims = [int(x.split()[0]) for x in lines[3:6]]
    rho = np.fromstring(' '.join(lines[6+abs(nat):]),sep=' ').reshape(dims)
    with np.load(state/'final_state.npz') as data:
        ourrho = data['density_grid'].sum(axis=0)
    assert list(ourrho.shape) == dims, (ourrho.shape, dims)
    const = Path(__file__).resolve().parents[2]/'third_party/jrystal_dependency/jrystal/_src/const.py'
    conv = next(float(ast.literal_eval(n.value)) for n in ast.parse(const.read_text()).body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ANGSTROM2BOHR')
    ourcell = np.array(p['input']['cell_A'])*conv
    assert abs(ourcell-cell).max() < 1e-6
    dv = abs(np.linalg.det(cell))/np.prod(dims)
    odv = abs(np.linalg.det(ourcell))/np.prod(dims)
    assert abs(rho.sum()*dv-ne) < 1e-5 and abs(ourrho.sum()*odv-ne) < 1e-8
    density = float(abs(ourrho*odv-rho*dv).sum()/ne)
    fdiff = float(s['free_energy_physical_Ha']-q['free_energy_Ha'])
    tol = json.loads((ROOT/'metadata/protocol.json').read_text())['reference_tolerances']
    tests = dict(free_energy=abs(fdiff)<=tol['free_energy_Ha_cell'],
                 bands=mae<=tol['aligned_band_MAE_eV'],density=density<=tol['density_L1_per_electron'])
    passed = check(state,audit)
    nocc = ne//2
    gap = (ours[:,nocc].min()-ours[:,nocc-1].max())*27.211386245988
    qgap = (ref[:,nocc].min()-ref[:,nocc-1].max())*27.211386245988
    return dict(status='completed', passed=passed['passed'] and all(tests.values()),
                solver_acceptance=passed, cross_code_tests=tests, QE_reference=str(folder),
                free_energy_difference_Ha_cell=fdiff, aligned_band_MAE_eV=mae,
                aligned_band_RMSE_eV=float(np.sqrt(np.mean((delta-shift)**2))*27.211386245988),
                global_shift_Ha=shift, density_L1_per_electron=density,k_match_error=err,
                mesh_KS_gap_eV=float(gap),QE_mesh_KS_gap_eV=float(qgap),gap_difference_meV=float((gap-qgap)*1000),
                gap_scope='Discrete finite-T SCF mesh; no SOC; not continuous-BZ or experimental gap',
                density_convention='Electron measure at identical fractional voxels; no charge fit',
                source_checkpoint_sha256=hashlib.sha256((state/'final_state.npz').read_bytes()).hexdigest())
