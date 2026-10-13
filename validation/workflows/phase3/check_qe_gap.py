"""Check a converged, nonmagnetic fixed-occupation QE point on its SCF mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from qe_precision import read_state


def inspect(folder, threshold_ev=1e-5):
    values, xml = read_state(folder)
    root = ET.parse(xml).getroot()
    band = root.find('output/band_structure')
    if band is None:
        raise ValueError('Missing band structure')
    for key in ('lsda', 'noncolin', 'spinorbit'):
        if band.findtext(key) != 'false':
            raise ValueError('Only scalar nonmagnetic bands are supported')
    if band.findtext('occupations_kind') != 'fixed':
        raise ValueError('Expected fixed occupations')
    electrons = float(band.findtext('nelec'))
    occupied = round(electrons / 2)
    if abs(electrons - 2 * occupied) > 1e-8 or occupied < 1:
        raise ValueError('Expected a positive even electron count')
    nbnd = int(band.findtext('nbnd'))
    if nbnd <= occupied:
        raise ValueError('No empty bands: cannot establish a gap')
    rows = []
    for point in band.findall('ks_energies'):
        eig = np.fromstring(point.findtext('eigenvalues'), sep=' ')
        occ = np.fromstring(point.findtext('occupations'), sep=' ')
        if len(eig) != nbnd or len(occ) != nbnd or not np.isfinite(eig).all():
            raise ValueError('Invalid eigenvalues or occupations')
        if (np.any(np.diff(eig) < -1e-10)
                or not np.allclose(occ[:occupied], 1, atol=1e-10, rtol=0)
                or not np.allclose(occ[occupied:], 0, atol=1e-10, rtol=0)):
            raise ValueError('Bands are unordered or occupations differ from the fixed filling')
        rows.append(dict(k=np.fromstring(point.findtext('k_point'), sep=' ').tolist(),
                         vbm_Ha=float(eig[occupied-1]), cbm_Ha=float(eig[occupied])))
    if not rows or len(rows) != int(band.findtext('nks')):
        raise ValueError('Incomplete SCF mesh')
    gap = (min(row['cbm_Ha'] for row in rows) - max(row['vbm_Ha'] for row in rows)) * 27.211386245988
    return dict(source=str(xml), source_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                occupied_bands=occupied, bands=nbnd, symmetry_reduced_points=len(rows),
                energy_Ha=values['energy_Ha'], mesh_gap_eV=gap,
                threshold_eV=threshold_ev, passed=gap > threshold_ev, rows=rows,
                scope='This QE geometry and SCF mesh only; no continuous-Brillouin-zone, '
                      'D3FT, parameter-variation or experimental-gap certificate')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('point', type=Path)
    ap.add_argument('--threshold-ev', type=float, default=1e-5)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not np.isfinite(args.threshold_ev) or args.threshold_ev <= 0:
        ap.error('Threshold must be finite and positive')
    result = inspect(args.point, args.threshold_ev)
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: val for key, val in result.items() if key != 'rows'}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
