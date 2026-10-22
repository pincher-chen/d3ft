"""Expand published Pna21 sites and freeze SG15 inputs for response_design pilots."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
import spglib

ROOT = Path(__file__).resolve().parents[3]
BOHR = 0.529177210903
SOURCE = 'https://arxiv.org/abs/1705.01515'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source-pdf', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pp = args.output / 'pseudopotentials'
    pp.mkdir()
    library = args.reference / 'pseudopotentials/sg15_oncv_upf_2020-02-06'
    pp_records = {}
    for symbol in ['Mg', 'Si', 'N', 'Ge']:
        source = library / f'{symbol}_ONCV_PBE-1.2.upf'
        target = pp / f'{symbol}.upf'
        shutil.copyfile(source, target)
        header = ET.parse(target).getroot().find('PP_HEADER').attrib
        pp_records[symbol] = dict(source=str(source.resolve()), sha256=sha(target),
                                  header=header)
        assert header['element'].strip() == symbol
        assert header['pseudo_type'].strip().upper() == 'NC'
        assert header.get('has_so', 'F').strip().upper() in ('F', 'FALSE')
    structures = {
        'MgSiN2': dict(lengths=[5.311, 6.495, 5.028], species=['Mg','Si','N','N'],
                       sites=[[.0849,.6228,.9886],[.0700,.1254,0.],
                              [.0485,.0957,.3471],[.1093,.6553,.4102]], table='I'),
        'MgGeN2': dict(lengths=[5.549, 6.658, 5.223], species=['Mg','Ge','N','N'],
                       sites=[[.0851,.6234,.9929],[.0736,.1258,0.],
                              [.0604,.1073,.3600],[.1002,.6431,.3975]], table='II'),
    }
    result = {}
    for name, spec in structures.items():
        atoms = crystal(spec['species'], basis=spec['sites'], spacegroup=33,
                        setting=1, cellpar=spec['lengths']+[90,90,90])
        assert len(atoms) == 16
        counts = Counter(atoms.get_chemical_symbols())
        assert counts['Mg'] == 4 and counts['N'] == 8
        assert counts['Si' if name == 'MgSiN2' else 'Ge'] == 4
        data = dict(name=name, cell_angstrom=atoms.cell.array.tolist(),
                    fractional_positions=atoms.get_scaled_positions().tolist(),
                    symbols=atoms.get_chemical_symbols(), spin=0,
                    source=dict(url=SOURCE, doi='10.1063/1.4985775',
                                table=spec['table'], row='PBE',
                                source_pdf_sha256=sha(args.source_pdf),
                                note='Published PAW-PBE starting geometry, not a converged SG15 result.'),
                    asymmetric_sites=spec)
        path = args.output / f'{name}.json'
        path.write_text(json.dumps(data, indent=2)+'\n')
        xyz = args.output / f'{name}.xyz'
        write(xyz, atoms, format='extxyz')
        loaded = read(xyz)
        assert np.allclose(loaded.cell, atoms.cell, atol=1e-10, rtol=0)
        assert np.allclose(loaded.positions, atoms.positions, atol=1e-8, rtol=0)
        distance = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distance, np.inf)
        assert distance.min() > 1.0
        dataset = spglib.get_symmetry_dataset((atoms.cell.array,
                    atoms.get_scaled_positions(), atoms.numbers), symprec=1e-4)
        assert dataset.number == 33
        nval = sum(float(pp_records[s]['header']['z_valence']) for s in data['symbols'])
        assert int(nval) % 2 == 0
        result[name] = dict(atoms=len(atoms), counts=dict(counts),
                           valence_electrons=nval, occupied_bands=int(nval)//2,
                           spacegroup_number=int(dataset.number),
                           minimum_distance_A=float(distance.min()), structure_sha256=sha(path))
    # Keep the exact old Si input potential for the general-driver regression.
    a = 5.4774657063939625
    cell = np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]])
    si = dict(name='Si_control', cell_angstrom=cell.tolist(),
              fractional_positions=[[0.,0.,0.],[.25,.25,.25]],
              symbols=['Si','Si'], spin=0,
              source=dict(kind='existing cubic worker control', a0_A=a))
    (args.output/'Si_control.json').write_text(json.dumps(si,indent=2)+'\n')
    legacy_si = args.reference/'pseudopotentials/sg15_pbe/si/Si.upf'
    legacy_pp = args.output/'si_control_pseudopotentials'
    legacy_pp.mkdir()
    shutil.copyfile(legacy_si, legacy_pp/'Si.upf')
    result['Si_control'] = dict(atoms=2, a0_A=a,
        legacy_pseudopotential_sha256=sha(legacy_si),
        sg15_v12_matches_legacy=sha(legacy_si)==pp_records['Si']['sha256'])
    shutil.copyfile(args.source_pdf,args.output/'source_arxiv1705.01515.pdf')
    manifest=dict(source=SOURCE, structure_checks=result, pseudopotentials=pp_records,
                  preparation_script_sha256=sha(Path(__file__)), status='input_preparation_only')
    (args.output/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['structure_checks'],indent=2),flush=True)


if __name__ == '__main__':
    main()
